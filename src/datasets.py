"""Versioned NPZ episode reading and deterministic splitting by complete scenes.

NPZ stores ``vector[T+1,N]``, optional ``coverage_map[T+1,3,64,64]``,
``actions[T,2]`` and ``executed_actions[T,2]``. Field-v2 instead stores
``potential[T+1,4,33,33]``, ``field_context[T+1,16]``, ``actions[T,3]``
with explicit dataset/observation/action schema metadata. ``metadata`` is JSON.
Only expert command ``actions`` are imitation labels. Whole episodes are loaded
on demand; training processes their contiguous chunks while carrying GRU state.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FIELD_OBSERVATION_SCHEMA = "obs-field-v2"
FIELD_ACTION_SCHEMA = "action-field-v2"
FIELD_DATASET_VERSION = "bc-dataset-field-v2"


def _validate_protocol(metadata, observation_schema, path):
    schema = metadata.get("observation_schema", metadata.get("obs_schema", metadata.get("schema_version", "obs-v1")))
    if schema != observation_schema:
        raise ValueError(f"episode {path} observation schema/protocol {schema} does not match {observation_schema}")
    if observation_schema == FIELD_OBSERVATION_SCHEMA:
        if metadata.get("dataset_version") != FIELD_DATASET_VERSION:
            raise ValueError(f"episode {path} requires dataset version {FIELD_DATASET_VERSION}")
        if metadata.get("action_schema") != FIELD_ACTION_SCHEMA:
            raise ValueError(f"episode {path} action schema must be {FIELD_ACTION_SCHEMA}")
        if "obs_schema" in metadata and metadata["obs_schema"] != observation_schema:
            raise ValueError(f"episode {path} has conflicting observation schemas")


def read_episode_metadata(path):
    with np.load(path, allow_pickle=False) as data:
        key = "metadata" if "metadata" in data else "metadata_json"
        if key not in data:
            raise ValueError(f"episode {path} is missing JSON metadata")
        value = data[key].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError(f"episode {path} metadata must be a JSON object")
    return metadata


def split_episodes(dataset_dir, task, validation_fraction=.2, seed=0, observation_schema="obs-v1"):
    """Return manifests; repeated static-layout groups remain together.

    scene_group_id (layout, independent of initial state) takes precedence over
    scene_id and scene_hash. This keeps variations of one layout in one subset.
    dataset_dir may be an accepted-manifest JSON with complete=true and an
    explicit paths list. That list is authoritative; unlisted stale caches are
    never discovered. Relative paths are relative to the manifest directory.

    A positive validation fraction requires at least two distinct scenes. No
    fallback duplicates a training scene as validation. A zero fraction gives an
    explicitly empty validation set for training-only use.
    """
    if task not in {"coverage", "tracking"}:
        raise ValueError("task must be coverage or tracking")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    source = Path(dataset_dir).resolve()
    source_manifest = None
    if source.is_file() or source.suffix.lower() == ".json":
        if not source.is_file():
            raise ValueError(f"missing accepted manifest {source}")
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            raise ValueError("accepted manifest is incomplete; collection must finish before training")
        if manifest.get("task", task) != task:
            raise ValueError("accepted manifest task does not match training task")
        entries = manifest.get("paths")
        if not isinstance(entries, list) or not entries or not all(isinstance(entry, str) for entry in entries):
            raise ValueError("accepted manifest requires a nonempty paths list")
        paths = []
        seen = set()
        for entry in entries:
            path = Path(entry)
            path = (path if path.is_absolute() else source.parent / path).resolve()
            canonical = os.path.normcase(str(path))
            if canonical in seen:
                raise ValueError(f"duplicate episode path in accepted manifest: {path}")
            seen.add(canonical)
            if not path.is_file() or path.suffix.lower() != ".npz":
                raise ValueError(f"missing NPZ episode from accepted manifest: {path}")
            paths.append(path)
        paths.sort()
        source_manifest = str(source)
    else:
        paths = sorted(source.rglob("*.npz"))
    groups = {}
    records = []
    has_layout_groups = False
    for path in paths:
        metadata = read_episode_metadata(path)
        if metadata.get("task") != task:
            raise ValueError(f"episode {path} task does not match {task}")
        if metadata.get("expert_valid") is False:
            if source_manifest is not None:
                raise ValueError(f"accepted manifest includes an invalid expert episode: {path}")
            continue
        _validate_protocol(metadata, observation_schema, path)
        scene_id = metadata.get("scene_id", metadata.get("scene_hash"))
        group_id = metadata.get("scene_group_id", scene_id)
        if group_id is None:
            raise ValueError(f"episode {path} requires scene_id or scene_hash to prevent scene leakage")
        has_layout_groups |= metadata.get("scene_group_id") is not None
        group_id = str(group_id)
        groups.setdefault(group_id, []).append(str(path))
        records.append(dict(path=str(path), scene_id=str(scene_id) if scene_id is not None else group_id,
                            scene_group_id=group_id,
                            seed=metadata.get("seed"), config_hash=metadata.get("config_hash")))
    if not groups:
        raise ValueError(f"no valid {task} episodes found in {dataset_dir}")
    if validation_fraction > 0 and len(groups) < 2:
        raise ValueError("at least two distinct scenes are needed for independent validation")
    scene_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(scene_ids)
    count = 0 if validation_fraction == 0 else min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_ids = sorted(scene_ids[:count])
    train_ids = sorted(scene_ids[count:])
    return dict(task=task, seed=int(seed), validation_fraction=float(validation_fraction),
                source_manifest=source_manifest,
                split_unit="scene_group_id" if has_layout_groups else "scene_id",
                train_scene_ids=train_ids, validation_scene_ids=validation_ids,
                train_scene_group_ids=train_ids, validation_scene_group_ids=validation_ids,
                train_paths=sorted(path for scene in train_ids for path in groups[scene]),
                validation_paths=sorted(path for scene in validation_ids for path in groups[scene]),
                episodes=records)


class SequenceDataset(Dataset):
    """One item is one whole episode, preserving history and label alignment."""

    def __init__(self, paths, task=None, observation_schema="obs-v1"):
        if isinstance(paths, (str, Path)):
            manifest = json.loads(Path(paths).read_text(encoding="utf-8"))
            paths = manifest.get("paths", manifest.get("train_paths", []))
            task = task or manifest.get("task")
        if task not in {"coverage", "tracking"}:
            raise ValueError("SequenceDataset requires task coverage or tracking")
        self.paths = [Path(path) for path in paths]
        self.task = task
        self.vector_dim = 111 if task == "coverage" else 134
        if observation_schema not in {"obs-v1", FIELD_OBSERVATION_SCHEMA}:
            raise ValueError("unsupported observation schema")
        self.observation_schema = observation_schema
        self.action_dim = 3 if observation_schema == FIELD_OBSERVATION_SCHEMA else 2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        metadata = read_episode_metadata(path)
        if metadata.get("task") != self.task:
            raise ValueError(f"episode {path} has the wrong task")
        _validate_protocol(metadata, self.observation_schema, path)
        with np.load(path, allow_pickle=False) as data:
            if "actions" not in data or "vector" not in data:
                raise ValueError(f"episode {path} is missing vector or actions")
            actions = np.asarray(data["actions"], dtype=np.float32)
            vector = np.asarray(data["vector"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != self.action_dim or len(actions) < 1:
                raise ValueError(f"episode {path} actions must have nonempty shape (T,{self.action_dim})")
            length = len(actions)
            if vector.shape != (length + 1, self.vector_dim):
                raise ValueError(f"episode {path} vector must have shape (T+1,{self.vector_dim})")
            if not np.isfinite(vector).all() or not np.isfinite(actions).all():
                raise ValueError(f"episode {path} observations and labels must be finite")
            norms = np.linalg.norm(actions[:, :2], axis=-1)
            if np.any(np.abs(actions) > 1 + 1e-6) or not np.allclose(norms, 1, atol=1e-4):
                raise ValueError(f"episode {path} expert actions must be unit XY directions")
            result = dict(vector=torch.from_numpy(vector[:-1].copy()),
                          actions=torch.from_numpy(actions.copy()))
            if "executed_actions" in data:
                executed = data["executed_actions"]
                valid_shapes = {(length, self.action_dim)}
                if self.action_dim == 3:
                    valid_shapes.add((length, 2))
                if executed.shape not in valid_shapes or not np.isfinite(executed).all():
                    raise ValueError(f"episode {path} executed_actions has incompatible shape or nonfinite values")
            if self.observation_schema == FIELD_OBSERVATION_SCHEMA:
                for key, shape in [("potential", (4, 33, 33)), ("field_context", (16,))]:
                    if key not in data:
                        raise ValueError(f"episode {path} is missing {key}")
                    array = np.asarray(data[key], dtype=np.float32)
                    if array.shape != (length + 1, *shape) or not np.isfinite(array).all():
                        raise ValueError(f"episode {path} {key} must be finite shape {(length + 1, *shape)}")
                    result[key] = torch.from_numpy(array[:-1].copy())
            elif self.task == "coverage":
                if "coverage_map" not in data:
                    raise ValueError(f"episode {path} is missing coverage_map")
                coverage_map = np.asarray(data["coverage_map"], dtype=np.float32)
                if coverage_map.shape != (length + 1, 3, 64, 64) or not np.isfinite(coverage_map).all():
                    raise ValueError(f"episode {path} coverage_map must be finite shape (T+1,3,64,64)")
                result["coverage_map"] = torch.from_numpy(coverage_map[:-1].copy())
        return result


def pad_episodes(episodes):
    """Pad parallel whole episodes; masks exclude all tail padding from metrics."""
    if not episodes:
        raise ValueError("cannot pad an empty batch")
    lengths = [len(episode["actions"]) for episode in episodes]
    maximum = max(lengths)
    batch = {}
    for key in episodes[0]:
        shape = (len(episodes), maximum, *episodes[0][key].shape[1:])
        batch[key] = torch.zeros(shape, dtype=torch.float32)
        for index, episode in enumerate(episodes):
            batch[key][index, :lengths[index]] = episode[key]
    batch["mask"] = torch.arange(maximum)[None, :] < torch.tensor(lengths)[:, None]
    return batch
