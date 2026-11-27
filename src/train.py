"""Independent deterministic sequence behavior cloning for coverage or tracking.

``train_policy(task, dataset_dir, output_dir, tf_log_dir, *, epochs=100,
batch_size=32, sequence_length=32, learning_rate=3e-4, weight_decay=1e-5,
grad_clip=1., validation_fraction=.2, seed=0, hidden_size=128, device='cpu',
num_threads=1, model_config=None, metadata=None, normalize_vectors=True,
normalization_min_std=.1, initial_checkpoint=None,
correction_weight_max=1., correction_threshold=.1, dagger_recovery_min_steps=0,
split_reference_checkpoint=None, field_steering_loss_weight=0.)``

An episode begins with zero hidden state. Its contiguous TBPTT chunks carry
``hidden.detach()``. Episodes, never individual fragments, are shuffled. Loss is
mean squared action distance per valid step; field-v2 includes normalized pan.
Action_MSE and Action_MAE divide by the actual number of valid components.
Neither is an accuracy or likelihood. Optional field steering supervision adds
an explicitly recorded execution-aware term to Loss, retaining the raw action
metrics and labels. Validation starts at the true episode start.
"""

import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import (FIELD_ACTION_SCHEMA, FIELD_OBSERVATION_SCHEMA, SequenceDataset,
                       pad_episodes, read_episode_metadata, split_episodes)
from .policy_coverage import CoveragePolicy
from .policy_tracking import TrackingPolicy
from .dynamics import GRAVITY


def create_policy(task, config=None, **kwargs):
    """Create fresh independent task weights; config contains model arguments only."""
    options = dict(config or {})
    options.update(kwargs.pop("model_config", None) or {})
    options.update(kwargs)
    if options.pop("potential_field", False):
        from .policy_field import FieldPolicy
        # Legacy experiment flags cannot reinterpret the explicit field_frame
        # architecture or select the unrelated legacy coverage map encoder.
        for key in ["heading_residual", "action_frame", "feature_frame", "ego_map_channels",
                    "ego_map_sigma", "ego_map_mode"]:
            options.pop(key, None)
        return FieldPolicy(task, **options)
    if task == "coverage":
        return CoveragePolicy(**options)
    if task == "tracking":
        return TrackingPolicy(**options)
    raise ValueError("task must be coverage or tracking")


def _fit_vector_normalization(dataset, model, minimum_std=.1):
    # Parallel-variance combination uses only training observations and avoids
    # decompressing large map arrays while estimating scalar-feature statistics.
    count = 0
    mean = np.zeros(dataset.vector_dim, np.float64)
    squared_deviations = np.zeros_like(mean)
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            vector = np.asarray(data["vector"][:-1], dtype=np.float64)
        if vector.ndim != 2 or vector.shape[1] != dataset.vector_dim or len(vector) == 0:
            raise ValueError(f"episode {path} has invalid training vector shape")
        if not np.isfinite(vector).all():
            raise ValueError(f"episode {path} contains nonfinite observations")
        vector = model.preprocess_vector(torch.as_tensor(vector, dtype=torch.float32)).numpy().astype(np.float64)
        local_count = len(vector)
        local_mean = vector.mean(axis=0)
        local_m2 = ((vector - local_mean) ** 2).sum(axis=0)
        delta = local_mean - mean
        combined_count = count + local_count
        squared_deviations += local_m2 + delta ** 2 * count * local_count / combined_count
        mean += delta * local_count / combined_count
        count = combined_count
    std = np.sqrt(squared_deviations / count)
    constant = std < 1e-6
    # Already physically scaled observations can have very narrow nominal-orbit
    # variance. Avoid magnifying a legitimate ship turn into a 40-sigma input.
    std = np.maximum(std, minimum_std)
    std[constant] = 1.0  # constant physical parameters must not amplify new values
    return mean.astype(np.float32), std.astype(np.float32)


def _training_sample_order(dataset, minimum_steps=0):
    """Replay complete accepted DAgger prefixes; never join unrelated histories.

    Splitting and normalization still use unique episodes. Only training draws
    change; each repeated episode starts with zero hidden state in _run_epoch.
    The budget is a minimum number of action exposures, not synthetic steps.
    """
    if isinstance(minimum_steps, (bool, np.bool_)) or not isinstance(minimum_steps, (int, np.integer)) or minimum_steps < 0:
        raise ValueError("dagger_recovery_min_steps must be a nonnegative integer")
    order = []
    repeated = []
    total_steps = 0
    if minimum_steps == 0:
        return np.arange(len(dataset)), dict(unique_episodes=len(dataset), episode_draws=len(dataset),
                                           expected_action_steps=None, repeated_episodes=[])
    for index, path in enumerate(dataset.paths):
        metadata = read_episode_metadata(path)
        with np.load(path, allow_pickle=False) as arrays:
            length = len(arrays["actions"])
        if length < 1:
            raise ValueError(f"episode {path} has no action labels")
        recovery = metadata.get("behavior") == "dagger_training" and metadata.get("accepted_prefix") is True
        copies = max(1, math.ceil(minimum_steps / length)) if recovery else 1
        order.extend([index] * copies)
        total_steps += length * copies
        if copies > 1:
            repeated.append(dict(path=str(path), action_steps=length, copies=copies,
                                 expected_action_steps=length * copies))
    return np.asarray(order, dtype=np.int64), dict(unique_episodes=len(dataset), episode_draws=len(order),
                                                  expected_action_steps=total_steps, repeated_episodes=repeated)


def _field_steering_loss(predicted, target, vector):
    """Per-step excess BCE against true requested bank from raw field inputs.

    omega*tau is a positive gradient-surrogate scale, not an exact bank model.
    The teacher probability uses the exact saturated physical bank request.
    Subtracting detached teacher BCE makes perfect raw imitation cost zero.
    """
    if vector.shape[-1] not in (111, 134):
        raise ValueError("field steering loss requires obs-field-v2")
    raw=vector.to(dtype=predicted.dtype)
    start=98 if raw.shape[-1]==111 else 106
    eps=1e-6
    speed=(raw[...,start]*(220/3.6)).clamp_min(eps)
    radius=(raw[...,start+1]*3000.).clamp_min(eps)
    max_bank=(raw[...,start+2]*math.pi).clamp(eps,math.pi/2-eps)
    tau=raw[...,start+5].clamp_min(eps)
    limit=torch.minimum(max_bank,torch.atan(speed.square()/(GRAVITY*radius))).clamp_min(eps)
    omega=GRAVITY*torch.tan(limit)/speed
    scale=(omega*tau).clamp_min(eps)
    heading=raw[...,2:4]
    def sine(action):
        cross=heading[...,0]*action[...,1]-heading[...,1]*action[...,0]
        normalizer=(heading.norm(dim=-1)*action[...,:2].norm(dim=-1)).clamp_min(eps)
        return cross/normalizer
    with torch.no_grad():
        cross=heading[...,0]*target[...,1]-heading[...,1]*target[...,0]
        dot=(heading*target[...,:2]).sum(-1)
        delta=torch.remainder(torch.atan2(cross,dot)+math.pi,2*math.pi)-math.pi
        rate=(delta/tau).clamp(-omega,omega)
        bank=torch.atan(speed*rate/GRAVITY)/limit
        probability=((bank+1)/2).clamp(0,1)
        baseline=torch.nn.functional.binary_cross_entropy_with_logits(
            2*sine(target)/scale,probability,reduction='none')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(
        2*sine(predicted)/scale,probability,reduction='none')
    return torch.relu(loss-baseline)


def _run_epoch(model, dataset, batch_size, sequence_length, device, optimizer=None,
               grad_clip=1., order=None, correction_weight_max=1., correction_threshold=.1,
               field_steering_loss_weight=0.):
    """Encode map chunks only, carrying causal state across contiguous chunks."""
    training = optimizer is not None
    if not math.isfinite(field_steering_loss_weight) or field_steering_loss_weight<0:
        raise ValueError("invalid field_steering_loss_weight")
    if field_steering_loss_weight and not getattr(model,'model_config',{}).get('potential_field',False):
        raise ValueError("field steering loss is supported only by field models")
    model.train(training)
    indices = list(range(len(dataset))) if order is None else order
    squared_sum = absolute_sum = 0.
    objective_sum = objective_weight_sum = 0.
    valid_steps = 0
    valid_components = 0
    for offset in range(0, len(indices), batch_size):
        episodes = [dataset[index] for index in indices[offset:offset + batch_size]]
        maximum = max(len(episode["actions"]) for episode in episodes)
        hidden = None
        for start in range(0, maximum, sequence_length):
            # Only the current chunk is padded/transferred/encoded. There is no
            # repeated prefix burn-in or autograd graph for the full episode.
            chunks = [{key: value[start:start + sequence_length]
                       for key, value in episode.items()} for episode in episodes]
            batch = {key: value.to(device) for key, value in pad_episodes(chunks).items()}
            observations = {key: value for key, value in batch.items()
                            if key not in {"actions", "mask"}}
            with torch.set_grad_enabled(training):
                predicted, hidden = model(observations, hidden)
                mask = batch["mask"].unsqueeze(-1)
                error = predicted - batch["actions"]
                square = (error.square() * mask).sum()
                step_count = int(batch["mask"].sum().item())
                per_step=error.square().sum(-1)
                if field_steering_loss_weight:
                    valid=batch['mask'].bool()
                    auxiliary=torch.zeros_like(per_step)
                    auxiliary[valid]=_field_steering_loss(
                        predicted[valid],batch['actions'][valid],batch['vector'][valid])
                    per_step=per_step+field_steering_loss_weight*auxiliary
                if training and correction_weight_max > 1:
                    # This is the previous command in raw common observations,
                    # not current heading, executed velocity or normalized MLP input.
                    correction = torch.linalg.vector_norm(
                        batch["actions"][..., :2] - batch["vector"][..., 8:10], dim=-1) > correction_threshold
                    weights = torch.ones_like(batch["mask"], dtype=error.dtype)
                    weights = torch.where(correction, weights * correction_weight_max, weights)
                    weights = weights * batch["mask"]
                    objective = (per_step * weights).sum()
                    weight_count = weights.sum()
                    loss = objective / weight_count
                else:
                    # Preserve the original graph and arithmetic when disabled;
                    # validation and initial epoch 0 always take this branch.
                    objective=(per_step*batch['mask']).sum() if field_steering_loss_weight else square
                    weight_count=step_count
                    loss = objective / step_count
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            hidden = hidden.detach()
            squared_sum += float(square.detach().item())
            absolute_sum += float((error.detach().abs() * mask).sum().item())
            objective_sum += float(objective.detach().item())
            objective_weight_sum += float(weight_count.detach().item() if torch.is_tensor(weight_count) else weight_count)
            valid_steps += step_count
            valid_components += step_count * error.shape[-1]
    if valid_steps == 0:
        raise ValueError("dataset contains no usable action steps")
    return dict(loss=objective_sum / objective_weight_sum,
                action_mse=squared_sum / valid_components,
                action_mae=absolute_sum / valid_components, steps=valid_steps)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _preserve_scene_assignments(split, previous, validation_fraction, seed):
    """Keep every historical scene group fixed; split only genuinely new groups."""
    if not isinstance(previous, dict):
        raise ValueError("warm-start checkpoint is missing historical scene group split metadata")
    fixed = previous.get("fixed_scene_groups", {})
    training = set(map(str, previous.get("train_scene_group_ids", previous.get("train_scene_ids", []))))
    validation = set(map(str, previous.get("validation_scene_group_ids", previous.get("validation_scene_ids", []))))
    training.update(map(str, fixed.get("train", [])))
    validation.update(map(str, fixed.get("validation", [])))
    if not training and not validation:
        raise ValueError("warm-start checkpoint has no historical scene group assignments")
    if training & validation:
        raise ValueError("historical training and validation scene groups overlap")
    train_paths, val_paths = set(previous.get("train_paths", [])), set(previous.get("validation_paths", []))
    if train_paths & val_paths:
        raise ValueError("historical training and validation episode paths overlap")
    aliases = {}
    # Scene aliases also handle old checkpoints whose split used scene_id before
    # scene_group_id was added to metadata. Conflicting layout histories reject.
    for record in previous.get("episodes", []):
        group = str(record.get("scene_group_id", record.get("scene_id")))
        scene = str(record.get("scene_id", group))
        memberships = set()
        if group in training or scene in training or record.get("path") in train_paths:
            memberships.add("train")
        if group in validation or scene in validation or record.get("path") in val_paths:
            memberships.add("validation")
        if len(memberships) > 1:
            raise ValueError("historical scene group has conflicting training/validation assignments")
        if memberships:
            subset = next(iter(memberships))
            if scene in aliases and aliases[scene] != subset:
                raise ValueError("historical scene alias has conflicting assignments")
            aliases[scene] = subset
            (training if subset == "train" else validation).add(group)
    if training & validation:
        raise ValueError("historical layout groups overlap after scene alias resolution")
    groups = {}
    memberships = {}
    for record in split["episodes"]:
        group = str(record.get("scene_group_id", record["scene_id"]))
        groups.setdefault(group, []).append(record["path"])
        assignment = memberships.setdefault(group, set())
        if group in training:
            assignment.add("train")
        if group in validation:
            assignment.add("validation")
        if str(record["scene_id"]) in aliases:
            assignment.add(aliases[str(record["scene_id"])])
        if len(assignment) > 1:
            raise ValueError("warm-start would mix historical training and validation in one scene group")
    current_train = {group for group, subset in memberships.items() if "train" in subset}
    current_val = {group for group, subset in memberships.items() if "validation" in subset}
    new_groups = sorted(set(groups) - current_train - current_val)
    np.random.default_rng(seed).shuffle(new_groups)
    target_val_count = 0 if validation_fraction == 0 else max(1, round(len(groups) * validation_fraction))
    new_val_count = min(len(new_groups), max(0, target_val_count - len(current_val)))
    if not current_train:
        new_val_count = min(new_val_count, max(0, len(new_groups) - 1))
    current_val.update(new_groups[:new_val_count])
    current_train.update(new_groups[new_val_count:])
    if not current_train:
        raise ValueError("warm-start has no training scenes; historical validation scenes cannot become training")
    if current_train & current_val or current_train & validation or current_val & training:
        raise ValueError("warm-start scene assignments would cross the historical split")
    result = dict(split)
    result.update(validation_fraction=float(validation_fraction),
                  train_scene_ids=sorted(current_train), validation_scene_ids=sorted(current_val),
                  train_scene_group_ids=sorted(current_train), validation_scene_group_ids=sorted(current_val),
                  train_paths=sorted(path for group in current_train for path in groups[group]),
                  validation_paths=sorted(path for group in current_val for path in groups[group]),
                  assignment_policy="preserve_checkpoint_scene_groups",
                  fixed_scene_groups={"train": sorted(training | current_train),
                                      "validation": sorted(validation | current_val)})
    return result


def train_policy(task, dataset_dir, output_dir, tf_log_dir, *, epochs=100, batch_size=32,
                 sequence_length=32, learning_rate=3e-4, weight_decay=1e-5,
                 grad_clip=1., validation_fraction=.2, seed=0, hidden_size=128,
                 device="cpu", num_threads=1, model_config=None, metadata=None,
                 normalize_vectors=True, normalization_min_std=.1, initial_checkpoint=None,
                 correction_weight_max=1., correction_threshold=.1, dagger_recovery_min_steps=0,
                 split_reference_checkpoint=None, field_steering_loss_weight=0.):
    """Train one task, write six meaningful TensorBoard metrics and best/last files.

    ``model_config`` accepts ``vector_dim``, ``hidden_size`` and
    ``heading_residual``, ``action_frame`` ('body' by default or 'world') and
    ``feature_frame`` ('body' by default; coverage retains its world map heading).
    ``potential_field=True`` selects field-v2's world-aligned CNN/context/GRU
    and three-component actions. Legacy frame/map flags are ignored in that
    mode. Its previous-command mask defaults True and can be set explicitly.
    Coverage also accepts ego_map_channels=False and ego_map_sigma=2 grid cells.
    A 3→6 map-channel warm-start requires explicit upgrade_coverage_checkpoint
    conversion; ordinary warm-start rejects incompatible architecture settings.
    A hidden_size in model_config overrides the standalone
    argument. Extra experiment/physical configuration goes in ``metadata`` and is
    saved unchanged alongside representative dataset metadata. Setting validation
    fraction to zero explicitly disables validation; best then uses training loss,
    validation results are None, and no validation scalars are fabricated. With
    validation disabled best selection uses unweighted training Action_MSE;
    when training draws repeat episodes, candidates are evaluated on the unique
    training episodes after updates so epoch-zero and later scores are comparable.
    normalization_min_std floors nonconstant train standard deviations in the
    physically scaled feature space; 0 recovers ordinary feature standardization.
    initial_checkpoint warm-starts model weights and normalization with a fresh
    Adam optimizer. Its architecture is inherited; any explicitly supplied model
    config must agree. Warm-start never refits normalization, even when the new
    dataset or normalization settings differ, preserving the learned coordinates.
    Historical scene groups keep their original subset regardless of a new seed
    or validation fraction; only new groups are split. The exact input checkpoint
    is saved once under initial-checkpoints/<sha256>.pt before output replacement.
    split_reference_checkpoint applies only its historical scene assignments to
    a fresh architecture; it never loads weights or normalization. Its task and
    public protocol must agree. With validation_fraction=0 all new groups become
    training groups while historical validation stays fixed. The source SHA256
    and a split-only JSON snapshot are saved for reproducibility.
    During optimizer updates only, a valid step has weight correction_weight_max
    when norm(expert_action - raw_vector[8:10]) > correction_threshold, otherwise
    weight 1. Loss is the weighted squared XY vector distance divided by valid
    weight sum. The default max=1 exactly disables weighting. Initial epoch 0,
    all validation metrics, and Train/Action_MSE/MAE remain unweighted.
    dagger_recovery_min_steps repeats complete accepted DAgger prefixes to a
    minimum training exposure budget. Each draw resets recurrent state; unique
    episodes determine normalization and splits. Epoch zero and validation do
    not resample. Training metrics describe the actual training draws.
    field_steering_loss_weight adds field-only excess soft-target BCE using the
    actual requested teacher bank from observed physical parameters. Its positive
    omega_max*heading_tau scale is a gradient surrogate, not an exact bank model.
    Correction weights multiply the combined action/steering objective; initial
    and validation Loss use the same objective without correction weighting.
    Raw three-component Action_MSE/MAE and original labels remain unchanged.
    The default zero preserves the previous loss arithmetic exactly.

    Returns JSON-safe initial/best validation MSE, initial validation metrics,
    checkpoint paths, history (epoch/Train/Validation), split and train config.
    The initial checkpoint is a candidate for best, so a worsening run preserves
    its actual best model instead of claiming its final epoch improved.
    """
    for name, value in [("epochs", epochs), ("batch_size", batch_size),
                        ("sequence_length", sequence_length), ("hidden_size", hidden_size),
                        ("num_threads", num_threads)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value, minimum, strict in [("learning_rate", learning_rate, 0, True),
                                        ("weight_decay", weight_decay, 0, False),
                                        ("normalization_min_std", normalization_min_std, 0, False),
                                        ("correction_weight_max", correction_weight_max, 1, False),
                                        ("correction_threshold", correction_threshold, 0, False),
                                        ("field_steering_loss_weight", field_steering_loss_weight, 0, False),
                                        ("grad_clip", grad_clip, 0, True)]:
        if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
            raise ValueError(f"invalid {name}")
    if tf_log_dir is None or not str(tf_log_dir).strip():
        raise ValueError("tf_log_dir is required; TensorBoard logging cannot be silently disabled")
    torch.set_num_threads(num_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = torch.device(device)
    output_dir, tf_log_dir = Path(output_dir), Path(tf_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf_log_dir.mkdir(parents=True, exist_ok=True)
    initial_state = None
    initial_path = None
    initial_requested_path = None
    initial_sha256 = None
    if initial_checkpoint is not None:
        initial_requested_path = Path(initial_checkpoint).resolve()
        if not initial_requested_path.is_file():
            raise ValueError(f"initial checkpoint is missing: {initial_requested_path}")
        initial_bytes = initial_requested_path.read_bytes()
        initial_sha256 = hashlib.sha256(initial_bytes).hexdigest()
        initial_state = torch.load(io.BytesIO(initial_bytes), map_location="cpu", weights_only=False)
        checkpoint_field = initial_state.get("checkpoint_version") == "bc-field-v2"
        if initial_state.get("checkpoint_version") not in {"bc-v1", "bc-field-v2"}:
            raise ValueError("initial checkpoint has an incompatible checkpoint version")
        expected_schema = FIELD_OBSERVATION_SCHEMA if checkpoint_field else "obs-v1"
        if initial_state.get("observation_schema") != expected_schema:
            raise ValueError("initial checkpoint observation schema is incompatible")
        if checkpoint_field and initial_state.get("action_schema") != FIELD_ACTION_SCHEMA:
            raise ValueError("initial checkpoint action schema is incompatible")
        if bool(initial_state.get("model_config", {}).get("potential_field", False)) != checkpoint_field:
            raise ValueError("initial checkpoint architecture and schema protocol disagree")
        if initial_state.get("task") != task:
            raise ValueError("initial checkpoint task does not match training task")
        options = dict(initial_state["model_config"])
        if not checkpoint_field:
            options.setdefault("action_frame", "world")
            options.setdefault("feature_frame", "world")
        for key, value in (model_config or {}).items():
            if checkpoint_field and key in {"heading_residual", "action_frame", "feature_frame", "ego_map_channels",
                                             "ego_map_sigma", "ego_map_mode"}:
                continue
            if key not in options or options[key] != value:
                raise ValueError(f"initial checkpoint model config mismatch for {key}")
        initial_path = (output_dir / "initial-checkpoints" / f"{initial_sha256}.pt").resolve()
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with initial_path.open("xb") as snapshot:
                snapshot.write(initial_bytes)
        except FileExistsError:
            if hashlib.sha256(initial_path.read_bytes()).hexdigest() != initial_sha256:
                raise ValueError("immutable input checkpoint snapshot has been modified")
    else:
        options = dict(hidden_size=hidden_size)
        options.update(model_config or {})
    field_mode = bool(options.get("potential_field", False))
    if field_steering_loss_weight and not field_mode:
        raise ValueError("field steering loss is supported only by field models")
    observation_schema = FIELD_OBSERVATION_SCHEMA if field_mode else "obs-v1"
    action_schema = FIELD_ACTION_SCHEMA if field_mode else "action-v1"
    reference_split = None
    reference_path = reference_sha256 = reference_snapshot = None
    if split_reference_checkpoint is not None:
        if initial_state is not None:
            raise ValueError("split reference is for fresh training; warm-start already preserves its own split")
        reference_path = Path(split_reference_checkpoint).resolve()
        reference_bytes = reference_path.read_bytes()
        reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
        reference = torch.load(io.BytesIO(reference_bytes), map_location="cpu", weights_only=False)
        expected_version = "bc-field-v2" if field_mode else "bc-v1"
        if (reference.get("task") != task or reference.get("checkpoint_version") != expected_version or
                reference.get("observation_schema") != observation_schema):
            raise ValueError("split reference task or observation protocol does not match fresh training")
        if field_mode and reference.get("action_schema") != FIELD_ACTION_SCHEMA:
            raise ValueError("split reference action protocol does not match field training")
        reference_split = reference.get("dataset_split")
        if not isinstance(reference_split, dict) or not reference_split.get("train_paths"):
            raise ValueError("split reference requires complete historical scene assignments")
        reference_snapshot = output_dir / "split-references" / f"{reference_sha256}.json"
        reference_snapshot.parent.mkdir(parents=True, exist_ok=True)
        _write_json(reference_snapshot, dict(checkpoint=str(reference_path), sha256=reference_sha256,
                                            dataset_split=reference_split))
    split = split_episodes(dataset_dir, task, validation_fraction=0 if initial_state is not None or reference_split is not None else validation_fraction,
                           seed=seed, observation_schema=observation_schema)
    if initial_state is not None:
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        split = _preserve_scene_assignments(split, initial_state.get("dataset_split"), validation_fraction, seed)
    elif reference_split is not None:
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        split = _preserve_scene_assignments(split, reference_split, validation_fraction, seed)
    train_dataset = SequenceDataset(split["train_paths"], task=task, observation_schema=observation_schema)
    validation_dataset = SequenceDataset(split["validation_paths"], task=task, observation_schema=observation_schema)
    training_order, sampling_report = _training_sample_order(train_dataset, dagger_recovery_min_steps)
    model = create_policy(task, options).to(target_device)
    if model.vector_dim != train_dataset.vector_dim:
        raise ValueError("model vector dimension does not match the task observation schema")
    if initial_state is not None:
        try:
            model.load_state_dict(initial_state["model_state_dict"])
        except RuntimeError as error:
            raise ValueError(f"initial checkpoint weights do not match model config: {error}") from error
    elif normalize_vectors:
        mean, std = _fit_vector_normalization(train_dataset, model, minimum_std=normalization_min_std)
        model.set_vector_normalization(mean, std)
    _write_json(output_dir / "dataset_split.json", split)
    for subset, key in [("train", "train_paths"), ("val", "validation_paths")]:
        _write_json(output_dir / f"{subset}_manifest.json",
                    dict(task=task, paths=split[key], seed=seed, split_unit=split["split_unit"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_config = dict(epochs=epochs, batch_size=batch_size, sequence_length=sequence_length,
                        learning_rate=float(learning_rate), weight_decay=float(weight_decay),
                        grad_clip=float(grad_clip), validation_fraction=float(validation_fraction),
                        seed=int(seed), hidden_size=model.hidden_size, device=str(target_device),
                        num_threads=num_threads, normalize_vectors=bool(normalize_vectors),
                        action_frame=model.action_frame, feature_frame=model.feature_frame)
    train_config["normalization_min_std"] = float(normalization_min_std)
    train_config["initial_checkpoint"] = None if initial_path is None else str(initial_path)
    train_config["initial_checkpoint_sha256"] = initial_sha256
    train_config["initial_checkpoint_requested"] = None if initial_requested_path is None else str(initial_requested_path)
    train_config["split_reference_checkpoint"] = None if reference_path is None else str(reference_path)
    train_config["split_reference_checkpoint_sha256"] = reference_sha256
    train_config["split_reference_snapshot"] = None if reference_snapshot is None else str(reference_snapshot.resolve())
    train_config["correction_weight_max"] = float(correction_weight_max)
    train_config["correction_threshold"] = float(correction_threshold)
    train_config["dagger_recovery_min_steps"] = int(dagger_recovery_min_steps)
    train_config["field_steering_loss_weight"] = float(field_steering_loss_weight)
    train_config["training_sampling"] = sampling_report
    loss_definition = dict(
        training=("sum(valid * weight * squared_world_xy_and_normalized_pan_error) / sum(valid * weight)"
                  if field_mode else "sum(valid * weight * squared_world_xy_vector_error) / sum(valid * weight)"),
        correction="norm(expert_action_xy - raw_vector[8:10]) > correction_threshold",
        weight="correction_weight_max on correction steps; otherwise 1",
        previous_command="raw observed last command, not actual velocity",
        epoch_zero="unweighted", validation="unweighted", action_mse_mae="unweighted",
        training_sampling="complete accepted DAgger prefix repeats; independent recurrent resets",
        evaluation_sampling="unique original episodes, no repeats")
    if field_steering_loss_weight:
        loss_definition['field_steering']=(
            'relu(BCEWithLogits(2*sin(delta_pred)/(omega_max*heading_tau), '
            '(true_requested_teacher_bank/bank_limit+1)/2) - detached teacher BCE baseline)')
        loss_definition['training']='sum(valid * correction_weight * (squared_three_component_error + field_steering_loss_weight * field_steering)) / sum(valid * correction_weight)'
        loss_definition['auxiliary_physics']='raw same-observation heading/speed/radius/max_bank/heading_tau; positive omega*tau gradient surrogate, not exact bank prediction; valid steps only'
        loss_definition['epoch_zero']='unweighted combined BC + field steering objective'
        loss_definition['validation']='unweighted combined BC + field steering objective'
    train_config["loss_definition"] = loss_definition
    train_config.update(observation_schema=observation_schema, action_schema=action_schema,
                        action_dim=train_dataset.action_dim)
    normalization = dict(method="train_feature_standardization" if normalize_vectors else "identity",
                         fit_subset="train", vector_mean=model.vector_mean.cpu().tolist(),
                         vector_std=model.vector_std.cpu().tolist(),
                         coverage_map="observation-scaled numerical channels; no image normalization",
                         action_frame=model.action_frame,
                         feature_frame=model.feature_frame,
                         minimum_std=float(normalization_min_std), constant_feature_std=1.,
                         direction=("rotate_by_heading(normalize(tanh(head+[atanh(0.5),0])))"
                                    if model.heading_residual else "rotate_by_heading(normalize(tanh(head)))")
                         if model.action_frame == "body" else (
                             "normalize(tanh(head+atanh(0.5*heading)))" if model.heading_residual
                             else "normalize(tanh(head))"), zero_fallback="vector[2:4]")
    if field_mode:
        normalization.update(local_field="physical signed field and semantic masks; no image normalization",
                             field_context="physical scales defined by obs-field-v2",
                             pan=("tanh(pan_head + atanh(clamp(observed_actual_pan/task_pan_limit,-.99,.99)))"
                                  if model.pan_residual else "tanh(pan_head)"),
                             field_frame=model.field_frame)
        normalization["pan_reference"] = model.pan_reference
        normalization["direction_reference"] = model.direction_reference
        if model.direction_reference == "goal_orbit":
            normalization["direction"] = "normalize(tanh(direction_head + atanh(0.5*public_goal_orbit_reference))) rotated to world"
        if model.pan_residual and model.pan_reference == "goal":
            normalization["pan"] = "tanh(pan_head + atanh(clamp(observed_goal_bearing/task_pan_limit,-.99,.99)))"
    if initial_state is not None:
        saved_normalization = initial_state.get("normalization", {})
        normalization.update(method=saved_normalization.get("method", "checkpoint_buffers"),
                             minimum_std=saved_normalization.get("minimum_std"),
                             fit_subset="initial_checkpoint", source_checkpoint=str(initial_path),
                             source_checkpoint_sha256=initial_sha256,
                             source_fit_subset=saved_normalization.get("fit_subset"),
                             normalization_refitted=False)
    dataset_metadata = read_episode_metadata(train_dataset.paths[0])
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    history = []
    initial_validation = (_run_epoch(model, validation_dataset, batch_size, sequence_length, target_device,
                                    field_steering_loss_weight=field_steering_loss_weight)
                          if len(validation_dataset) else None)
    initial_train = _run_epoch(model, train_dataset, batch_size, sequence_length, target_device,
                               field_steering_loss_weight=field_steering_loss_weight)
    best_mse = (initial_validation or initial_train)["action_mse"]
    best_epoch = 0

    def save_checkpoint(path, epoch, train_metrics, validation_metrics, selection_metrics=None):
        torch.save(dict(checkpoint_version="bc-field-v2" if field_mode else "bc-v1",
                        observation_schema=observation_schema, action_schema=action_schema, task=task,
                        model_config=model.model_config, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), epoch=epoch, seed=int(seed),
                        normalization=normalization, train_config=train_config,
                        metadata=metadata or {}, dataset_metadata=dataset_metadata,
                        dataset_split=split, train_metrics=train_metrics,
                        loss_definition=loss_definition,
                        validation_metrics=validation_metrics,
                        selection_metrics=selection_metrics or validation_metrics or train_metrics,
                        selection_subset="validation_unique" if len(validation_dataset) else "train_unique",
                        initial_checkpoint=None if initial_path is None else str(initial_path),
                        initial_checkpoint_sha256=initial_sha256,
                        initial_checkpoint_epoch=None if initial_state is None else initial_state.get("epoch"),
                        architecture_conversion=None if initial_state is None else initial_state.get("architecture_conversion")), path)

    save_checkpoint(best_path, 0, initial_train, initial_validation)
    rng = np.random.default_rng(seed)
    with SummaryWriter(log_dir=str(tf_log_dir)) as writer:
        def log_metrics(subset, metrics, epoch):
            for tag, key in [("Loss", "loss"), ("Action_MSE", "action_mse"), ("Action_MAE", "action_mae")]:
                writer.add_scalar(f"{subset}/{tag}", metrics[key], epoch)

        def print_progress(epoch, train_metrics, validation_metrics):
            val_loss = "None" if validation_metrics is None else f"{validation_metrics['loss']:.8g}"
            val_mse = "None" if validation_metrics is None else f"{validation_metrics['action_mse']:.8g}"
            print(f"[{task}] epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.8g} "
                  f"val_loss={val_loss} val_mse={val_mse}", flush=True)

        log_metrics("Train", initial_train, 0)
        if initial_validation is not None:
            log_metrics("Validation", initial_validation, 0)
        print_progress(0, initial_train, initial_validation)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(training_order).tolist()
            train_metrics = _run_epoch(model, train_dataset, batch_size, sequence_length,
                                       target_device, optimizer=optimizer, grad_clip=grad_clip, order=order,
                                       correction_weight_max=correction_weight_max,
                                       correction_threshold=correction_threshold,
                                       field_steering_loss_weight=field_steering_loss_weight)
            validation_metrics = (_run_epoch(model, validation_dataset, batch_size, sequence_length, target_device,
                                            field_steering_loss_weight=field_steering_loss_weight)
                                  if len(validation_dataset) else None)
            selection_metrics = validation_metrics
            if selection_metrics is None:
                selection_metrics = (_run_epoch(model, train_dataset, batch_size, sequence_length, target_device,
                                               field_steering_loss_weight=field_steering_loss_weight)
                                     if len(training_order) > len(train_dataset) else train_metrics)
            score = selection_metrics["action_mse"]
            if score < best_mse:
                best_mse, best_epoch = score, epoch
                save_checkpoint(best_path, epoch, train_metrics, validation_metrics, selection_metrics)
            save_checkpoint(last_path, epoch, train_metrics, validation_metrics, selection_metrics)
            log_metrics("Train", train_metrics, epoch)
            if validation_metrics is not None:
                log_metrics("Validation", validation_metrics, epoch)
            history.append(dict(epoch=epoch, Train=train_metrics, Validation=validation_metrics))
            writer.flush()
            print_progress(epoch, train_metrics, validation_metrics)
    # A task-named export is a full checkpoint with schema and normalization.
    export_path = output_dir / f"{task}_policy.pt"
    export_path.write_bytes(best_path.read_bytes())
    result = dict(task=task,
                  initial_validation_mse=None if initial_validation is None else initial_validation["action_mse"],
                  best_validation_mse=None if initial_validation is None else best_mse,
                  initial_validation=initial_validation, initial_train=initial_train,
                  best_selection_mse=best_mse, best_epoch=best_epoch,
                  best_checkpoint=str(best_path.resolve()), last_checkpoint=str(last_path.resolve()),
                  exported_checkpoint=str(export_path.resolve()), history=history, split=split,
                  train_config=train_config, model_config=model.model_config,
                  loss_definition=loss_definition,
                  initial_checkpoint=None if initial_path is None else str(initial_path),
                  initial_checkpoint_sha256=initial_sha256,
                  validation_available=initial_validation is not None)
    _write_json(output_dir / "training_history.json", result)
    return result
