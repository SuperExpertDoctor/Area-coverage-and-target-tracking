"""Independent coverage CNN/MLP/GRU; optional observed UAV raster coordinates."""

import hashlib
import io
import math
from pathlib import Path

import torch
from torch import nn

from .policy_tracking import NeuralPolicy, _DirectionalRecurrentPolicy


class CoveragePolicy(_DirectionalRecurrentPolicy):
    """(3,64,64) map + 111 vector → CNN/MLP → GRU → unit world XY."""

    task = "coverage"

    def __init__(self, vector_dim=111, hidden_size=128, heading_residual=True,
                 action_frame="body", feature_frame="body", ego_map_channels=False,
                 ego_map_sigma=2., ego_map_mode="gaussian"):
        super().__init__(vector_dim, hidden_size, heading_residual, action_frame, feature_frame)
        if ego_map_mode not in {"gaussian", "body_coordinates", "world_coordinates"}:
            raise ValueError("ego_map_mode must be gaussian, body_coordinates or world_coordinates")
        if not math.isfinite(ego_map_sigma) or ego_map_sigma <= 0:
            raise ValueError("ego_map_sigma must be finite and positive (in grid cells)")
        if ego_map_channels and vector_dim < 98:
            raise ValueError("ego map channels require the observed ROI fields")
        self.ego_map_channels = bool(ego_map_channels)
        self.ego_map_sigma = float(ego_map_sigma)
        self.ego_map_mode = ego_map_mode
        self.model_config.update(ego_map_channels=self.ego_map_channels, ego_map_sigma=self.ego_map_sigma,
                                 ego_map_mode=self.ego_map_mode)
        # Nonpersistent coordinate buffers preserve old checkpoint compatibility.
        centers = torch.arange(64, dtype=torch.float32) + .5
        self.register_buffer("_grid_x", centers.view(1, 1, 1, 64), persistent=False)
        self.register_buffer("_grid_y", centers.view(1, 1, 64, 1), persistent=False)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(6 if self.ego_map_channels else 3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.ReLU())
        self.gru = nn.GRU(256, hidden_size, num_layers=1, batch_first=True)

    def preprocess_vector(self, vector):
        result = super().preprocess_vector(vector)
        if self.ego_map_channels:
            result = result.clone()
            # World XY is redundant with observed ROI/boundaries; UAV may cross
            # islands, while the numerical sea mask already represents target land.
            result[..., :2] = 0
            result[..., 54:94] = 0
        return result

    def build_ego_channels(self, obs_sequence):
        """Observed aircraft position/heading channels on the existing ROI grid.

        The three public map channels remain unchanged. These internal channels
        derive exclusively from vector[94:98] and the current observed heading.
        Row coordinates increase northward, matching coverage_map's lower origin.
        Gaussian mode retains the original (location, location*cos, location*sin)
        encoding. Body-coordinate mode uses (cell body x/20000m, cell body y/20000m,
        location Gaussian); x points forward and y points left. World-coordinate
        mode uses (cell relative world x/20000m, cell relative world y/20000m,
        location Gaussian), independent of the aircraft's heading. Dense coordinates
        are never clipped, so aircraft outside the ROI remain spatially represented
        even when their Gaussian is numerically zero everywhere on the grid.
        """
        vector = obs_sequence["vector"]
        width = (vector[..., 95] - vector[..., 94]).clamp_min(1e-6)
        height = (vector[..., 97] - vector[..., 96]).clamp_min(1e-6)
        x = (-vector[..., 94] / width * 64).unsqueeze(-1).unsqueeze(-1)
        y = (-vector[..., 96] / height * 64).unsqueeze(-1).unsqueeze(-1)
        gaussian = torch.exp(-.5 * ((self._grid_x - x).square() + (self._grid_y - y).square())
                             / self.ego_map_sigma ** 2)
        heading = vector[..., 2:4]
        if self.ego_map_mode in {"body_coordinates", "world_coordinates"}:
            # obs-v1 ROI offsets use 20000m. Locate physical cell centers first,
            # then rotate their aircraft-relative displacement by raw heading.
            # Vector normalization/body preprocessing never enters this transform.
            scale_m = 20000.
            dx = (vector[..., 94, None, None] + self._grid_x / 64 * width[..., None, None]) * scale_m
            dy = (vector[..., 96, None, None] + self._grid_y / 64 * height[..., None, None]) * scale_m
            if self.ego_map_mode == "world_coordinates":
                return torch.stack((dx.expand_as(gaussian) / scale_m,
                                    dy.expand_as(gaussian) / scale_m, gaussian), dim=2)
            cosine, sine = heading[..., 0, None, None], heading[..., 1, None, None]
            body_x = (cosine * dx + sine * dy) / scale_m
            body_y = (-sine * dx + cosine * dy) / scale_m
            return torch.stack((body_x, body_y, gaussian), dim=2)
        return torch.stack((gaussian, gaussian * heading[..., 0, None, None],
                            gaussian * heading[..., 1, None, None]), dim=2)

    def forward(self, obs_sequence, hidden=None):
        vector_features = self._vector_features(obs_sequence)
        coverage_map = obs_sequence["coverage_map"]
        batch, time = vector_features.shape[:2]
        if coverage_map.shape != (batch, time, 3, 64, 64):
            raise ValueError("coverage_map must have shape (B,T,3,64,64)")
        if self.ego_map_channels:
            coverage_map = torch.cat((coverage_map, self.build_ego_channels(obs_sequence)), dim=2)
        image_features = self.map_encoder(coverage_map.reshape(batch * time, self.map_encoder[0].in_channels, 64, 64))
        encoded = torch.cat([vector_features, image_features.reshape(batch, time, 128)], dim=-1)
        return self._directions(encoded, obs_sequence["vector"], hidden)


def upgrade_coverage_checkpoint(source_path, target_path, *, ego_map_sigma=2., ego_map_mode="gaussian"):
    """Explicit 3→6 channel upgrade, preserving source and all unmasked weights.

    New convolution channels start at zero. Newly masked vector inputs have zero
    first-layer weights, mean zero and std one; unmasked normalization is retained.
    The selected encoding is explicit; existing six-channel weights cannot be
    reinterpreted as a different encoding. The original three-channel checkpoint
    is retained as a content-addressed snapshot, independent of production paths.
    Old loss metrics and optimizer state are discarded because this changes the
    function. Warm-start train_policy using the returned converted checkpoint.
    """
    source_path, target_path = Path(source_path).resolve(), Path(target_path).resolve()
    if source_path == target_path:
        raise ValueError("conversion target must be different from the source checkpoint")
    payload = source_path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    snapshot_path = target_path.parent / "source-checkpoints" / f"{source_hash}.pt"
    if target_path == snapshot_path:
        raise ValueError("conversion target must be different from the immutable source snapshot")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with snapshot_path.open("xb") as snapshot:
            snapshot.write(payload)
    except FileExistsError:
        pass
    snapshot_payload = snapshot_path.read_bytes()
    if hashlib.sha256(snapshot_payload).hexdigest() != source_hash:
        raise ValueError("source checkpoint snapshot has been modified")
    checkpoint = torch.load(io.BytesIO(snapshot_payload), map_location="cpu", weights_only=False)
    if (checkpoint.get("checkpoint_version") != "bc-v1" or checkpoint.get("observation_schema") != "obs-v1"
            or checkpoint.get("task") != "coverage"):
        raise ValueError("upgrade requires a coverage bc-v1 / obs-v1 checkpoint")
    config = dict(checkpoint["model_config"])
    config.setdefault("action_frame", "world")
    config.setdefault("feature_frame", "world")
    if config.get("ego_map_channels", False):
        raise ValueError("source checkpoint already has ego map channels")
    original = CoveragePolicy(**config)
    original.load_state_dict(checkpoint["model_state_dict"])
    config.update(ego_map_channels=True, ego_map_sigma=ego_map_sigma, ego_map_mode=ego_map_mode)
    upgraded = CoveragePolicy(**config)
    state = {key: value.clone() for key, value in original.state_dict().items()}
    convolution = torch.zeros_like(upgraded.map_encoder[0].weight)
    convolution[:, :3] = state["map_encoder.0.weight"]
    state["map_encoder.0.weight"] = convolution
    masked = [0, 1] + list(range(54, 94))
    state["vector_mean"][masked] = 0
    state["vector_std"][masked] = 1
    state["vector_encoder.0.weight"][:, masked] = 0
    upgraded.load_state_dict(state)
    coordinate_frame = "body" if ego_map_mode == "body_coordinates" else "world"
    channels = (["ego_gaussian", "ego_gaussian_cos_heading", "ego_gaussian_sin_heading"]
                if ego_map_mode == "gaussian" else [f"cell_{coordinate_frame}_x_over_20000m",
                                                   f"cell_{coordinate_frame}_y_over_20000m", "ego_gaussian"])
    conversion = dict(source_checkpoint=str(snapshot_path), requested_source_checkpoint=str(source_path),
                      source_checkpoint_sha256=source_hash, source_epoch=checkpoint.get("epoch"),
                      added_map_channels=channels, ego_map_mode=ego_map_mode,
                      new_channel_weights="zero", ego_map_sigma=float(ego_map_sigma),
                      masked_vector_fields=["absolute_world_xy", "island_vector_slots"],
                      masked_normalization="mean0_std1", unmasked_normalization="retained", optimizer_reset=True)
    normalization = dict(checkpoint.get("normalization", {}))
    normalization.update(vector_mean=state["vector_mean"].tolist(), vector_std=state["vector_std"].tolist(),
                         architecture_conversion=conversion, normalization_refitted=False)
    checkpoint.update(model_config=upgraded.model_config, model_state_dict=state,
                      normalization=normalization, optimizer_state_dict={}, epoch=0,
                      train_metrics=None, validation_metrics=None, architecture_conversion=conversion)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target_path)
    return target_path


def load_policy(path, device="cpu"):
    policy = NeuralPolicy.from_checkpoint(path, device=device)
    if policy.model.task != "coverage":
        raise ValueError("checkpoint does not contain a coverage policy")
    return policy
