"""Causal tracking policy and the common stateful inference adapter.

The default body-frame action head predicts a local forward/left direction and
rotates it to world XY using the observed heading. With the heading residual it
adds [atanh(0.5), 0] to learned logits before tanh and normalization, initializing
heading hold while retaining every direction. action_frame='world' retains the
earlier world-coordinate head for checkpoints and controlled comparisons.
Set ``heading_residual=False``, ``action_frame='world'`` and
``feature_frame='world'`` for the literal direction head in design section 9.
Each model owns all of its own weights and recurrent state.
"""

from pathlib import Path
import hashlib
import io

import numpy as np
import torch
from torch import nn


def normalize_direction(direction, heading, epsilon=1e-6):
    """Apply the same unit action / heading fallback during training and inference."""
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    east = torch.zeros_like(heading)
    east[..., 0] = 1
    fallback = torch.where(heading_norm >= epsilon, heading / heading_norm.clamp_min(epsilon), east)
    return torch.where(norm >= epsilon, direction / norm.clamp_min(epsilon), fallback)


class _DirectionalRecurrentPolicy(nn.Module):
    """Shared implementation, instantiated independently by the two task classes."""

    def __init__(self, vector_dim, hidden_size, heading_residual, action_frame, feature_frame):
        super().__init__()
        if vector_dim < 4 or hidden_size < 1:
            raise ValueError("vector_dim must be >= 4 and hidden_size positive")
        self.vector_dim = int(vector_dim)
        self.hidden_size = int(hidden_size)
        self.heading_residual = bool(heading_residual)
        if action_frame not in {"body", "world"}:
            raise ValueError("action_frame must be body or world")
        self.action_frame = action_frame
        if feature_frame not in {"body", "world"}:
            raise ValueError("feature_frame must be body or world")
        self.feature_frame = feature_frame
        self.model_config = dict(vector_dim=self.vector_dim, hidden_size=self.hidden_size,
                                 heading_residual=self.heading_residual, action_frame=self.action_frame,
                                 feature_frame=self.feature_frame)
        self.register_buffer("vector_mean", torch.zeros(vector_dim))
        self.register_buffer("vector_std", torch.ones(vector_dim))
        self.vector_encoder = nn.Sequential(nn.Linear(vector_dim, 128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 2))
        # A small directional correction avoids near-zero random unit vectors at initialization.
        nn.init.normal_(self.action_head[-1].weight, std=0.001)
        nn.init.zeros_(self.action_head[-1].bias)
        if not self.heading_residual:
            with torch.no_grad():
                self.action_head[-1].bias[0] = .5

    def set_vector_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.vector_mean.dtype, device=self.vector_mean.device)
        std = torch.as_tensor(std, dtype=self.vector_std.dtype, device=self.vector_std.device)
        if mean.shape != self.vector_mean.shape or std.shape != self.vector_std.shape:
            raise ValueError("normalization dimensions do not match the observation vector")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("normalization must contain finite mean and positive finite std")
        self.vector_mean.copy_(mean)
        self.vector_std.copy_(std)

    def preprocess_vector(self, vector):
        """Express relative vector pairs in forward/left coordinates, without new inputs.

        Boundary distances and scalar physical fields remain available. Tracking
        masks absolute world XY and canonicalizes heading to [1,0]; coverage
        retains world heading because its numerical map uses fixed world axes.
        This is an inductive bias, not full rotation invariance of every feature.
        """
        if self.feature_frame == "world":
            return vector
        heading = normalize_direction(vector[..., 2:4], vector[..., 2:4])
        result = vector.clone()
        starts = [4, 8] + [base + 5 * slot for base in [14, 54] for slot in range(8)]
        if self.task == "tracking":
            starts += [94, 97, 99]
            result[..., :2] = 0
            result[..., 2] = 1
            result[..., 3] = 0
        for start in starts:
            pair = vector[..., start:start + 2]
            result[..., start] = heading[..., 0] * pair[..., 0] + heading[..., 1] * pair[..., 1]
            result[..., start + 1] = -heading[..., 1] * pair[..., 0] + heading[..., 0] * pair[..., 1]
        return result

    def _vector_features(self, obs):
        vector = obs["vector"]
        if vector.ndim != 3 or vector.shape[-1] != self.vector_dim:
            raise ValueError(f"vector must have shape (B,T,{self.vector_dim})")
        vector = self.preprocess_vector(vector)
        return self.vector_encoder((vector - self.vector_mean) / self.vector_std)

    def _directions(self, encoded, vector, hidden):
        recurrent, new_hidden = self.gru(encoded, hidden)
        logits = self.action_head(recurrent)
        heading = normalize_direction(vector[..., 2:4], vector[..., 2:4])
        baseline = torch.zeros_like(heading)
        baseline[..., 0] = 1
        if self.action_frame == "world":
            baseline = heading
        if self.heading_residual:
            # Adding a bounded correction AFTER tanh would forbid a reversal
            # from cardinal headings. Put the heading baseline BEFORE tanh so
            # every direction remains attainable and zero logits hold heading.
            logits = logits + torch.atanh(.5 * baseline.clamp(-1, 1))
        raw = torch.tanh(logits)
        direction = normalize_direction(raw, baseline)
        if self.action_frame == "body":
            direction = torch.stack((heading[..., 0] * direction[..., 0] - heading[..., 1] * direction[..., 1],
                                     heading[..., 1] * direction[..., 0] + heading[..., 0] * direction[..., 1]), dim=-1)
        # A float32 rotation can yield 1.000000119 on a cardinal component.
        # Renormalize and enforce the exact public Box in both train and deploy.
        return normalize_direction(direction, heading).clamp(-1, 1), new_hidden


class TrackingPolicy(_DirectionalRecurrentPolicy):
    """134-vector → MLP(128,128) → causal GRU → deterministic XY direction."""

    task = "tracking"

    def __init__(self, vector_dim=134, hidden_size=128, heading_residual=True,
                 action_frame="body", feature_frame="body"):
        super().__init__(vector_dim, hidden_size, heading_residual, action_frame, feature_frame)
        self.gru = nn.GRU(128, hidden_size, num_layers=1, batch_first=True)

    def forward(self, obs_sequence, hidden=None):
        return self._directions(self._vector_features(obs_sequence), obs_sequence["vector"], hidden)


class NeuralPolicy:
    """Stateful ``reset()/act(obs)`` adapter; call reset after every environment reset."""

    def __init__(self, model, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.hidden = None

    def reset(self):
        self.hidden = None

    @torch.inference_mode()
    def act(self, obs):
        if "potential" in obs or "field_context" in obs:
            raise ValueError("obs-field-v2 observation requires a field-v2 checkpoint; protocol mismatch")
        if "vector" not in obs:
            raise ValueError("observation requires vector")
        expected = {"vector": (self.model.vector_dim,)}
        if self.model.task == "coverage":
            expected["coverage_map"] = (3, 64, 64)
        tensors = {}
        for key, shape in expected.items():
            if key not in obs:
                raise ValueError(f"observation requires {key}")
            array = np.asarray(obs[key], dtype=np.float32)
            if array.shape != shape:
                raise ValueError(f"{key} must have shape {shape}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError("observations must be finite")
            tensors[key] = torch.as_tensor(array, device=self.device).unsqueeze(0).unsqueeze(0)
        actions, self.hidden = self.model(tensors, self.hidden)
        return actions[0, 0].cpu().numpy().copy()

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a versioned checkpoint, including train-fitted input normalization."""
        from .train import create_policy
        source = Path(path).resolve()
        payload = source.read_bytes()
        checkpoint = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
        field_mode = checkpoint.get("checkpoint_version") == "bc-field-v2"
        if checkpoint.get("checkpoint_version") not in {"bc-v1", "bc-field-v2"}:
            raise ValueError("unsupported behavior-cloning checkpoint version")
        expected_schema = "obs-field-v2" if field_mode else "obs-v1"
        if checkpoint.get("observation_schema") != expected_schema:
            raise ValueError("checkpoint observation schema is incompatible")
        if field_mode and checkpoint.get("action_schema") != "action-field-v2":
            raise ValueError("checkpoint action schema is incompatible")
        model_config = dict(checkpoint["model_config"])
        if bool(model_config.get("potential_field", False)) != field_mode:
            raise ValueError("checkpoint model config and schema protocol disagree")
        if not field_mode:
            model_config.setdefault("action_frame", "world")  # pre-body-head checkpoints
            model_config.setdefault("feature_frame", "world")  # pre-body-feature checkpoints
        model = create_policy(checkpoint["task"], model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        if field_mode:
            from .policy_field import FieldNeuralPolicy
            policy = FieldNeuralPolicy(model, device=device)
        else:
            policy = cls(model, device=device)
        policy.checkpoint_metadata = {'path':str(source), 'sha256':hashlib.sha256(payload).hexdigest(),
                                      'epoch':checkpoint.get('epoch'), 'model_config':model.model_config}
        return policy


def load_policy(path, device="cpu"):
    return NeuralPolicy.from_checkpoint(path, device=device)
