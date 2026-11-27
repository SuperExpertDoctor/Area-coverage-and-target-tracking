"""Independent local-field CNN/physical MLP/GRU policies for field-v2.

The public field and goal use world axes. An explicit body-frame model rotates
them internally using only the observed physical heading.
Previous teacher commands are masked by default: heading, bank, public goals,
field geometry and recurrent memory provide the causal control inputs.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .policy_tracking import NeuralPolicy, _DirectionalRecurrentPolicy, normalize_direction


class FieldPolicy(_DirectionalRecurrentPolicy):
    """vector[111/134], potential[4,33,33], context[16] → unit XY and pan."""

    action_dim = 3
    observation_schema = "obs-field-v2"
    action_schema = "action-field-v2"

    def __init__(self, task, hidden_size=128, vector_dim=None, mask_previous_command=True,
                 field_frame="world", pan_residual=False, pan_reference="actual",
                 direction_reference="heading", reference_orbit_radius=1700., reference_radial_gain=2.5):
        if task not in {"coverage", "tracking"}:
            raise ValueError("task must be coverage or tracking")
        expected = 111 if task == "coverage" else 134
        if vector_dim is not None and vector_dim != expected:
            raise ValueError("vector_dim does not match the field task schema")
        self.task = task
        if field_frame not in {"body", "world"}:
            raise ValueError("field_frame must be body or world")
        self.field_frame = field_frame
        self.pan_residual = bool(pan_residual)
        if pan_reference not in {"actual", "goal"}:
            raise ValueError("pan_reference must be actual or goal")
        self.pan_reference = pan_reference
        if direction_reference not in {"heading", "goal_orbit"}:
            raise ValueError("direction_reference must be heading or goal_orbit")
        for name, value in [("reference_orbit_radius", reference_orbit_radius),
                            ("reference_radial_gain", reference_radial_gain)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.direction_reference = direction_reference
        self.reference_orbit_radius = float(reference_orbit_radius)
        self.reference_radial_gain = float(reference_radial_gain)
        super().__init__(expected, hidden_size, True, field_frame, field_frame)
        self.mask_previous_command = bool(mask_previous_command)
        self.model_config = dict(potential_field=True, hidden_size=self.hidden_size,
                                 vector_dim=self.vector_dim,
                                 mask_previous_command=self.mask_previous_command,
                                 field_frame=field_frame, pan_residual=self.pan_residual,
                                 pan_reference=pan_reference, direction_reference=direction_reference,
                                 reference_orbit_radius=self.reference_orbit_radius,
                                 reference_radial_gain=self.reference_radial_gain)
        self.field_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.ReLU())
        self.context_encoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(),
                                             nn.Linear(64, 64), nn.ReLU())
        self.gru = nn.GRU(320, hidden_size, num_layers=1, batch_first=True)
        self.pan_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1))

    def preprocess_vector(self, vector):
        result = super().preprocess_vector(vector)
        if self.field_frame == "body":
            result = result.clone()
            result[..., :2] = 0
            result[..., 2] = 1
            result[..., 3] = 0
        if self.mask_previous_command:
            result = result.clone()
            result[..., 8:10] = 0
        return result

    def preprocess_context(self, obs):
        context = obs["field_context"]
        if self.field_frame == "world":
            return context
        heading = normalize_direction(obs["vector"][..., 2:4], obs["vector"][..., 2:4])
        result = context.clone()
        goal = context[..., 7:9]
        result[..., 7] = heading[..., 0] * goal[..., 0] + heading[..., 1] * goal[..., 1]
        result[..., 8] = -heading[..., 1] * goal[..., 0] + heading[..., 0] * goal[..., 1]
        return result

    def preprocess_field(self, obs):
        field = obs["potential"]
        if self.field_frame == "world":
            return field
        batch, time = field.shape[:2]
        heading = normalize_direction(obs["vector"][..., 2:4], obs["vector"][..., 2:4]).reshape(-1, 2)
        affine = field.new_zeros(batch * time, 2, 3)
        affine[:, 0, 0] = heading[:, 0]
        affine[:, 0, 1] = -heading[:, 1]
        affine[:, 1, 0] = heading[:, 1]
        affine[:, 1, 1] = heading[:, 0]
        image = field.reshape(batch * time, 4, 33, 33)
        # The public raster has inclusive centers at +/-R; align_corners=True
        # preserves its physical scale. Tensor rows and columns map to Y/X.
        grid = F.affine_grid(affine, image.shape, align_corners=True)
        return F.grid_sample(image, grid, align_corners=True).reshape_as(field)

    def _world_reference(self, obs, heading):
        if self.direction_reference == "heading":
            return heading
        # Public-goal actuator coordinates remove the need to infer atan2 and
        # orbit stability from sparse demonstrations. Field-sensitive decisions
        # remain learned residuals; this reference has no collision planner.
        vector = obs["vector"]
        context = obs["field_context"]
        if self.task == "coverage":
            speed = vector[..., 98:99] * (220 / 3.6)
            velocity = vector[..., 4:6] * speed
            delay = vector[..., 101:102] + vector[..., 103:104] + .5 * vector[..., 105:106]
            relative = -context[..., 7:9] * 3000 + velocity * delay
            radius = self.reference_orbit_radius
        else:
            speed = vector[..., 106:107] * (220 / 3.6)
            velocity = vector[..., 4:6] * speed
            target_velocity = vector[..., 97:99] * (220 / 3.6)
            delay = vector[..., 109:110] + vector[..., 111:112] + .5 * vector[..., 122:123]
            relative = -context[..., 7:9] * 3000 + (velocity - target_velocity) * delay
            radius = (vector[..., 131:132] * 3000).clamp_min(1.)
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1.)
        radial = relative / distance
        tangent = torch.stack((radial[..., 1], -radial[..., 0]), dim=-1)
        correction = self.reference_radial_gain * (distance - radius) / radius
        reference = normalize_direction(tangent - correction * radial, heading)
        if self.task == "tracking":
            # Relative-orbit coordinates compensate the observed moving target
            # while preserving the fixed aircraft speed before normalization.
            projection = (target_velocity * reference).sum(-1, keepdim=True)
            target_speed2 = target_velocity.square().sum(-1, keepdim=True)
            relative_speed = -projection + (projection.square() + speed.square() - target_speed2).clamp_min(0.).sqrt()
            reference = normalize_direction(target_velocity + relative_speed * reference, heading)
        return torch.where(context[..., 10:11] > .5, reference, heading)

    def forward(self, obs_sequence, hidden=None):
        physical = self._vector_features(obs_sequence)
        batch, time = physical.shape[:2]
        potential = obs_sequence["potential"]
        context = obs_sequence["field_context"]
        if potential.shape != (batch, time, 4, 33, 33):
            raise ValueError("potential must have shape (B,T,4,33,33)")
        if context.shape != (batch, time, 16):
            raise ValueError("field_context must have shape (B,T,16)")
        potential = self.preprocess_field(obs_sequence)
        context = self.preprocess_context(obs_sequence)
        spatial = self.field_encoder(potential.reshape(batch * time, 4, 33, 33))
        encoded = torch.cat((physical, spatial.reshape(batch, time, 128),
                             self.context_encoder(context)), dim=-1)
        recurrent, new_hidden = self.gru(encoded, hidden)
        heading = normalize_direction(obs_sequence["vector"][..., 2:4],
                                      obs_sequence["vector"][..., 2:4])
        baseline = self._world_reference(obs_sequence, heading)
        if self.field_frame == "body":
            if self.direction_reference == "heading":
                baseline = torch.zeros_like(heading)
                baseline[..., 0] = 1
            else:
                baseline = torch.stack((heading[..., 0] * baseline[..., 0] + heading[..., 1] * baseline[..., 1],
                                        -heading[..., 1] * baseline[..., 0] + heading[..., 0] * baseline[..., 1]), -1)
        raw_xy = torch.tanh(self.action_head(recurrent) + torch.atanh(.5 * baseline))
        xy = normalize_direction(raw_xy, baseline)
        if self.field_frame == "body":
            xy = torch.stack((heading[..., 0] * xy[..., 0] - heading[..., 1] * xy[..., 1],
                              heading[..., 1] * xy[..., 0] + heading[..., 0] * xy[..., 1]), -1)
        xy = normalize_direction(xy, heading).clamp(-1, 1)
        pan_logits = self.pan_head(recurrent)
        if self.pan_residual:
            raw_context = obs_sequence["field_context"]
            if self.pan_reference == "goal":
                # This is an observation-derived actuator coordinate, analogous
                # to the heading reference for XY. The learned residual remains
                # free to command either direction; no expert is invoked.
                goal = raw_context[..., 7:9]
                angle = (torch.atan2(goal[..., 1], goal[..., 0]) -
                         torch.atan2(heading[..., 1], heading[..., 0]) -
                         raw_context[..., 4] * torch.pi)
                reference = torch.atan2(torch.sin(angle), torch.cos(angle))[..., None]
                reference = reference / (raw_context[..., 5:6] * torch.pi).clamp_min(1e-6)
            else:
                reference = raw_context[..., 3:4] / raw_context[..., 5:6].clamp_min(1e-6)
            pan_logits = pan_logits + torch.atanh(reference.clamp(-.99, .99))
        pan = torch.tanh(pan_logits)
        return torch.cat((xy, pan), dim=-1), new_hidden


class FieldNeuralPolicy(NeuralPolicy):
    """Field-only inference; no expert fallback or hidden environmental inputs."""

    def __init__(self, model, device="cpu"):
        if not isinstance(model, FieldPolicy):
            raise ValueError("FieldNeuralPolicy requires a field-v2 model; protocol mismatch")
        super().__init__(model, device=device)

    @torch.inference_mode()
    def act(self, obs):
        expected = {"vector": (self.model.vector_dim,), "potential": (4, 33, 33),
                    "field_context": (16,)}
        tensors = {}
        for key, shape in expected.items():
            if key not in obs:
                raise ValueError(f"field observation requires {key}; obs-field-v2 protocol required")
            array = np.asarray(obs[key], dtype=np.float32)
            if array.shape != shape:
                raise ValueError(f"{key} must have shape {shape}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError("observations must be finite")
            tensors[key] = torch.as_tensor(array, device=self.device)[None, None]
        actions, self.hidden = self.model(tensors, self.hidden)
        return actions[0, 0].cpu().numpy().copy()
