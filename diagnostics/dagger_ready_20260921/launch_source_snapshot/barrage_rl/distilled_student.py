"""History-aware residual student for the tracked image policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .tracked_policy import ActionQueryPolicy, TrackedPolicyAgent


@dataclass(frozen=True)
class DistilledStudentSpec:
    model_width: int = 192
    hidden_width: int = 128
    action_features: int = 6
    history_decisions: int = 60

    def __post_init__(self) -> None:
        if min(
            self.model_width,
            self.hidden_width,
            self.action_features,
            self.history_decisions,
        ) <= 0:
            raise ValueError("distilled student dimensions must be positive")

    @classmethod
    def from_checkpoint(cls, values: Mapping[str, object]) -> "DistilledStudentSpec":
        """Read active fields while ignoring retired checkpoint metadata."""

        names = {item.name for item in fields(cls)}
        return cls(**{name: values[name] for name in names if name in values})

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class DistilledStudentPrediction:
    policy_correction: torch.Tensor
    hidden: torch.Tensor


def causal_frame_latent(
    object_tokens: torch.Tensor,
    object_mask: torch.Tensor,
    action_queries: torch.Tensor,
) -> torch.Tensor:
    """Compress one image-derived decision for causal history storage."""

    weight = object_mask.to(object_tokens.dtype).unsqueeze(-1)
    pooled_objects = (object_tokens * weight).sum(dim=1) / weight.sum(
        dim=1
    ).clamp_min(1.0)
    return 0.5 * (pooled_objects + action_queries.mean(dim=1))


def _linear(in_features: int, out_features: int, gain: float = 1.0) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class DistilledStudentNetwork(nn.Module):
    """Recurrent residual policy trained on image-derived features."""

    schema_version = 2
    compatible_schema_versions = (1, 2)

    def __init__(self, spec: DistilledStudentSpec = DistilledStudentSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.history_projection = nn.Sequential(
            _linear(spec.model_width, spec.hidden_width, 2**0.5),
            nn.GELU(),
            nn.LayerNorm(spec.hidden_width),
        )
        self.history = nn.GRUCell(spec.hidden_width, spec.hidden_width)
        self.action_feature_encoder = nn.Sequential(
            _linear(spec.action_features, spec.hidden_width, 2**0.5),
            nn.GELU(),
            nn.LayerNorm(spec.hidden_width),
        )
        self.action_trunk = nn.Sequential(
            _linear(
                spec.model_width + 2 * spec.hidden_width,
                spec.hidden_width,
                2**0.5,
            ),
            nn.GELU(),
            nn.LayerNorm(spec.hidden_width),
        )
        self.policy_head = _linear(spec.hidden_width, 1, 0.01)

    def load_compatible_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        """Load active weights from current or legacy student checkpoints."""

        active = self.state_dict()
        filtered = {
            name: value
            for name, value in state.items()
            if name in active and active[name].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if unexpected:
            raise ValueError(f"unexpected active student weights: {unexpected}")
        if missing:
            raise ValueError(f"missing active student weights: {missing}")

    def initial_hidden(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.spec.hidden_width, device=device, dtype=dtype
        )

    @staticmethod
    def action_features(
        base_policy: torch.Tensor,
        teacher_cost: torch.Tensor,
        collision_logits: torch.Tensor,
    ) -> torch.Tensor:
        collision = torch.sigmoid(collision_logits)
        return torch.stack(
            (
                base_policy,
                teacher_cost,
                collision[:, 0],
                collision[:, 1],
                collision[:, 2],
                collision[:, 3],
            ),
            dim=-1,
        )

    def forward(
        self,
        frame_latent: torch.Tensor,
        action_queries: torch.Tensor,
        action_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> DistilledStudentPrediction:
        batch, actions, width = action_queries.shape
        if width != self.spec.model_width:
            raise ValueError("action-query width differs from distilled student spec")
        if tuple(action_features.shape) != (
            batch,
            actions,
            self.spec.action_features,
        ):
            raise ValueError("distilled action features have the wrong shape")
        if hidden is None:
            hidden = self.initial_hidden(
                batch, device=frame_latent.device, dtype=frame_latent.dtype
            )
        projected = self.history_projection(frame_latent)
        next_hidden = self.history(projected, hidden)
        repeated = next_hidden[:, None, :].expand(-1, actions, -1)
        encoded_features = self.action_feature_encoder(action_features)
        values = self.action_trunk(
            torch.cat((action_queries, encoded_features, repeated), dim=-1)
        )
        return DistilledStudentPrediction(
            policy_correction=self.policy_head(values).squeeze(-1),
            hidden=next_hidden,
        )

    def forward_from_backbone(
        self,
        object_tokens: torch.Tensor,
        object_mask: torch.Tensor,
        action_queries: torch.Tensor,
        base_policy: torch.Tensor,
        teacher_cost: torch.Tensor,
        collision_logits: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> DistilledStudentPrediction:
        frame = causal_frame_latent(object_tokens, object_mask, action_queries)
        features = self.action_features(base_policy, teacher_cost, collision_logits)
        return self(frame, action_queries, features, hidden)


class UnifiedDistilledAgent(TrackedPolicyAgent):
    """One recurrent learned policy with unified action arbitration."""

    def __init__(
        self,
        model: ActionQueryPolicy,
        distilled: DistilledStudentNetwork,
        device: torch.device,
        **kwargs: object,
    ) -> None:
        configured = dict(kwargs)
        configured["analytic_shield"] = False
        super().__init__(model, device, **configured)
        self.distilled = distilled
        self.distilled.eval()
        self._distilled_hidden: dict[int, torch.Tensor] = {}
        self._distilled_last_decision: dict[int, int] = {}
        self.student_decision_count = 0

    @property
    def action_selector_mode(self) -> str:
        return "unified_distilled_student"

    @property
    def safety_filter_mode(self) -> str:
        return "distilled_student_learned_collision"

    @staticmethod
    def _state_keys(
        batch_size: int,
        episode_indices: np.ndarray | None,
    ) -> tuple[int, ...]:
        if episode_indices is None:
            return tuple(range(batch_size))
        values = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        if len(values) != batch_size:
            raise ValueError("episode_indices must match the policy batch")
        if len(np.unique(values)) != len(values):
            raise ValueError("episode_indices must be unique within a batch")
        return tuple(int(value) for value in values)

    def reset_state(self, episode_indices: np.ndarray | None = None) -> None:
        if episode_indices is None:
            self._distilled_hidden.clear()
            self._distilled_last_decision.clear()
            return
        for raw in np.asarray(episode_indices, dtype=np.int64).reshape(-1):
            key = int(raw)
            self._distilled_hidden.pop(key, None)
            self._distilled_last_decision.pop(key, None)

    def _hidden_for(
        self,
        keys: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        decision_indices: np.ndarray | None,
    ) -> torch.Tensor:
        decisions = (
            None
            if decision_indices is None
            else np.asarray(decision_indices, dtype=np.int64).reshape(-1)
        )
        rows = []
        for index, key in enumerate(keys):
            if decisions is not None:
                decision = int(decisions[index])
                previous = self._distilled_last_decision.get(key)
                if previous is not None and decision <= previous:
                    self._distilled_hidden.pop(key, None)
                self._distilled_last_decision[key] = decision
            hidden = self._distilled_hidden.get(key)
            if hidden is None:
                hidden = torch.zeros(
                    self.distilled.spec.hidden_width,
                    device=device,
                    dtype=dtype,
                )
            rows.append(hidden)
        return torch.stack(rows)

    def _store_hidden(
        self, keys: Sequence[int], hidden: torch.Tensor
    ) -> None:
        for key, value in zip(keys, hidden.detach()):
            self._distilled_hidden[int(key)] = value

    @torch.inference_mode()
    def _act(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        *,
        episode_indices: np.ndarray | None,
        decision_indices: np.ndarray | None,
        include_diagnostics: bool = True,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        object_tensor = torch.as_tensor(
            objects, device=self.device, dtype=torch.float32
        )
        mask_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
        global_tensor = torch.as_tensor(
            globals_, device=self.device, dtype=torch.float32
        )
        policy, teacher_cost, collision_logits, geometry, latents = (
            self.model.forward_with_arbiter_latents(
                object_tensor,
                mask_tensor,
                global_tensor,
            )
        )
        keys = self._state_keys(len(objects), episode_indices)
        hidden = self._hidden_for(
            keys,
            device=object_tensor.device,
            dtype=object_tensor.dtype,
            decision_indices=decision_indices,
        )
        prediction = self.distilled.forward_from_backbone(
            latents.object_tokens,
            mask_tensor,
            latents.action_queries,
            policy,
            teacher_cost,
            collision_logits,
            hidden,
        )
        self._store_hidden(keys, prediction.hidden)

        policy = policy + prediction.policy_correction
        selection = self._action_selector.select(
            policy,
            teacher_cost,
            collision_logits,
            mask_tensor,
            global_tensor,
            geometry,
            deterministic=True,
        )
        actions = selection.actions
        raw_actions = selection.raw_actions
        student_actions = selection.learned_actions
        self.student_decision_count += len(objects)
        output = self._export_selection(selection)

        if not include_diagnostics:
            return output, {}

        row = torch.arange(len(objects), device=self.device)
        collision = torch.sigmoid(collision_logits)
        maximum_risk = collision.amax(dim=1)
        diagnostics = {
            "raw_policy_actions": raw_actions.cpu().numpy(),
            "learned_filter_actions": student_actions.cpu().numpy(),
            "raw_immediate_risk": collision[:, 0][row, raw_actions].cpu().numpy(),
            "selected_immediate_risk": collision[:, 0][row, actions].cpu().numpy(),
            "action_was_filtered": (actions != raw_actions).cpu().numpy(),
            "all_actions_unsafe": selection.all_unsafe.cpu().numpy(),
            "learned_filter_max_risk": maximum_risk[
                row, student_actions
            ].cpu().numpy(),
            "selected_max_risk": maximum_risk[row, actions].cpu().numpy(),
            "learned_filter_teacher_cost": teacher_cost[
                row, student_actions
            ].cpu().numpy(),
            "selected_teacher_cost": teacher_cost[row, actions].cpu().numpy(),
        }
        return output, diagnostics

    @torch.inference_mode()
    def act_features(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        deterministic: bool = True,
        episode_indices: np.ndarray | None = None,
        decision_indices: np.ndarray | None = None,
        **_: object,
    ) -> np.ndarray:
        del deterministic
        return self._act(
            objects,
            masks,
            globals_,
            episode_indices=episode_indices,
            decision_indices=decision_indices,
            include_diagnostics=False,
        )[0]

    @torch.inference_mode()
    def act_features_with_diagnostics(
        self,
        objects: np.ndarray,
        masks: np.ndarray,
        globals_: np.ndarray,
        episode_indices: np.ndarray | None = None,
        decision_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self._act(
            objects,
            masks,
            globals_,
            episode_indices=episode_indices,
            decision_indices=decision_indices,
        )
