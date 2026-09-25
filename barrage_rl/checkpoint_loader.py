"""Inference checkpoint loading without evaluation or training imports."""
from __future__ import annotations
from typing import Any
import torch
from .tracked_policy import ActionQueryPolicy, TrackedPolicyAgent, TrackedPolicySpec


def load_tracked_agent(
    checkpoint_path: str,
    device: torch.device,
    *,
    analytic_shield: bool = False,
    analytic_guard_horizon_seconds: float = 0.10,
    analytic_clearance_margin: float = 0.0,
    analytic_shield_gate: str = "always",
    analytic_min_tracked_count_ratio: float = 0.90,
    analytic_max_model_risk_increase: float = 0.005,
    analytic_max_selected_violation: float = 0.50,
    long_horizon_risk_weight: float = 0.0,
    teacher_cost_ranking_weight: float = 0.0,
    action_hysteresis_bonus: float = 0.0,
    experimental_controller: str | None = None,
    checkpoint_data: dict[str, Any] | None = None,
    materialize_state: bool = False,
) -> tuple[TrackedPolicyAgent, TrackedPolicySpec, dict[str, Any]]:
    checkpoint = checkpoint_data
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_controller = checkpoint.get("experimental_controller")
    if required_controller is not None and required_controller != experimental_controller:
        raise ValueError(
            "This checkpoint requires its experimental image controller; "
            "use tools/train_targeted_dagger.py --evaluate CHECKPOINT. "
            "The default deployment controller cannot evaluate it interchangeably."
        )
    spec = TrackedPolicySpec(**checkpoint["tracked_policy_spec"])
    model_hparams = dict(checkpoint.get("model_hparams", {}))
    if (
        int(checkpoint.get("model_version", 10)) <= 10
        and "geometry_statistics" not in model_hparams
    ):
        model_hparams["geometry_statistics"] = ("minimum",)
    if materialize_state:
        # Every persistent tensor is supplied by the checkpoint. Meta construction
        # avoids random initialization and immediately overwritten CPU storage.
        with torch.device('meta'):
            model = ActionQueryPolicy(spec, **model_hparams)
        expected_state = model.state_dict()
        state = {name: value.to(device=device, dtype=expected_state[name].dtype if name in expected_state else value.dtype)
                 for name, value in checkpoint['model'].items()}
        model.load_state_dict(state, assign=True)
    else:
        model = ActionQueryPolicy(spec, **model_hparams).to(device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
    agent_kwargs = {
        "analytic_shield": analytic_shield,
        "analytic_guard_horizon_seconds": analytic_guard_horizon_seconds,
        "analytic_clearance_margin": analytic_clearance_margin,
        "analytic_shield_gate": analytic_shield_gate,
        "analytic_min_tracked_count_ratio": analytic_min_tracked_count_ratio,
        "analytic_max_model_risk_increase": analytic_max_model_risk_increase,
        "analytic_max_selected_violation": analytic_max_selected_violation,
        "long_horizon_risk_weight": long_horizon_risk_weight,
        "teacher_cost_ranking_weight": teacher_cost_ranking_weight,
        "action_hysteresis_bonus": action_hysteresis_bonus,
    }
    if "distilled_student" in checkpoint:
        from .distilled_student import DistilledStudentNetwork, DistilledStudentSpec, UnifiedDistilledAgent
        schema = int(checkpoint.get("distilled_student_schema_version", -1))
        if schema not in DistilledStudentNetwork.compatible_schema_versions:
            raise ValueError(
                f"unsupported distilled student schema: {schema}"
            )
        distilled_spec = DistilledStudentSpec.from_checkpoint(
            checkpoint["distilled_student_spec"]
        )
        distilled = DistilledStudentNetwork(distilled_spec).to(device)
        distilled.load_compatible_state_dict(checkpoint["distilled_student"])
        distilled.eval()
        agent = UnifiedDistilledAgent(
            model,
            distilled,
            device,
            **agent_kwargs,
        )
    else:
        agent = TrackedPolicyAgent(model, device, **agent_kwargs)
    return agent, spec, checkpoint


def checkpoint_action_delay_steps(checkpoint: dict[str, object]) -> int:
    """Return the control timing that the checkpoint was trained to use.

    Current checkpoints record their timing in the saved training config.
    Missing metadata retains immediate, same-boundary action application.
    """
    config = dict(checkpoint.get("config", {}))
    for key in (
        "evaluation_causal_action_delay_steps",
        "collection_causal_action_delay_steps",
    ):
        if key in config:
            delay = int(config[key])
            if delay not in (0, 1):
                raise ValueError(f"checkpoint {key} must be zero or one")
            return delay
    return 0


