from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from barrage_rl.distilled_student import (
    DistilledStudentNetwork,
    DistilledStudentSpec,
    UnifiedDistilledAgent,
)
from barrage_rl.evaluate_tracked_policy import load_tracked_agent
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec


class DistilledStudentTests(unittest.TestCase):
    def test_reused_replay_cannot_relabel_old_bullet_distribution(self) -> None:
        from tools.train_distilled_student import DistillationConfig, run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "run_manifest.json"
            manifest.write_text(json.dumps({"task": {
                "bullet_count": 250, "targeted_bullet_probability": 0.10,
            }}), encoding="utf-8")
            original = manifest.read_bytes()
            output = root / "new_training"
            with self.assertRaisesRegex(ValueError, "collected with 300 bullets"):
                run(backbone_path=root / "unused.pt", output=output,
                    config=DistillationConfig(), reuse_replay=root / "replay")
            self.assertFalse(output.exists())
            self.assertEqual(manifest.read_bytes(), original)

    def test_forward_shapes_and_recurrent_state(self) -> None:
        model = DistilledStudentNetwork(DistilledStudentSpec(
            model_width=32, hidden_width=16, history_decisions=8,
        ))
        prediction = model(
            torch.randn(4, 32),
            torch.randn(4, 9, 32),
            torch.randn(4, 9, 6),
        )
        self.assertEqual(prediction.policy_correction.shape, (4, 9))
        self.assertEqual(prediction.hidden.shape, (4, 16))

    def test_agent_keeps_episode_state_isolated_and_resettable(self) -> None:
        policy_spec = TrackedPolicySpec(
            max_objects=6, tracker_capacity=6, expected_bullet_count=5,
        )
        backbone = ActionQueryPolicy(
            policy_spec, width=32, attention_layers=1, attention_heads=4,
        )
        student = DistilledStudentNetwork(DistilledStudentSpec(
            model_width=32, hidden_width=16, history_decisions=8,
        ))
        agent = UnifiedDistilledAgent(
            backbone, student, torch.device("cpu"),
            use_safety_filter=True, safety_threshold=0.5,
        )
        objects = np.zeros((2, 6, 16), np.float32)
        masks = np.zeros((2, 6), np.bool_)
        globals_ = np.zeros((2, 16), np.float32)
        globals_[:, 8:10] = 1.0
        actions = agent.act_features(
            objects, masks, globals_,
            episode_indices=np.asarray([10, 20]),
            decision_indices=np.asarray([0, 0]),
        )
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(set(agent._distilled_hidden), {10, 20})
        agent.reset_state(np.asarray([10]))
        self.assertEqual(set(agent._distilled_hidden), {20})
        agent.reset_state()
        self.assertFalse(agent._distilled_hidden)

    def test_legacy_checkpoint_loads_only_active_student_weights(self) -> None:
        policy_spec = TrackedPolicySpec(
            max_objects=6, tracker_capacity=6, expected_bullet_count=5,
        )
        backbone = ActionQueryPolicy(
            policy_spec, width=32, attention_layers=1, attention_heads=4,
        )
        student_spec = DistilledStudentSpec(
            model_width=32, hidden_width=16, history_decisions=8,
        )
        student = DistilledStudentNetwork(student_spec)
        legacy_spec = student_spec.to_dict() | {
            "fallback_threshold": 0.5,
            "fallback_decisions": 3,
        }
        legacy_weights = dict(student.state_dict())
        legacy_weights["clearance_head.weight"] = torch.zeros(9, 16)
        checkpoint = {
            "tracked_policy_spec": policy_spec.__dict__,
            "model_hparams": {
                "width": 32, "attention_layers": 1, "attention_heads": 4,
            },
            "model_version": backbone.model_version,
            "model": backbone.state_dict(),
            "use_safety_filter": True,
            "safety_threshold": 0.5,
            "distilled_student_schema_version": 1,
            "distilled_student_spec": legacy_spec,
            "distilled_student": legacy_weights,
            "distilled_planner_fallback": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            torch.save(checkpoint, path)
            agent, loaded_spec, _ = load_tracked_agent(
                str(path), torch.device("cpu")
            )
        self.assertIsInstance(agent, UnifiedDistilledAgent)
        self.assertEqual(loaded_spec.max_objects, 6)


if __name__ == "__main__":
    unittest.main()
