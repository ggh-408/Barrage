"""Migration and training contracts for the two-head deployment policy."""
import unittest
import torch
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from barrage_rl.train_tracked_policy import TrackedDAggerConfig, _loss


class RiskRemovalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self):
        return ActionQueryPolicy(TrackedPolicySpec(max_objects=4), width=16,
                                 attention_layers=1, attention_heads=2)

    def test_legacy_migration_only_discards_retired_parameters(self):
        model = self.model()
        state = dict(model.state_dict())
        state["collision_head.weight"] = torch.randn(4, 16)
        state["collision_head.bias"] = torch.randn(4)
        restored = self.model()
        restored.load_state_dict(state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict({**state, "unrecognized.weight": torch.zeros(1)})
        del state["policy_head.weight"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(state)

    def test_retired_collision_targets_cannot_change_loss_or_gradients(self):
        model = self.model()
        objects = torch.rand(2, 4, 16)
        masks = torch.ones(2, 4, dtype=torch.bool)
        globals_ = torch.rand(2, 16)
        actions, regrets = torch.tensor([0, 1]), torch.rand(2, 9)
        outcomes = []
        for labels in (torch.zeros(2, 4, 9), torch.ones(2, 4, 9)):
            model.zero_grad(set_to_none=True)
            loss, metrics = _loss(model, objects, masks, globals_, actions,
                                  regrets, labels, TrackedDAggerConfig())
            loss.backward()
            outcomes.append((loss.detach().clone(), {n:p.grad.clone() for n,p in model.named_parameters()}))
            self.assertNotIn("collision", metrics)
        torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0, atol=0)
        for name in outcomes[0][1]:
            torch.testing.assert_close(outcomes[0][1][name], outcomes[1][1][name], rtol=0, atol=0)
        self.assertFalse(any("collision_head" in n for n, _ in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
