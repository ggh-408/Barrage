"""The policy imitates recovery cost; the risk head labels fixed action holds."""
import torch

from barrage_rl.train_tracked_policy import _teacher_distribution


def test_recoverable_root_retains_mass_when_its_fixed_hold_collides():
    # A real replay pattern: only root 4 has a low continuation cost, while
    # holding it for 0.1 s collides. Several other roots collide as well.
    regrets = torch.full((1, 9), 20.)
    regrets[0, 4] = 0.
    collisions = torch.zeros(1, 4, 9)
    collisions[0, 0, [1, 2, 4, 6]] = 1.
    target = _teacher_distribution(regrets, collisions, .08)
    assert target.argmax(1).item() == 4
    assert target[0, 4] > .999
    assert target[0, [1, 2, 6]].sum() < 1e-20
    assert torch.isfinite(target).all()


def test_fixed_hold_labels_do_not_change_continuation_preferences():
    regrets = torch.tensor([[0., .04, .1, .5, 1., 2., 4., 10., 20.]])
    safe = torch.zeros(1, 4, 9)
    unsafe = torch.ones_like(safe)
    mixed = safe.clone()
    mixed[:, :, ::2] = 1.
    expected = torch.softmax(-regrets / .08, 1)
    for targets in (safe, unsafe, mixed):
        torch.testing.assert_close(_teacher_distribution(regrets, targets, .08), expected)


def test_low_temperature_and_all_colliding_targets_have_finite_gradients():
    regrets = torch.tensor([[20., 20., 0., 20., 20., 20., 20., 20., 20.]])
    target = _teacher_distribution(regrets, torch.ones(1, 4, 9), 0.)
    logits = torch.zeros_like(regrets, requires_grad=True)
    loss = torch.nn.functional.kl_div(logits.log_softmax(1), target, reduction='batchmean')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] < 0
    assert torch.all(logits.grad[0, torch.arange(9) != 2] > 0)
