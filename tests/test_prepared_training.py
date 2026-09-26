"""Prepared warm-start accepts only the controller that the entry installs."""
import pytest

from tools import train_targeted_dagger as entry


@pytest.fixture
def prepared_validator(monkeypatch):
    # Register every process-local installation change for automatic restoration.
    for name in (
        '_validate_plain_dagger_warm_start', 'TrackedPolicySpec', '_checkpoint',
        '_MANIFEST_SOURCE_FILES', 'atomic_write_json',
    ):
        monkeypatch.setattr(entry.training, name, getattr(entry.training, name))
    monkeypatch.setattr(entry.evaluation, 'load_tracked_agent', entry.evaluation.load_tracked_agent)
    monkeypatch.setattr(entry.deployment, 'configure_image_controller', entry.deployment.configure_image_controller)
    entry.install_runtime()
    return entry.training._validate_plain_dagger_warm_start


def test_prepared_default_uses_window_checkpoint():
    assert entry.TargetedConfig().initial_checkpoint == 'best.pt'
    assert entry.load_config().initial_checkpoint == 'best.pt'


def test_generic_training_still_rejects_experimental_controller():
    with pytest.raises(ValueError, match='composite deployment'):
        entry.training._validate_plain_dagger_warm_start(
            {'experimental_controller': entry.CONTROLLER}, 'best.pt')


def test_prepared_training_accepts_matching_controller(prepared_validator):
    checkpoint = {'experimental_controller': entry.CONTROLLER, 'model': {}}
    prepared_validator(checkpoint, 'best.pt')
    assert checkpoint['experimental_controller'] == entry.CONTROLLER


@pytest.mark.parametrize('checkpoint', [
    {},
    {'experimental_controller': 'unmatched'},
    {'experimental_controller': entry.CONTROLLER, 'distilled_student': {}},
    {'experimental_controller': entry.CONTROLLER, 'inference_head': 'student'},
])
def test_prepared_training_rejects_other_action_semantics(prepared_validator, checkpoint):
    with pytest.raises(ValueError):
        prepared_validator(checkpoint, 'best.pt')
