"""Version-12 loading must be explicit and preserve both loading modes."""
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from barrage_rl.checkpoint_loader import load_tracked_agent
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from tools.train_distilled_student import _load_backbone
from barrage_rl.train_tracked_policy import (
    TrackedDAggerConfig, train_tracked_policy, _load_migrated_model_state,
)


@pytest.mark.parametrize("version", [10, 11, None])
@pytest.mark.parametrize("materialize", [False, True])
def test_common_loader_rejects_old_or_missing_version_before_building(version, materialize):
    checkpoint = {} if version is None else {"model_version": version}
    with pytest.raises(ValueError, match="require model_version 12"):
        load_tracked_agent("unused.pt", torch.device("cpu"),
                           checkpoint_data=checkpoint, materialize_state=materialize)


@pytest.mark.parametrize("version", [10, 11, None])
def test_distillation_rejects_old_or_missing_version(version):
    checkpoint = {} if version is None else {"model_version": version}
    with patch("tools.train_distilled_student.torch.load", return_value=checkpoint):
        with pytest.raises(ValueError, match="require model_version 12"):
            _load_backbone("unused.pt", torch.device("cpu"))


@pytest.mark.parametrize("version", [10, 11, None])
def test_training_rejects_old_or_missing_version_before_creating_output(tmp_path, version):
    source = tmp_path / "input.pt"
    output = tmp_path / "output"
    torch.save({} if version is None else {"model_version": version}, source)
    with pytest.raises(ValueError, match="require model_version 12"):
        train_tracked_policy(TrackedDAggerConfig(
            initial_checkpoint=str(source), output_dir=str(output), smoke_test=True, rounds=0,
        ))
    assert not output.exists()


def test_training_state_copy_rejects_risk_head():
    model = ActionQueryPolicy(TrackedPolicySpec(max_objects=4),
                              width=16, attention_layers=1, attention_heads=2)
    with pytest.raises(ValueError, match="Unexpected model parameters"):
        _load_migrated_model_state(model, {**model.state_dict(), "collision_head.bias": torch.zeros(4)})


@pytest.mark.parametrize("materialize", [False, True])
def test_current_checkpoint_preserves_outputs_and_rejects_extra_head(materialize):
    torch.set_num_threads(1)
    spec = TrackedPolicySpec(max_objects=4)
    hparams = dict(width=16, attention_layers=1, attention_heads=2)
    model = ActionQueryPolicy(spec, **hparams).eval()
    checkpoint = dict(model_version=12, model=model.state_dict(),
                      tracked_policy_spec=asdict(spec), model_hparams=hparams)
    loaded, _, _ = load_tracked_agent("unused.pt", torch.device("cpu"),
                                     checkpoint_data=checkpoint, materialize_state=materialize)
    inputs = (torch.rand(2, 4, 16), torch.ones(2, 4, dtype=torch.bool), torch.rand(2, 16))
    with torch.inference_mode():
        expected, actual = model(*inputs), loaded.model(*inputs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    checkpoint["model"] = {**checkpoint["model"], "collision_head.bias": torch.zeros(4)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_tracked_agent("unused.pt", torch.device("cpu"),
                           checkpoint_data=checkpoint, materialize_state=materialize)
