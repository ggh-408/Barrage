"""Tracked DAgger loss and CLI tests."""

import json
import unittest
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

import torch

from barrage_rl.train_tracked_policy import (
    PROJECT_ROOT,
    TrackedDAggerConfig,
    TrackedReplay,
    _build_cli_parser,
    _boost_failed_episode_tail,
    _checkpoint,
    _checkpoint_selection_key,
    _should_promote_checkpoint,
    _collection_priorities,
    _configure_trainable_scope,
    _config_from_args,
    _episode_validation_mask,
    _finalize_round,
    _loss,
    _MANIFEST_SOURCE_FILES,
    _next_round_index,
    _prioritized_epoch_indices,
    _renamespace_initial_replay_episodes,
    _restore_selected_checkpoint,
    _resolve_run_seeds,
    _validate_plain_dagger_warm_start,
    _validate_resume_plain_dagger_warm_start,
    train_tracked_policy,
)
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec


class TrainTrackedPolicyTests(unittest.TestCase):
    def test_checkpoint_promotion_prefers_new_on_all_ties(self) -> None:
        for candidate, incumbent, expected in (
            (1.0, 1.0, True), (0.995, 0.995, True),
            (0.0, 0.0, True), (1.0, 0.995, True),
            (0.995, 1.0, False), (0.9, 0.8, True),
        ):
            with self.subTest(candidate=candidate, incumbent=incumbent):
                self.assertEqual(_should_promote_checkpoint((candidate,), (incumbent,)), expected)

    def test_resume_nonperfect_tie_uses_latest_round_even_with_unordered_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            history = []
            for round_index, success in ((2, .995), (0, .995), (3, .99), (1, .995)):
                candidate = output / f'round{round_index}' / 'candidate.pt'
                candidate.parent.mkdir()
                candidate.write_bytes(f'round{round_index}'.encode())
                history.append(dict(round=round_index, success_at_limit=success))
            result = _restore_selected_checkpoint(output, history, 'success_at_limit', 120.)
            self.assertEqual(result, (.995,))
            self.assertEqual((output / 'best.pt').read_bytes(), b'round2')
            self.assertEqual((output / 'latest.pt').read_bytes(), b'round2')
            self.assertEqual(json.loads((output / 'best_summary.json').read_text())['round'], 2)
            self.assertEqual([r['checkpoint_promoted'] for r in history], [1., 1., 0., 1.])

    def test_regret_priority_targets_recoverable_costly_actions(self) -> None:
        config = TrackedDAggerConfig(
            priority_mode="behavior_regret",
            regret_priority=4.0,
            unsafe_behavior_priority=8.0,
            early_state_priority=1.0,
            late_state_priority=1.0,
        )
        actions = np.asarray([1, 2, 3])
        teachers = np.asarray([0, 0, 0])
        regrets = np.zeros((3, 9), dtype=np.float32)
        regrets[1:, actions[1:]] = 20.0
        collisions = np.zeros((3, 4, 9), dtype=np.bool_)
        collisions[1, 2:, 2] = True
        collisions[2] = True
        priorities = _collection_priorities(
            actions, teachers, regrets, collisions, np.zeros(3), config
        )
        self.assertEqual(priorities[0], 1.0)
        self.assertEqual(priorities[1], 5.0)
        self.assertEqual(priorities[2], 1.0)

    def test_imported_replay_episode_ids_are_disjoint_and_grouped(self) -> None:
        spec = TrackedPolicySpec(max_objects=2)
        replay = TrackedReplay(5, spec, 4)
        replay.size = 5
        replay.episode_ids[:5] = np.asarray([20, 10, 20, 30, 10])
        count = _renamespace_initial_replay_episodes(replay)
        self.assertEqual(count, 3)
        self.assertTrue(np.all(replay.episode_ids[:5] < 0))
        self.assertEqual(replay.episode_ids[0], replay.episode_ids[2])
        self.assertEqual(replay.episode_ids[1], replay.episode_ids[4])
        self.assertEqual(len(np.unique(replay.episode_ids[:5])), 3)

    def test_manifest_sources_match_the_shared_runtime(self) -> None:
        self.assertIn("barrage_rl/runtime_core.py", _MANIFEST_SOURCE_FILES)
        self.assertNotIn("barrage_rl/dynamics.py", _MANIFEST_SOURCE_FILES)
        self.assertTrue(all(
            (PROJECT_ROOT / path).is_file() for path in _MANIFEST_SOURCE_FILES
        ))

    def test_new_run_allocates_disjoint_collection_and_evaluation_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = TrackedDAggerConfig(
                output_dir=str(Path(directory) / "new-run"),
                rounds=3,
                num_envs=4,
                evaluation_episodes=8,
            )
            _resolve_run_seeds(config)
        self.assertIsInstance(config.collection_seed, int)
        self.assertIsInstance(config.evaluation_seed, int)
        collection = {
            config.collection_seed,
            *(
                config.collection_seed + round_index * 1_000_000 + env_index
                for round_index in range(1, config.rounds + 1)
                for env_index in range(config.num_envs)
            ),
        }
        evaluation = set(range(
            config.evaluation_seed,
            config.evaluation_seed + config.evaluation_episodes,
        ))
        self.assertTrue(collection.isdisjoint(evaluation))

    def test_cli_exposes_collection_exploration_probability(self) -> None:
        args = _build_cli_parser().parse_args([
            "--random-action-probability", "0",
        ])
        config = _config_from_args(args)
        self.assertEqual(config.random_action_probability, 0.0)

    def test_exact_teacher_reaction_duration_defaults_to_three_steps(self) -> None:
        self.assertEqual(TrackedDAggerConfig().teacher_reaction_seconds, 0.10)
        default_config = _config_from_args(_build_cli_parser().parse_args([]))
        self.assertEqual(default_config.teacher_reaction_seconds, 0.10)
        self.assertEqual(default_config.bullet_count, 300)
        self.assertEqual(default_config.evaluation_bullet_count, 300)
        override = _config_from_args(_build_cli_parser().parse_args([
            "--evaluation-bullets", "350",
        ]))
        self.assertEqual(override.evaluation_bullet_count, 350)
        self.assertEqual(default_config.max_objects, 384)
        self.assertEqual(default_config.tracker_capacity, 384)
        self.assertEqual(default_config.evaluation_episodes, 200)
        explicit_config = _config_from_args(_build_cli_parser().parse_args([
            "--teacher-reaction-seconds", "0.45",
        ]))
        self.assertEqual(explicit_config.teacher_reaction_seconds, 0.45)

    def test_explicit_collection_and_evaluation_seed_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = TrackedDAggerConfig(
                output_dir=str(Path(directory) / "new-run"),
                rounds=0,
                collection_seed=1234,
                evaluation_seed=1234,
            )
            with self.assertRaisesRegex(ValueError, "evaluation_seed reuses"):
                _resolve_run_seeds(config)

    def test_new_run_rejects_seed_assigned_to_historical_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "runs" / "visual_set_v1"
            previous.mkdir(parents=True)
            (previous / "config.json").write_text(
                '{"collection_seed": 1234, "rounds": 0, "num_envs": 2, '
                '"evaluation_seed": 5678, "evaluation_episodes": 2}',
                encoding="utf-8",
            )
            config = TrackedDAggerConfig(
                output_dir=str(root / "runs" / "visual_set_v2"),
                rounds=0,
                collection_seed=1234,
                evaluation_seed=9000,
            )
            with patch(
                "barrage_rl.train_tracked_policy.PROJECT_ROOT", root
            ), self.assertRaisesRegex(ValueError, "collection_seed reuses"):
                _resolve_run_seeds(config)

    def test_historical_evaluation_seed_can_be_an_explicit_failure_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "runs" / "visual_set_v1"
            previous.mkdir(parents=True)
            (previous / "config.json").write_text(
                '{"evaluation_seed": 5678, "evaluation_episodes": 2}',
                encoding="utf-8",
            )
            config = TrackedDAggerConfig(
                output_dir=str(root / "runs" / "visual_set_v2"),
                rounds=1,
                num_envs=4,
                collection_seed_list=(5678,),
            )
            with patch(
                "barrage_rl.train_tracked_policy.PROJECT_ROOT", root
            ):
                _resolve_run_seeds(config)
        self.assertNotEqual(config.collection_seed, 5678)
        self.assertNotIn(
            config.evaluation_seed,
            {5678, config.collection_seed, config.collection_seed + 1_000_000},
        )

    def test_resume_restores_saved_automatic_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing-run"
            output.mkdir()
            (output / "config.json").write_text(
                '{"collection_seed": 8123, "evaluation_seed": 9123, '
                '"collection_seed_list": [101, 202]}',
                encoding="utf-8",
            )
            config = TrackedDAggerConfig(output_dir=str(output), resume=True)
            _resolve_run_seeds(config)
        self.assertEqual(config.collection_seed, 8123)
        self.assertEqual(config.evaluation_seed, 9123)
        self.assertEqual(config.collection_seed_list, (101, 202))

    def test_resume_rejects_evaluation_episode_limit_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing-run"
            output.mkdir()
            config = TrackedDAggerConfig(
                output_dir=str(output),
                initial_checkpoint="",
                initial_replay="",
                collection_seed=8123,
                evaluation_seed=9123,
                evaluation_episode_limit_seconds=120.0,
                smoke_test=True,
                resume=True,
            )
            saved_config = asdict(config)
            saved_config["evaluation_episode_limit_seconds"] = 90.0
            (output / "config.json").write_text(
                json.dumps(saved_config, allow_nan=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError,
                "resume configuration mismatch for evaluation_episode_limit_seconds",
            ):
                train_tracked_policy(config)

    def test_resume_rejects_teacher_reaction_semantic_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing-run"
            output.mkdir()
            config = TrackedDAggerConfig(
                output_dir=str(output),
                initial_checkpoint="",
                initial_replay="",
                collection_seed=8123,
                evaluation_seed=9123,
                teacher_reaction_seconds=0.30,
                smoke_test=True,
                resume=True,
            )
            saved_config = asdict(config)
            saved_config["teacher_reaction_seconds"] = 1.0 / 30.0
            (output / "config.json").write_text(
                json.dumps(saved_config, allow_nan=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError,
                "resume configuration mismatch for teacher_reaction_seconds",
            ):
                train_tracked_policy(config)

    def test_episode_holdout_hashes_environment_and_serial(self) -> None:
        ids = np.asarray([
            3 * 10**12 + environment * 10**6 + serial
            for environment in range(108)
            for serial in range(3)
        ], np.int64)
        mask = _episode_validation_mask(ids)
        self.assertGreater(int(mask.sum()), 15)
        self.assertLess(int(mask.sum()), 50)
        # Repeated states from one episode must never cross the split.
        repeated = np.repeat(ids[:20], 7)
        repeated_mask = _episode_validation_mask(repeated).reshape(20, 7)
        self.assertTrue(np.all(repeated_mask == repeated_mask[:, :1]))

    def test_checkpoint_selection_uses_only_success_and_keeps_ties(self) -> None:
        below_limit_lower_iqm = {
            "model_iqm": 118.0,
            "success_at_limit": 0.99,
            "model_mean": 119.9,
        }
        below_limit_higher_iqm = {
            "model_iqm": 119.0,
            "success_at_limit": 0.50,
            "model_mean": 80.0,
        }
        saturated = {
            "model_iqm": 120.0,
            "success_at_limit": 0.90,
            "model_mean": 118.0,
        }
        equal_success_higher_mean = {
            "model_iqm": 120.0,
            "success_at_limit": 0.90,
            "model_mean": 119.0,
        }
        mode = "success_at_limit"
        self.assertGreater(
            _checkpoint_selection_key(saturated, mode, 120.0),
            _checkpoint_selection_key(below_limit_higher_iqm, mode, 120.0),
        )
        self.assertEqual(
            _checkpoint_selection_key(equal_success_higher_mean, mode, 120.0),
            _checkpoint_selection_key(saturated, mode, 120.0),
        )
        self.assertGreater(
            _checkpoint_selection_key(below_limit_lower_iqm, mode, 120.0),
            _checkpoint_selection_key(below_limit_higher_iqm, mode, 120.0),
        )

    def test_next_round_index_uses_maximum_round_including_baseline(self) -> None:
        self.assertEqual(_next_round_index([]), 1)
        self.assertEqual(_next_round_index([{"round": "0.0"}]), 1)
        self.assertEqual(
            _next_round_index([
                {"round": "0.0"},
                {"round": "2.0"},
                {"round": "1.0"},
            ]),
            3,
        )

    def test_round0_baseline_survives_a_degraded_candidate_on_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline = output / "round0" / "candidate.pt"
            degraded = output / "round1" / "candidate.pt"
            baseline.parent.mkdir()
            degraded.parent.mkdir()
            baseline.write_bytes(b"baseline")
            degraded.write_bytes(b"degraded")
            history = [
                {
                    "round": 0.0,
                    "phase": "initial_baseline",
                    "model_iqm": 119.0,
                    "success_at_limit": 0.9,
                    "model_mean": 118.0,
                },
                {
                    "round": 1.0,
                    "phase": "dagger_collection",
                    "model_iqm": 110.0,
                    "success_at_limit": 0.8,
                    "model_mean": 119.0,
                },
            ]

            selected = _restore_selected_checkpoint(
                output,
                history,
                "success_at_limit",
                120.0,
            )

            self.assertEqual(selected, (0.9,))
            self.assertEqual((output / "best.pt").read_bytes(), b"baseline")
            self.assertEqual((output / "latest.pt").read_bytes(), b"baseline")
            self.assertEqual(history[0]["checkpoint_promoted"], 1.0)
            self.assertEqual(history[1]["checkpoint_promoted"], 0.0)

    def test_round0_only_run_saves_replay_and_resumes_refinement(self) -> None:
        def reject_nonfinite(value: str) -> None:
            raise ValueError(f"non-finite JSON constant: {value}")

        evaluation_values = iter((
            (10.0, 5.0, 9.0),
            (11.0, 6.0, 10.0),
        ))

        def fake_evaluation(checkpoint: str, **kwargs: object) -> dict[str, object]:
            iqm, cvar5, mean = next(evaluation_values)
            output_dir = Path(str(kwargs["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            source_model = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            torch.save({"model": source_model}, output_dir / "evaluated_model.pt")
            evaluation_config = {
                "episodes": kwargs["episodes"],
                "seed": kwargs["seed"],
                "episode_limit_seconds": kwargs["episode_limit_seconds"],
                "bullet_count": kwargs["bullet_count"],
                "targeted_bullet_probability": kwargs[
                    "targeted_bullet_probability"
                ],
                "rendered_rgb": kwargs["rendered_rgb"],
                "causal_action_delay_steps": kwargs[
                    "causal_action_delay_steps"
                ],
            }
            summary: dict[str, object] = {
                "model_iqm": iqm,
                "model_cvar5": cvar5,
                "success_at_limit": 0.0,
                "model_mean": mean,
                "episode_limit_seconds": kwargs["episode_limit_seconds"],
            }
            (output_dir / "evaluation_config.json").write_text(
                json.dumps(evaluation_config, allow_nan=False), encoding="utf-8"
            )
            (output_dir / "evaluation_summary.json").write_text(
                json.dumps(summary, allow_nan=False), encoding="utf-8"
            )
            return summary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "backbone.pt"
            output = root / "runs" / "visual_set_v44"
            config = TrackedDAggerConfig(
                output_dir=str(output),
                initial_checkpoint=str(source_path),
                initial_replay="",
                rounds=0,
                replay_capacity=8,
                bullet_count=4,
                max_objects=4,
                tracker_capacity=4,
                model_width=16,
                attention_layers=1,
                attention_heads=4,
                evaluation_episodes=2,
                evaluation_workers=1,
                evaluation_batch_size=1,
                smoke_test=True,
                device="cpu",
            )
            spec = TrackedPolicySpec(
                max_objects=4,
                tracker_capacity=4,
                expected_bullet_count=4,
            )
            source_model = ActionQueryPolicy(
                spec,
                width=16,
                attention_layers=1,
                attention_heads=4,
            )
            source_optimizer = torch.optim.AdamW(source_model.parameters())
            torch.save(
                _checkpoint(
                    source_model,
                    source_optimizer,
                    config,
                    spec,
                    2,
                    {"deployment_safety_threshold": 0.4},
                ),
                source_path,
            )

            with patch(
                "barrage_rl.train_tracked_policy.evaluate_tracked_checkpoint",
                side_effect=fake_evaluation,
            ), patch(
                "barrage_rl.train_tracked_policy.save_round_summary_plot"
            ), patch(
                "barrage_rl.train_tracked_policy._train_round",
                return_value=(0, {"deployment_safety_threshold": 0.4}),
            ):
                self.assertEqual(train_tracked_policy(config), output / "best.pt")
                replay_path = output / "replay_latest.npz"
                self.assertTrue(replay_path.is_file())
                restored = TrackedReplay.load(replay_path, 8, spec, 4)
                self.assertEqual(restored.size, 0)
                baseline_summary = json.loads(
                    (output / "best_summary.json").read_text(encoding="utf-8"),
                    parse_constant=reject_nonfinite,
                )
                self.assertEqual(baseline_summary["phase"], "initial_baseline")
                self.assertNotIn(
                    "behavior_teacher_agreement", baseline_summary
                )

                resume_config = TrackedDAggerConfig(**asdict(config))
                resume_config.resume = True
                resume_config.refine_replay_only = True
                resume_config.refinement_epochs = 1
                self.assertEqual(
                    train_tracked_policy(resume_config), output / "best.pt"
                )

            selected_summary = json.loads(
                (output / "best_summary.json").read_text(encoding="utf-8"),
                parse_constant=reject_nonfinite,
            )
            self.assertEqual(selected_summary["phase"], "replay_refinement")
            self.assertNotIn(
                "behavior_teacher_agreement", selected_summary
            )

    def test_initial_evaluation_can_be_skipped_for_a_fresh_warm_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "backbone.pt"
            output = root / "runs" / "visual_set_v45"
            config = TrackedDAggerConfig(
                output_dir=str(output),
                initial_checkpoint=str(source_path),
                initial_replay="",
                evaluate_initial_checkpoint=False,
                rounds=0,
                replay_capacity=8,
                bullet_count=4,
                max_objects=4,
                tracker_capacity=4,
                model_width=16,
                attention_layers=1,
                attention_heads=4,
                evaluation_episodes=2,
                evaluation_workers=1,
                evaluation_batch_size=1,
                smoke_test=True,
                device="cpu",
            )
            spec = TrackedPolicySpec(
                max_objects=4,
                tracker_capacity=4,
                expected_bullet_count=4,
            )
            source_model = ActionQueryPolicy(
                spec,
                width=16,
                attention_layers=1,
                attention_heads=4,
            )
            source_optimizer = torch.optim.AdamW(source_model.parameters())
            torch.save(
                _checkpoint(
                    source_model,
                    source_optimizer,
                    config,
                    spec,
                    2,
                    {"deployment_safety_threshold": 0.4},
                ),
                source_path,
            )

            with patch(
                "barrage_rl.train_tracked_policy.evaluate_tracked_checkpoint"
            ) as evaluation:
                train_tracked_policy(config)

            evaluation.assert_not_called()
            self.assertFalse((output / "round0").exists())
            self.assertFalse((output / "round_summaries.csv").exists())
            saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["evaluate_initial_checkpoint"])

    def test_priority_sampler_retains_uniform_and_favors_rare_states(self) -> None:
        training = np.arange(100, dtype=np.int64)
        priorities = np.ones(100, dtype=np.float32)
        priorities[-1] = 1_000.0
        sampled = _prioritized_epoch_indices(
            training, priorities, 0.5, np.random.default_rng(9)
        )
        self.assertEqual(len(sampled), len(training))
        self.assertGreater(int(np.count_nonzero(sampled == 99)), 10)
        self.assertGreater(len(np.unique(sampled)), 40)

    def test_failed_episode_boosts_only_its_recent_tail(self) -> None:
        spec = TrackedPolicySpec(max_objects=4)
        replay = TrackedReplay(8, spec, 4)
        replay.size = 6
        replay.episode_ids[:6] = np.asarray([10, 10, 10, 10, 11, 11])
        replay.episode_steps[:6] = np.asarray([0, 1, 2, 3, 2, 3])
        replay.priorities[:6] = 1.0
        boosted = _boost_failed_episode_tail(replay, 10, 3, 2, 6.0)
        self.assertEqual(boosted, 2)
        np.testing.assert_array_equal(
            replay.priorities[:6],
            np.asarray([1.0, 1.0, 6.0, 6.0, 1.0, 1.0]),
        )

    def test_loss_trains_only_policy_and_teacher_cost_heads(self) -> None:
        spec = TrackedPolicySpec()
        model = ActionQueryPolicy(spec)
        batch = 2
        objects = torch.randn(batch, spec.max_objects, spec.object_features)
        masks = torch.ones(batch, spec.max_objects, dtype=torch.bool)
        globals_ = torch.randn(batch, spec.global_features)
        actions = torch.tensor([1, 2])
        regrets = torch.rand(batch, 9)
        collisions = torch.zeros(batch, 4, 9)
        collisions[:, :, 0] = 1.0
        loss, metrics = _loss(
            model, objects, masks, globals_, actions, regrets, collisions,
            TrackedDAggerConfig(),
        )
        loss.backward()
        self.assertGreater(model.policy_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.teacher_cost_head.weight.grad.abs().sum().item(), 0.0
        )
        self.assertFalse(hasattr(model, "collision_head"))
        self.assertNotIn("collision", metrics)
        self.assertIn("accuracy", metrics)

    def test_removed_collision_scope_is_rejected(self):
        model = ActionQueryPolicy(TrackedPolicySpec())
        with self.assertRaises(ValueError):
            _configure_trainable_scope(model, "collision_head")

    def test_backbone_scope_freezes_all_output_heads(self) -> None:
        model = ActionQueryPolicy(TrackedPolicySpec())
        selected = _configure_trainable_scope(model, "backbone")
        selected_ids = {id(parameter) for parameter in selected}
        head_prefixes = ("policy_head.", "teacher_cost_head.", "collision_head.")
        for name, parameter in model.named_parameters():
            if name.startswith(head_prefixes):
                self.assertFalse(parameter.requires_grad)
                self.assertNotIn(id(parameter), selected_ids)
            else:
                self.assertTrue(parameter.requires_grad)
                self.assertIn(id(parameter), selected_ids)

    def test_full_scope_unfreezes_all_current_model_parameters(self) -> None:
        model = ActionQueryPolicy(TrackedPolicySpec())
        selected = _configure_trainable_scope(model, "full")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(
            sum(parameter.numel() for parameter in selected),
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertEqual(sum(parameter.numel() for parameter in selected), 753_986)

    def test_plain_dagger_warm_start_accepts_backbone_checkpoint(self) -> None:
        _validate_plain_dagger_warm_start(
            {
                "model": {},
                "tracked_policy_spec": {},
                "model_hparams": {},
                "inference_head": "policy",
                "use_safety_filter": True,
                "safety_threshold": 0.5,
            },
            "backbone.pt",
        )

    def test_plain_dagger_warm_start_rejects_composite_controller_families(self) -> None:
        cases = (
            ("distilled prefix", {"distilled_student": {}}),
            ("distillation prefix", {"distillation_recipe": {}}),
            ("option prefix", {"option_arbiter_spec": {}}),
            ("sequence prefix", {"sequence_gate": {}}),
            ("viability prefix", {"viability_controller_state": {}}),
            ("exact marker", {"deployment_action_modules": {}}),
            ("planner marker", {"planner_fallback": {}}),
            ("generic controller prefix", {"controller_router": {}}),
            ("generic controller suffix", {"fallback_controller": {}}),
            ("non-policy inference head", {"inference_head": "teacher_cost"}),
        )
        for case, checkpoint_fields in cases:
            with self.subTest(case=case), self.assertRaisesRegex(
                ValueError,
                "pre-composition backbone checkpoint",
            ):
                _validate_plain_dagger_warm_start(
                    {"model": {}, **checkpoint_fields}, "composite.pt"
                )

    def test_v43_style_resume_rejects_composite_source_without_round0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runs" / "visual_set_v42" / "best.pt"
            source.parent.mkdir(parents=True)
            torch.save(
                {
                    "model": {},
                    "distilled_student": {},
                    "distilled_planner_fallback": {},
                },
                source,
            )
            output = root / "runs" / "visual_set_v43"
            output.mkdir()
            config = TrackedDAggerConfig(
                output_dir=str(output),
                initial_checkpoint="",
                initial_replay="",
                rounds=2,
                collection_seed=101,
                evaluation_seed=202,
                smoke_test=True,
                resume=True,
            )
            saved_config = asdict(config)
            saved_config["resume"] = False
            saved_config["initial_checkpoint"] = (
                "runs\\visual_set_v42\\best.pt"
            )
            (output / "config.json").write_text(
                json.dumps(saved_config, allow_nan=False), encoding="utf-8"
            )
            (output / "run_manifest.json").write_text(
                json.dumps({"initial_checkpoint": str(source.resolve())}),
                encoding="utf-8",
            )
            (output / "round_summaries.csv").write_text(
                "round,phase,new_samples,model_iqm,success_at_limit,model_mean\n"
                "1.0,dagger_collection,400000.0,106.0,0.575,88.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "no trusted round0 initial_baseline.*runs/visual_set_v46",
            ):
                train_tracked_policy(config)

    def test_resume_guard_allows_legacy_pure_or_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "legacy-run"
            output.mkdir()
            pure_source = root / "backbone.pt"
            torch.save(
                {"model": {}, "inference_head": "policy"}, pure_source
            )
            history = [{
                "round": "1.0",
                "phase": "dagger_collection",
                "model_iqm": "10.0",
                "success_at_limit": "0.0",
                "model_mean": "9.0",
            }]

            _validate_resume_plain_dagger_warm_start(
                output,
                {"initial_checkpoint": str(pure_source)},
                history,
            )
            _validate_resume_plain_dagger_warm_start(
                output,
                {"initial_checkpoint": str(root / "missing-backbone.pt")},
                history,
            )

    def test_resume_guard_rejects_replaced_manifest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "legacy-run"
            output.mkdir()
            source = root / "backbone.pt"
            torch.save({"model": {}, "inference_head": "policy"}, source)
            (output / "run_manifest.json").write_text(
                json.dumps({
                    "initial_checkpoint": str(source.resolve()),
                }),
                encoding="utf-8",
            )

            torch.save({"model": {"changed": torch.ones(1)}}, output / "initial_model_reference.pt")
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                _validate_resume_plain_dagger_warm_start(
                    output,
                    {"initial_checkpoint": str(source)},
                    [],
                )

    def test_resume_guard_accepts_verified_round0_after_composite_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "round0-run"
            output.mkdir()
            source = root / "composite.pt"
            torch.save({"model": {}, "distilled_student": {}}, source)
            (output / "run_manifest.json").write_text(
                json.dumps({"initial_checkpoint": str(source.resolve())}),
                encoding="utf-8",
            )

            candidate = output / "round0" / "candidate.pt"
            evaluation_dir = candidate.parent / "evaluation"
            evaluation_dir.mkdir(parents=True)
            torch.save(
                {"model": {}, "optimizer": {}, "round": 0}, candidate
            )
            torch.save({"model": {}}, evaluation_dir / "evaluated_model.pt")
            saved_config = {
                "initial_checkpoint": str(source),
                "evaluation_episodes": 200,
                "evaluation_seed": 303,
                "evaluation_episode_limit_seconds": 120.0,
                "bullet_count": 300,
                "targeted_bullet_probability": 0.10,
                "deployment_rgb_observation": True,
                "evaluation_causal_action_delay_steps": 0,
            }
            evaluation_config = {
                "episodes": 200,
                "seed": 303,
                "episode_limit_seconds": 120.0,
                "bullet_count": 300,
                "targeted_bullet_probability": 0.10,
                "rendered_rgb": True,
                "causal_action_delay_steps": 0,
            }
            evaluation_summary = {
                "model_iqm": 119.0,
                "success_at_limit": 0.8,
                "model_mean": 118.0,
            }
            (evaluation_dir / "evaluation_config.json").write_text(
                json.dumps(evaluation_config, allow_nan=False), encoding="utf-8"
            )
            (evaluation_dir / "evaluation_summary.json").write_text(
                json.dumps(evaluation_summary, allow_nan=False), encoding="utf-8"
            )
            history = [{
                "round": "0.0",
                "phase": "initial_baseline",
                "new_samples": "0.0",
                "model_iqm": "119.0",
                "success_at_limit": "0.8",
                "model_mean": "118.0",
            }]

            _validate_resume_plain_dagger_warm_start(
                output, saved_config, history
            )

    def test_composite_initial_checkpoint_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "distilled.pt"
            output = root / "new-run"
            torch.save(
                {
                    "model": {},
                    "distilled_student": {},
                    "distilled_student_schema_version": 2,
                    "model_version": 12,
                },
                checkpoint,
            )
            with self.assertRaisesRegex(
                ValueError,
                "corresponding distillation, option, or controller training pipeline",
            ):
                train_tracked_policy(TrackedDAggerConfig(
                    output_dir=str(output),
                    initial_checkpoint=str(checkpoint),
                    initial_replay="",
                    smoke_test=True,
                    rounds=0,
                ))
            self.assertFalse(output.exists())

    def test_smoke_config_reduces_work_explicitly(self) -> None:
        self.assertEqual(TrackedDAggerConfig().output_dir, "runs/visual_set_v52")
        self.assertEqual(
            _build_cli_parser().parse_args([]).output_dir,
            "runs/visual_set_v52",
        )
        self.assertEqual(
            TrackedDAggerConfig().initial_checkpoint,
            "diagnostics/risk_removal_20260920/policy_teacher_cost.pt",
        )
        self.assertEqual(
            TrackedDAggerConfig().initial_replay,
            "",
        )
        self.assertEqual(TrackedDAggerConfig().bullet_count, 300)
        self.assertEqual(TrackedDAggerConfig().max_objects, 384)
        self.assertEqual(TrackedDAggerConfig().batch_size, 512)
        self.assertEqual(TrackedDAggerConfig().learning_rate, 1e-5)
        self.assertEqual(TrackedDAggerConfig().priority_sample_fraction, 0.75)
        self.assertEqual(TrackedDAggerConfig().priority_mode, "action_disagreement")
        self.assertEqual(TrackedDAggerConfig().failure_tail_priority, 12.0)
        self.assertEqual(TrackedDAggerConfig().failure_tail_decisions, 36)
        self.assertEqual(TrackedDAggerConfig().trainable_scope, "full")
        self.assertEqual(
            TrackedDAggerConfig().collection_causal_action_delay_steps, 0
        )
        self.assertEqual(
            TrackedDAggerConfig().evaluation_causal_action_delay_steps, 0
        )
        args = _build_cli_parser().parse_args([
            "--output-dir", ".tmp/tracked_smoke", "--smoke-test"
        ])
        config = _config_from_args(args)
        self.assertTrue(config.smoke_test)
        self.assertEqual(config.evaluation_episodes, 2)
        self.assertEqual(config.rounds, 1)
        self.assertEqual(config.teacher_kind, "exact")
        self.assertFalse(config.bootstrap_with_teacher_behavior)

        delayed_args = _build_cli_parser().parse_args([
            "--output-dir", ".tmp/tracked_delayed_smoke", "--smoke-test",
            "--collection-causal-action-delay-steps", "1",
        ])
        delayed_config = _config_from_args(delayed_args)
        self.assertEqual(delayed_config.collection_causal_action_delay_steps, 1)
        self.assertEqual(delayed_config.evaluation_causal_action_delay_steps, 1)

        warm_smoke_args = _build_cli_parser().parse_args([
            "--output-dir", ".tmp/tracked_warm_smoke",
            "--smoke-test",
            "--skip-initial-evaluation",
            "--smoke-samples", "4096",
            "--smoke-evaluation-episodes", "20",
            "--smoke-episode-seconds", "10",
            "--smoke-num-envs", "16",
            "--smoke-workers", "8",
            "--smoke-epochs", "2",
            "--smoke-batch-size", "128",
        ])
        warm_smoke_config = _config_from_args(warm_smoke_args)
        self.assertEqual(
            warm_smoke_config.initial_checkpoint,
            "diagnostics/risk_removal_20260920/policy_teacher_cost.pt",
        )
        self.assertFalse(warm_smoke_config.evaluate_initial_checkpoint)
        self.assertEqual(warm_smoke_config.samples_per_round, 4096)
        self.assertEqual(warm_smoke_config.replay_capacity, 8192)
        self.assertEqual(warm_smoke_config.evaluation_episodes, 20)
        self.assertEqual(warm_smoke_config.evaluation_episode_limit_seconds, 10.0)
        self.assertEqual(warm_smoke_config.collection_episode_seconds, 10.0)
        self.assertEqual(warm_smoke_config.num_envs, 16)
        self.assertEqual(warm_smoke_config.cpu_workers, 8)
        self.assertEqual(warm_smoke_config.epochs_per_round, 2)
        self.assertEqual(warm_smoke_config.batch_size, 128)

        exact_args = _build_cli_parser().parse_args([
            "--output-dir", ".tmp/tracked_smoke", "--teacher-kind", "exact",
            "--evaluation-workers", "6", "--epochs-per-round", "7",
            "--student-behavior-from-round1",
            "--trainable-scope", "backbone",
            "--collection-seeds", "3400001,3400002",
            "--repeat-collection-seeds",
            "--failure-tail-priority", "7",
            "--failure-tail-decisions", "150",
            "--priority-mode", "behavior_regret",
            "--regret-priority", "5",
            "--priority-cap", "32",
            "--resume", "--refine-replay-only", "--refinement-epochs", "18",
            "--resume-replay", ".tmp/augmented.npz",
            "--skip-initial-evaluation",
        ])
        exact_config = _config_from_args(exact_args)
        self.assertEqual(exact_config.teacher_kind, "exact")
        self.assertEqual(exact_config.evaluation_workers, 6)
        self.assertEqual(exact_config.epochs_per_round, 7)
        self.assertEqual(
            exact_config.selection_mode, "success_at_limit"
        )
        self.assertFalse(exact_config.bootstrap_with_teacher_behavior)
        self.assertEqual(exact_config.trainable_scope, "backbone")
        self.assertEqual(exact_config.collection_seed_list, (3400001, 3400002))
        self.assertTrue(exact_config.repeat_collection_seeds)
        self.assertEqual(exact_config.failure_tail_priority, 7.0)
        self.assertEqual(exact_config.failure_tail_decisions, 150)
        self.assertEqual(exact_config.priority_mode, "behavior_regret")
        self.assertEqual(exact_config.regret_priority, 5.0)
        self.assertEqual(exact_config.priority_cap, 32.0)
        self.assertTrue(exact_config.resume)
        self.assertEqual(exact_config.resume_replay, ".tmp/augmented.npz")
        self.assertFalse(exact_config.evaluate_initial_checkpoint)
        self.assertTrue(exact_config.refine_replay_only)
        self.assertEqual(exact_config.refinement_epochs, 18)

    def test_refinement_requires_resume_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                train_tracked_policy(TrackedDAggerConfig(
                    output_dir=str(output),
                    initial_checkpoint="",
                    initial_replay="",
                    smoke_test=True,
                    refine_replay_only=True,
                    resume=False,
                ))
            self.assertFalse(output.exists())

    def test_repeated_collection_seeds_require_a_seed_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            with self.assertRaisesRegex(
                ValueError, "requires collection_seed_list"
            ):
                train_tracked_policy(TrackedDAggerConfig(
                    output_dir=str(output),
                    initial_checkpoint="",
                    initial_replay="",
                    smoke_test=True,
                    repeat_collection_seeds=True,
                ))
            self.assertFalse(output.exists())

    def test_replay_round_trip_preserves_resume_state(self) -> None:
        spec = TrackedPolicySpec(max_objects=4)
        replay = TrackedReplay(8, spec, 4)
        rng = np.random.default_rng(3)
        replay.add(
            rng.normal(size=(3, 4, 12)).astype(np.float32),
            np.ones((3, 4), np.bool_),
            rng.normal(size=(3, 9)).astype(np.float32),
            np.asarray([1, 2, 3]),
            rng.normal(size=(3, 9)).astype(np.float32),
            np.zeros((3, 4, 9), np.bool_),
            np.asarray([10, 11, 12], np.int64),
            np.asarray([0, 90, 900], np.int32),
            np.asarray([3.0, 1.0, 2.0], np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            restored = TrackedReplay.load(path, 8, spec, 4)
        self.assertEqual(restored.size, replay.size)
        self.assertEqual(restored.position, replay.position)
        np.testing.assert_array_equal(
            restored.episode_ids[:3], replay.episode_ids[:3]
        )
        np.testing.assert_array_equal(
            restored.episode_steps[:3], replay.episode_steps[:3]
        )
        np.testing.assert_array_equal(
            restored.priorities[:3], replay.priorities[:3]
        )

    def test_replay_expansion_appends_after_retained_samples(self) -> None:
        spec = TrackedPolicySpec(max_objects=4)
        replay = TrackedReplay(3, spec, 4)
        rng = np.random.default_rng(17)
        replay.add(
            rng.normal(size=(3, 4, 12)).astype(np.float32),
            np.ones((3, 4), np.bool_),
            rng.normal(size=(3, 9)).astype(np.float32),
            np.asarray([1, 2, 3]),
            rng.normal(size=(3, 9)).astype(np.float32),
            np.zeros((3, 4, 9), np.bool_),
            np.asarray([10, 11, 12], np.int64),
        )
        self.assertEqual(replay.position, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            expanded = TrackedReplay.load(path, 5, spec, 4)
        self.assertEqual(expanded.size, 3)
        self.assertEqual(expanded.position, 3)

    def test_replay_wrap_and_migration_padding_preserve_ring_semantics(self) -> None:
        spec = TrackedPolicySpec(max_objects=4, object_features=16, global_features=16)
        replay = TrackedReplay(5, spec, 4)

        def add(episode_ids: np.ndarray, value_offset: int) -> None:
            count = len(episode_ids)
            objects = np.empty((count, 2, 3), dtype=np.float32)
            globals_ = np.empty((count, 5), dtype=np.float32)
            for index in range(count):
                objects[index].fill(value_offset + index)
                globals_[index].fill(value_offset + index)
            replay.add(
                objects,
                np.ones((count, 2), dtype=np.bool_),
                globals_,
                np.arange(count) % 9,
                np.zeros((count, 9), dtype=np.float32),
                np.zeros((count, 4, 9), dtype=np.bool_),
                episode_ids,
            )

        add(np.asarray([10, 11, 12, 13], dtype=np.int64), 1)
        add(np.asarray([20, 21, 22], dtype=np.int64), 5)
        self.assertEqual(replay.size, 5)
        self.assertEqual(replay.position, 2)
        np.testing.assert_array_equal(
            replay.episode_ids,
            np.asarray([21, 22, 12, 13, 20], dtype=np.int64),
        )
        self.assertTrue(np.all(replay.objects[:, 2:] == 0.0))
        self.assertTrue(np.all(replay.objects[:, :2, 3:] == 0.0))
        self.assertTrue(np.all(~replay.masks[:, 2:]))
        self.assertTrue(np.all(replay.globals[:, 5:] == 0.0))
        self.assertTrue(np.all(replay.episode_steps == 0))
        self.assertTrue(np.all(replay.priorities == 1.0))

    def test_new_checkpoint_contains_only_active_heads(self):
        spec = TrackedPolicySpec(max_objects=4)
        model = ActionQueryPolicy(spec)
        optimizer = torch.optim.AdamW(model.parameters())
        checkpoint = _checkpoint(model, optimizer, TrackedDAggerConfig(), spec, 1, {})
        self.assertEqual(checkpoint["model_version"], 12)
        self.assertNotIn("safety_threshold", checkpoint)
        self.assertNotIn("use_safety_filter", checkpoint)
        self.assertFalse(any("collision_head" in name for name in checkpoint["model"]))
        self.assertNotIn("collision_weight", checkpoint["config"])


if __name__ == "__main__":
    unittest.main()
