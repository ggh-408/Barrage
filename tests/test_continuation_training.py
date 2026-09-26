"""Boundary and checkpoint invariants for deployable continuation labels."""
from dataclasses import asdict, replace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from barrage_rl.artifacts import contents_equal
from barrage_rl.tracked_policy import ActionQueryPolicy, TrackedPolicySpec
from barrage_rl.train_tracked_policy import TrackedReplay, TrackedDAggerConfig
from barrage_rl.evaluate_tracked_policy import load_tracked_agent


class StayAgent:
    def __init__(self): self.reset_ids=[]
    def act_features(self,objects,masks,globals_,**kwargs):
        return np.zeros(len(objects),np.int64)
    def reset_state(self,episode_indices=None):
        if episode_indices is not None: self.reset_ids.extend(episode_indices.tolist())


class ContinuationTrainingTests(unittest.TestCase):
    def test_branch_forces_root_and_measures_horizon_from_snapshot(self):
        from barrage_rl.env import BarrageVisionEnv
        from barrage_rl.tracked_collection import _RenderedRGBObservation
        from barrage_rl.tracked_policy import TrackedFeatureExtractor
        from barrage_rl.parallel_evaluation import ParallelRolloutInitialState, run_parallel_rollout
        spec=TrackedPolicySpec(max_objects=4,tracker_capacity=4,expected_bullet_count=1)
        kwargs=dict(bullet_count=1,targeted_bullet_probability=.10,
                    observation_size=192,randomize_initial_phase=False,max_episode_seconds=120.)
        env=BarrageVisionEnv(**kwargs)
        try:
            env.reset(seed=7)
            env.plane_position[:]=(410.,410.)
            env.bullet_positions=np.asarray([[422.,410.]],np.float32)
            env.bullet_velocities=np.zeros((1,2),np.float32)
            env.bullet_is_targeted=np.zeros(1,bool)
            env.opening_spawned_batches=10
            env.physics_steps=120;env.episode_steps=30
            rendered=_RenderedRGBObservation(env,192)
            detection=rendered.detections(True)
            extractor=TrackedFeatureExtractor(spec)
            features=extractor.reset_detections(detection.bullet_positions,detection.plane_position)
            initial=ParallelRolloutInitialState(env.capture_state(),deepcopy(extractor),
                deepcopy(vars(rendered.semanticizer)),*features,forced_action=2,
                forced_decisions=1,continuation_seconds=1/30)
        finally: env.close()
        agent=StayAgent()
        result=run_parallel_rollout(agent,spec,2,2,71,kwargs,40.,rendered_rgb=True,
            causal_action_delay_steps=0,initial_states=[initial,replace(initial,forced_action=1)])
        self.assertEqual(result.termination_reasons,['collision','time_limit'])
        self.assertLess(result.survival_times[0],1+1/30)
        self.assertAlmostEqual(result.survival_times[1],1+1/30)
        self.assertEqual(set(agent.reset_ids),{0,1})

    def test_uniform_success_labels_cannot_start_formal_head_training(self):
        from tools.train_continuation_value import supervision_summary
        records=[dict(validation=False),dict(validation=True)]
        labels=np.ones((2,9,2),np.float32)
        self.assertFalse(supervision_summary(records,labels)['informative'])
        labels[0,0,1]=0
        self.assertFalse(supervision_summary(records,labels)['informative'])
        labels[1,0,1]=0
        self.assertTrue(supervision_summary(records,labels)['informative'])

    def test_disabled_continuation_is_exact_and_original_risk_heads_are_preserved(self):
        torch.manual_seed(7)
        spec=TrackedPolicySpec(max_objects=4,tracker_capacity=4)
        base=ActionQueryPolicy(spec,width=16,attention_heads=4,attention_layers=1).eval()
        candidate=ActionQueryPolicy(spec,width=16,attention_heads=4,attention_layers=1,
            continuation_horizons=(.6,1.2),continuation_weight=0).eval()
        missing,extra=candidate.load_state_dict(base.state_dict(),strict=False)
        self.assertFalse(extra)
        self.assertTrue(all(k.startswith('continuation_head.') for k in missing))
        objects=torch.rand(2,4,16); masks=torch.ones(2,4,dtype=torch.bool); globals_=torch.rand(2,16)
        with torch.no_grad():
            old=base(objects,masks,globals_);disabled=candidate(objects,masks,globals_)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(old,disabled)))
            candidate.continuation_weight=1.
            enabled=candidate(objects,masks,globals_)
        self.assertTrue(torch.equal(old[1],enabled[1]))
        self.assertTrue(torch.equal(old[2],enabled[2]))
        self.assertFalse(torch.equal(old[0],enabled[0]))

    def test_behavior_provenance_survives_replay_roundtrip(self):
        spec=TrackedPolicySpec(max_objects=2)
        replay=TrackedReplay(2,spec,4)
        replay.add(np.zeros((2,2,16)),np.ones((2,2),bool),np.zeros((2,16)),
            np.array([1,2]),np.zeros((2,9)),np.zeros((2,4,9)),np.array([10,11]),
            behavior_actions=np.array([3,4]),exploration=np.array([False,True]))
        with TemporaryDirectory() as folder:
            path=Path(folder)/'replay.npz';replay.save(path)
            loaded=TrackedReplay.load(path,2,spec,4)
        np.testing.assert_array_equal(loaded.actions,[1,2])
        np.testing.assert_array_equal(loaded.behavior_actions,[3,4])
        np.testing.assert_array_equal(loaded.exploration,[False,True])

    def test_continuation_checkpoint_loads_for_the_common_agent(self):
        spec=TrackedPolicySpec(max_objects=4,tracker_capacity=4)
        hparams=dict(width=16,attention_heads=4,attention_layers=1,
            continuation_horizons=(.6,1.2),continuation_weight=1.)
        model=ActionQueryPolicy(spec,**hparams)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            torch.save(dict(model=model.state_dict(),tracked_policy_spec=asdict(spec),
                model_hparams=hparams,model_version=model.model_version,config=asdict(TrackedDAggerConfig())),path)
            loaded,_,_=load_tracked_agent(str(path),torch.device('cpu'))
        self.assertTrue(contents_equal(model.state_dict(),loaded.model.state_dict()))
        self.assertEqual(loaded.model.continuation_weight,1.)

    def test_new_collection_defaults_are_natural_deployed_trajectories(self):
        config=TrackedDAggerConfig()
        self.assertEqual(config.pixel_guard,'receding')
        self.assertEqual(config.random_action_probability,0.)
        self.assertEqual(config.teacher_reaction_seconds,.10)
        self.assertEqual((config.bullet_count,config.max_objects,config.tracker_capacity),(300,384,384))

if __name__=='__main__':unittest.main()
