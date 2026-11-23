"""Regression and CUDA integration tests for V6 survival improvements."""
import math
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carrywm.model import Actor, CarryGeometry, Critic, DT, WorldModel
from carrywm.replay_v4 import VectorReplay
from carrywm.sim import CarryEnv, FALL_HEIGHT_ERR_M, FALL_MIN_HEIGHT_M, FALL_TILT_ERR
from carrywm.train import GAMMA, HORIZON, OBS_NOISE_BASE, OFFSETS
from carrywm.v5 import soft_termination_probability
from carrywm.v6 import (
    BALANCE_REWARD_WEIGHT,
    GRAVITY_Z_THRESHOLD,
    HEIGHT_THRESHOLD_M,
    TERMINATION_SOFTNESS,
    LearnerV6,
    teacher_schedule,
    v6_tracking_weights,
    v6_value_max,
)
from carrywm.warp_sim import finish_step_gpu


class V6UnitTests(unittest.TestCase):
    def test_relative_fall_allows_deep_bend_and_catches_real_collapse(self):
        # Deep bend matching reference pelvis r22 = -0.10 (< 0.35), z = 0.64: must NOT fall.
        n = 4
        states = torch.zeros((n, 84), dtype=torch.float32)
        states[:, 2] = torch.tensor([0.64, 0.78, 0.12, 0.78])
        states[0, 3:7] = torch.tensor([math.sqrt(0.45), math.sqrt(0.55), 0.0, 0.0])  # r22 = -0.10
        states[1, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        states[2, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # World 3: reference upright (1.0), robot tilted to r22 = 0.25 -> |0.25 - 1.0| = 0.75 > 0.60 -> fallen.
        states[3, 3:7] = torch.tensor([math.sqrt(0.625), math.sqrt(0.375), 0.0, 0.0])
        states[:, 71:74] = torch.tensor([0.0, 0.0, 0.85])
        states[:, 74] = 1.0

        contacts = torch.zeros((n, 6), dtype=torch.float32)
        support = torch.zeros((n, 3), dtype=torch.bool)
        initial_box = states[:, 71:74].clone()
        goal = torch.full((n, 3), 2.0)
        goal_on_floor = torch.zeros(n, dtype=torch.bool)
        env_index = torch.zeros(n, dtype=torch.int32)
        clip_len = torch.full((n,), 20, dtype=torch.int32)
        flags = torch.zeros(n, dtype=torch.bool)
        counts = torch.zeros(n, dtype=torch.int32)
        ref_z = torch.tensor([0.65, 0.40, 0.78, 0.78], dtype=torch.float32)  # World 1: |0.78 - 0.40| > 0.30
        ref_r22 = torch.tensor([-0.10, 1.0, 1.0, 1.0], dtype=torch.float32)

        *_, fallen, _ = finish_step_gpu(
            states, contacts, support, initial_box, goal, goal_on_floor,
            env_index, clip_len, flags, counts, counts, counts,
            ref_z=ref_z, ref_r22=ref_r22,
        )
        self.assertEqual(fallen.tolist(), [False, True, True, True])
        self.assertLess(FALL_MIN_HEIGHT_M, 0.20)
        self.assertAlmostEqual(FALL_HEIGHT_ERR_M, 0.30)
        self.assertAlmostEqual(FALL_TILT_ERR, 0.60)

    def test_v6_soft_termination_tighter_tilt_and_softness(self):
        state = torch.zeros(2, 84)
        target = torch.zeros(2, 111)
        state[:, 2] = target[:, 99] = 0.75
        state[:, 3] = target[:, 100] = 1.0
        # tilt error = 0.45 -> exactly at V6 midpoint p = 0.5, whereas V5 (0.8) gave ~0.
        state[0, 4] = math.sqrt(0.45 / 2.0)
        state[0, 3] = math.sqrt(1.0 - 0.45 / 2.0)
        p_v6 = soft_termination_probability(
            state, target, height_threshold=HEIGHT_THRESHOLD_M,
            gravity_z_threshold=GRAVITY_Z_THRESHOLD, softness=TERMINATION_SOFTNESS,
        )
        p_v5 = soft_termination_probability(state, target)
        self.assertAlmostEqual(float(p_v6[0].item()), 0.5, places=5)
        self.assertLess(float(p_v5[0].item()), 1e-3)

    def test_critic_running_norm_and_geometry_weights_and_obs_noise(self):
        critic = Critic(16, hidden=(16,), normalize=True)
        self.assertIsNotNone(critic.inputs)
        s = torch.randn(8, 84) * 5.0
        g = torch.randn(8, 11, 113) * 3.0
        c = torch.randn(8, 16)
        before_mean = critic.inputs.mean.clone()
        critic.update_stats(s, g, c)
        self.assertFalse(torch.allclose(critic.inputs.mean, before_mean))
        out = critic(s, g, c)
        self.assertEqual(out.shape, (8,))

        self.assertEqual(len(OBS_NOISE_BASE), 141)
        self.assertTrue(all(x == 0.0 for x in OBS_NOISE_BASE[79:108]))

        w = v6_tracking_weights(0.5)
        self.assertEqual(w["anchor_pos"], 0.5)
        self.assertEqual(w["body_rot"], 0.5)
        self.assertAlmostEqual(v6_value_max(0.5), DT * sum(w.values()) / (1.0 - GAMMA))

        sim_root = os.getenv("SIM2SIM_ROOT")
        if sim_root and Path(sim_root).exists():
            geo = CarryGeometry(sim_root, tracking_weights=w, action_rate_weight=-0.25, action_l2_weight=-0.005)
            prev_s = torch.zeros(2, 84)
            prev_s[:, [3, 74]] = 1.0
            act = torch.ones(2, 29) * 2.0
            tgt = torch.zeros(2, 111)
            tgt[:, [100, 107]] = 1.0
            terms = geo.reward_terms(prev_s, prev_s, torch.zeros(2, 29), act, tgt, torch.zeros(2, 6))
            self.assertIn("action_l2", terms)
            self.assertLess(float(terms["action_l2"].mean().item()), 0.0)

    def test_replay_teacher_batch_and_legacy_unpickle(self):
        rep = VectorReplay(capacity=40, seed=11, num_envs=2)
        clip0 = np.full((60, 111), 1.0, np.float32)
        clip1 = np.full((60, 111), 2.0, np.float32)
        for t in range(10):
            s = np.full((2, 84), float(t), np.float32)
            a = np.zeros((2, 29), np.float32)
            a[0] = float(10 + t)  # World 0 = teacher action marker
            a[1] = -99.0          # World 1 = student action (must not appear in teacher_batch)
            obs = np.full((2, 141), float(t), np.float32)
            rep.append(
                s, s + 1, a, a, np.zeros((2, 6), np.float32), np.zeros((2, 16), np.float32),
                obs, np.array([t, t], np.int32), [clip0, clip1], np.array([t == 0, t == 0]),
                is_teacher=np.array([True, False]),
            )
        batch = rep.teacher_batch(16, OFFSETS)
        self.assertIsNotNone(batch)
        ts, th, tr, ta = batch
        self.assertEqual((ts.shape, th.shape, tr.shape, ta.shape),
                         ((16, 84), (16, 10, 141), (16, 11, 111), (16, 29)))
        self.assertTrue((ta >= 10.0).all())
        self.assertTrue((tr[:, :, 0] == 1.0).all())

        # Unpickle a synthetic legacy NumPy VectorReplay (without clip_bank/seed/is_teacher).
        legacy_obj = VectorReplay.__new__(VectorReplay)
        legacy_state = {
            "num_envs": 2,
            "rows": 12,
            "capacity": 24,
            "rng": np.random.default_rng(3),
            "ticks": 9,
            "_first": np.zeros(2, np.int64),
            "_restart": False,
            "states": np.zeros((12, 2, 84), np.float32),
            "after": np.zeros((12, 2, 84), np.float32),
            "actions": np.zeros((12, 2, 29), np.float32),
            "previous": np.zeros((12, 2, 29), np.float32),
            "contacts": np.zeros((12, 2, 6), np.float32),
            "contexts": np.zeros((12, 2, 16), np.float32),
            "observations": np.zeros((21, 2, 141), np.float32),
            "frames": np.zeros((12, 2), np.int32),
            "episode_start": np.zeros((12, 2), np.int64),
            "clips": [[clip0 if r < 9 else None, clip1 if r < 9 else None] for r in range(12)],
        }
        legacy_obj.__dict__.update(legacy_state)
        probe = VectorReplay.__new__(VectorReplay)
        probe.__setstate__(legacy_state)
        self.assertEqual(probe.valid_windows(8), 4)
        self.assertIsNone(probe.teacher_batch(4, OFFSETS))

    def test_learner_v6_bc_update_reduces_loss(self):
        torch.manual_seed(0)
        net = LearnerV6.__new__(LearnerV6)
        net.device = torch.device("cpu")
        net.args = SimpleNamespace(
            precision="fp32", goal_mode="full", drop_penalty=1.0,
            actor_microbatch=4, critic_chunk=64, checkpoint_encoder=False,
            termination_height_m=HEIGHT_THRESHOLD_M,
            termination_gravity_z=GRAVITY_Z_THRESHOLD,
            termination_softness=TERMINATION_SOFTNESS,
            critic_value_max=13.2,
            teacher_bc_weight=0.5,
        )
        net.offsets = torch.tensor(OFFSETS)
        net.actor = Actor(first_width=16, decoder_width=16, hidden=(8,))
        net.wm = WorldModel(16, width=16)
        net.critic = Critic(16, hidden=(8,), normalize=True)
        net.models = [net.wm, net.actor, net.critic]
        net.optimizers = [torch.optim.Adam(m.parameters(), lr=1e-2) for m in net.models]
        net._obs_noise_std = torch.zeros(141)
        net.geo = SimpleNamespace(
            fk_from_target=lambda targets: targets[:, :3],
            forward_kinematics=lambda state: state[:, :3],
            reference=lambda state, raw, fk=None: torch.nn.functional.pad(raw, (0, 2)),
            observation=lambda state, action, ctx, fk=None: state.new_zeros(len(state), 141),
            reward=lambda before, *args, **kwargs: before.new_ones(len(before)),
        )

        rep = VectorReplay(capacity=32, seed=5, num_envs=2)
        clip = np.zeros((100, 111), np.float32)
        clip[:, [100, 107]] = 1.0
        target_act = np.full((2, 29), 1.5, np.float32)
        for t in range(8):
            s = np.zeros((2, 84), np.float32)
            s[:, [3, 74]] = 1.0
            s[:, 2] = 0.78
            rep.append(
                s, s, np.zeros((2, 29), np.float32), target_act,
                np.zeros((2, 6), np.float32), np.zeros((2, 16), np.float32),
                np.ones((2, 141), np.float32) * 0.2, np.full(2, t, np.int32),
                [clip, clip], np.array([t == 0, t == 0]), is_teacher=np.array([True, True]),
            )

        losses = []
        for _ in range(5):
            m = net.policy_update(rep, 4, bc_weight=5.0, bc_batch=4)
            losses.append(m["actor/bc_loss"])
        self.assertLess(losses[-1], losses[0])

    def test_v6_cli_and_teacher_schedule(self):
        from carrywm import train_v4, train_v5, train_v6
        self.assertAlmostEqual(teacher_schedule(1, 0.25, 400), 0.25)
        self.assertAlmostEqual(teacher_schedule(201, 0.25, 400), 0.125)
        self.assertEqual(teacher_schedule(401, 0.25, 400), 0.0)

        flags = ["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."]
        args = train_v6.args_from_cli(flags)
        self.assertAlmostEqual(args.termination_gravity_z, 0.45)
        self.assertAlmostEqual(args.termination_softness, 0.08)
        self.assertAlmostEqual(args.balance_reward_weight, 0.5)
        self.assertAlmostEqual(args.critic_value_max, 13.2)
        self.assertEqual(args.eval_episodes, 6)
        self.assertTrue(args.critic_norm)

        ck = {"version": 6, "iteration": 50, "config": vars(args).copy()}
        train_v4.check_resume(ck, args, version=6)
        with self.assertRaisesRegex(ValueError, "V6.*V5"):
            train_v4.check_resume(ck, train_v5.args_from_cli(flags), version=5)
        with patch("carrywm.train_v4.train") as runner:
            train_v6.train(args)
        runner.assert_called_once_with(args, version=6, learner_type=LearnerV6)


@unittest.skipUnless(
    os.getenv("CARRY_SIM_ROOT")
    and os.getenv("CARRY_MOTION_DIR")
    and os.getenv("CARRY_WARP_DEVICE", "").startswith("cuda")
    and torch.cuda.is_available(),
    "set CARRY_SIM_ROOT, CARRY_MOTION_DIR, CARRY_WARP_DEVICE=cuda:N on a CUDA machine",
)
class V6CudaIntegrationTests(unittest.TestCase):
    def test_warp_teacher_parity_and_v6_one_round_gpu(self):
        from carrywm.collector_v4 import VectorCollector
        from carrywm.train_v3 import wm_iteration
        from carrywm.train_v6 import args_from_cli
        from carrywm.warp_sim import WarpCarryWorlds

        root = Path(os.environ["CARRY_SIM_ROOT"])
        motion = Path(os.environ["CARRY_MOTION_DIR"])
        device = os.environ["CARRY_WARP_DEVICE"]

        # 1. Parity between WarpCarryWorlds(teacher_history=True) and CPU CarryEnv
        w = WarpCarryWorlds(root, motion, num_envs=4, seed=17, device=device, cfgen_ratio=0.0, teacher_history=True)
        cpu = CarryEnv(root, motion, seed=17, split="train", cfgen_ratio=0.0)
        w.reset(range(4), source="npz")
        cpu.reset(source="npz")
        a_gpu0 = w.envs[0].teacher_action()
        a_cpu0 = cpu.teacher_action()
        np.testing.assert_allclose(a_gpu0, a_cpu0, atol=1e-4, rtol=1e-4)

        for _ in range(15):
            a_t = w.envs[0].teacher_action()
            acts = torch.zeros((4, 29), dtype=torch.float32, device=device)
            acts[0] = torch.as_tensor(a_t, dtype=torch.float32, device=device)
            acts[1] = torch.as_tensor(w.envs[1].teacher_action(), dtype=torch.float32, device=device)
            next_refs = torch.as_tensor(
                np.stack([w.envs[i].clip[min(w.envs[i].index + 1, len(w.envs[i].clip) - 1)] for i in range(4)]),
                dtype=torch.float32, device=device,
            )
            _, _, done_np, _, _ = w.step_gpu(acts, next_ref=next_refs, teacher_ids=[0, 1])
            cpu.step(a_cpu0)
            np.testing.assert_array_equal(w.envs[0].last_action, a_t)
            if done_np[0]:
                break
            a_cpu0 = cpu.teacher_action()
            a_gpu1 = w.envs[0].teacher_action()
            np.testing.assert_allclose(a_gpu1, a_cpu0, atol=0.15)
        w.close()

        # 2. Full V6 learner (compile=True + CUDA Graph + obs_noise + critic_norm + teacher BC)
        with TemporaryDirectory() as tmp:
            args = args_from_cli([
                "--sim-root", str(root),
                "--motion-dir", str(motion),
                "--output-dir", tmp,
                "--device", device,
                "--precision", "bf16",
                "--iterations", "1",
                "--eval-episodes", "1",
                "--seed", "42",
                "--teacher-init-fraction", "0.5",
                "--teacher-decay-iters", "4",
                "--teacher-bc-batch", "32",
            ])
            net = LearnerV6(args)
            rep = VectorReplay(capacity=64, seed=42, num_envs=4, device=device)
            worlds = WarpCarryWorlds(
                root, motion, num_envs=4, seed=42, device=device,
                cfgen_ratio=0.0, use_cuda_graph=True, teacher_history=True,
            )
            col = VectorCollector(worlds, net, rep, np.random.default_rng(42))
            for _ in range(10):
                col.one_tick(noise=0.1, update_normalizers=True, defer_reward=True, episode_teacher_frac=0.5)
            wm_metrics = wm_iteration(net, rep, batch_size=16)
            pol_metrics = net.policy_update(rep, 16, bc_weight=0.5, bc_batch=16)
            self.assertTrue(math.isfinite(wm_metrics["wm/loss"]))
            self.assertTrue(math.isfinite(pol_metrics["actor/return"]))
            self.assertTrue(math.isfinite(pol_metrics["actor/bc_loss"]))
            worlds.close()

        # 3. Official Teacher lifts a box in real MuJoCo CarryEnv under P0 without false bend-falls
        cpu_val = CarryEnv(root, motion, seed=100, split="val", cfgen_ratio=0.0)
        cpu_val.reset(source="npz")
        lifted_seen = False
        for _ in range(min(450, len(cpu_val.clip))):
            a_t = cpu_val.teacher_action()
            _, _, done, info = cpu_val.step(a_t)
            lifted_seen |= bool(info["lifted"])
            self.assertFalse(info["fallen"], f"unexpected false fall at index={cpu_val.index}")
            if lifted_seen or done:
                break
        self.assertTrue(lifted_seen)


if __name__ == "__main__":
    unittest.main()
