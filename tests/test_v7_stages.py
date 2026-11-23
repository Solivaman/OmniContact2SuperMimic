"""Regression checks for reference-frame alignment, reset physics and curriculum bins."""
import os
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from carrywm.stages_v7 import (BINS, ClipStarts, NoStageStart, StageSampler, _stage_labels,
                               _support_starts, generated_twists)
from carrywm.data import DT


class V7StagesTests(TestCase):
    def test_half_uniform_mixture_conditioned_on_eligible_bins_and_failures(self):
        sampler = StageSampler()
        rng = np.random.default_rng(11)
        frames = np.arange(BINS)
        bins = tuple(frames[i:i+1] if i in (2, 12) else frames[:0] for i in range(BINS))
        cat = ClipStarts(frames[[2, 12]], np.zeros(BINS, int), bins, None)
        for _ in range(100):
            sampler.finish("npz", 0, 2, BINS, False)
            sampler.finish("npz", 12, 15, BINS, True)
        draws = [sampler.choose(cat, "npz", rng) for _ in range(1500)]
        self.assertEqual(set(draws), {2, 12})
        self.assertAlmostEqual(np.mean(np.array(draws) == 2), .75, delta=.04)
        restored = StageSampler()
        restored.load_state_dict(sampler.state_dict())
        np.testing.assert_array_equal(restored.visits, sampler.visits)
        np.testing.assert_array_equal(restored.failures, sampler.failures)
        with self.assertRaises(NoStageStart):
            restored.choose(cat, "npz", rng, "lift")

    def test_generated_velocities_handle_quaternion_sign_flips_and_world_frame(self):
        frames = np.zeros((30, 111), np.float32)
        t = np.arange(len(frames)) * DT
        frames[:, 97] = t
        frames[:, 106] = 2 * t
        q = np.stack((np.cos(t/2), np.zeros_like(t), np.zeros_like(t), np.sin(t/2)), axis=1)
        q[::2] *= -1
        frames[:, 100:104] = frames[:, 107:111] = q
        twists = generated_twists(frames)
        np.testing.assert_allclose(twists[:, 0, 0], 1, atol=2e-6)
        np.testing.assert_allclose(twists[:, 1, 2], 1, atol=2e-6)
        np.testing.assert_allclose(twists[:, 2, 2], 2, atol=4e-6)
        np.testing.assert_allclose(twists[:, 3, 2], 1, atol=2e-6)

    def test_stage_labels_cover_pickup_place_and_trim_postrelease(self):
        f = np.zeros((200, 111))
        f[50:170, 37:39] = 1.
        f[55:81, 106] = np.linspace(0., .3, 26)
        f[81:170, 106] = .3
        f[80:165, 104] = np.linspace(0, 1, 85)
        f[165:, 104] = 1.
        labels, last = _stage_labels(f)
        self.assertEqual(last, 169)
        np.testing.assert_array_equal(labels[[0, 48, 52, 100, 162]], [0, 1, 2, 3, 4])

    def test_free_box_requires_support_or_hands_and_no_deep_intersection(self):
        f = np.zeros((4,111))
        f[:, 107] = 1.
        f[:, 106] = [.15, .55, .55, .49]
        f[1:, 104] = 2.
        f[2:,37:39] = 1.
        f[1, 106] = .8
        valid = _support_starts(f, [.15]*3, [[2.,0,.3]], [[.3,.3,.1]])
        np.testing.assert_array_equal(valid, [True, False, True, False])

    def test_second_pickup_and_revisited_destination_keep_distinct_stages(self):
        f = np.zeros((300, 111))
        for first in (40, 180):
            f[first:first + 100, 37:39] = 1
            f[first + 10:first + 90, 106] = .30
            f[first:first+100, 104] = np.interp(np.arange(100), [0, 25, 55, 99], [0, 1, 0, 1])
        labels, last = _stage_labels(f)
        np.testing.assert_array_equal(labels[[65, 136, 150, 172, 184, 205, 276]],
                                      [3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(last, 279)

    @skipUnless(os.getenv("CARRY_SIM_ROOT"), "official carry assets required")
    def test_middle_npz_start_restores_recorded_velocity_without_teacher(self):
        from carrywm.sim import CarryEnv, _matrix
        from carrywm.data import load_demo
        root = Path(os.environ["CARRY_SIM_ROOT"])
        env = CarryEnv(root, root/"data/carrybox", split="train", v7=True)
        path = env.paths[0]
        env.paths = (path,)
        scene = load_demo(path, staged=True)
        idx = len(scene.frames)//2
        with patch.object(env.stage_sampler, "catalogue") as cat, \
                patch.object(env.stage_sampler, "choose", return_value=idx), \
                patch("carrywm.sim._teacher", side_effect=AssertionError):
            cat.return_value.labels = np.ones(len(scene.frames), dtype=int)
            env.reset("npz")
        f = scene.frames[idx]
        twists = scene.frame_twists[idx]
        self.assertEqual(env.start_index, idx)
        np.testing.assert_allclose(env.state[:7], f[97:104])
        np.testing.assert_allclose(env.state[36:39], twists[0])
        np.testing.assert_allclose(env.state[39:42], _matrix(f[100:104]).T @ twists[1])
        np.testing.assert_allclose(env.state[78:81], twists[2])
        np.testing.assert_allclose(env.state[81:84], _matrix(f[107:111]).T @ twists[3])
        np.testing.assert_allclose(env._initial_box, scene.frames[0, 104:107])
        np.testing.assert_array_equal(env.last_action, np.zeros(29))
        val = CarryEnv(root, root/"data/carrybox", split="val", v7=True)
        val.reset("npz")
        self.assertEqual(val.index, 0)
        self.assertIsNone(val.stage_sampler)
