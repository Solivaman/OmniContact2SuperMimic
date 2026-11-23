"""Exercise V7's staged starts through the live CUDA collector and replay ring."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.skipif(not os.getenv("CARRY_SIM_ROOT") or not os.getenv("CARRY_WARP_DEVICE", "").startswith("cuda"),
                    reason="requires the released G1 scenes and MuJoCo Warp on CUDA")
def test_student_collects_from_nonzero_reference_frames(tmp_path):
    from carrywm.collector_v4 import VectorCollector
    from carrywm.replay_v4 import VectorReplay
    from carrywm.train_v7 import args_from_cli
    from carrywm.v7 import LearnerV7
    from carrywm.warp_sim import WarpCarryWorlds

    root = Path(os.environ["CARRY_SIM_ROOT"])
    motion = root / "data/carrybox"
    device = os.environ["CARRY_WARP_DEVICE"]
    args = args_from_cli(["--sim-root", str(root), "--motion-dir", str(motion),
                          "--output-dir", str(tmp_path), "--device", device, "--no-compile",
                          "--encoder-width", "32", "--decoder-width", "32", "--wm-width", "32"])
    net = LearnerV7(args)
    worlds = WarpCarryWorlds(root, motion, 8, seed=37, device=device, v7=True, cfgen_ratio=0.)
    try:
        replay = VectorReplay(8 * 32, 37, 8, device=device)
        collector = VectorCollector(worlds, net, replay, np.random.default_rng(37))
        for tick in range(12):
            metrics = collector.one_tick(noise=.1, update_normalizers=tick < 8)
            assert metrics["sim/teacher_fraction"] == 0
            if tick == 0:
                starts = torch.tensor([env.start_index for env in worlds.envs])
                torch.testing.assert_close(replay.frames[0].cpu().long(), starts)
                assert starts.max() > 0
        assert collector.steps == 96 and replay.has_window(8)
        loss_metrics = net.wm_update(replay, 8)
        assert all(np.isfinite(v) for v in loss_metrics.values())
    finally:
        worlds.close()
