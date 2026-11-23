"""The batched CPU pose staging is CUDA-independent; the integration test needs a CUDA runtime."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_batched_host_poses_match_scalar_including_unactuated_coordinates():
    pytest.importorskip("mujoco")
    from carrywm.sim import CarryEnv
    from carrywm.reset_v7 import _normal_poses

    rng = np.random.default_rng(11)
    model = SimpleNamespace(nq=47, nv=46, qpos0=rng.normal(size=47))
    def episode():
        return SimpleNamespace(bp=39,bv=37,m=model,
                               d=SimpleNamespace(qpos=model.qpos0.copy(),qvel=np.zeros(model.nv)),
                               _forward_reset=lambda: None)

    actual, expected = [], []
    for j in range(13):
        f = rng.normal(size=111).astype(np.float32)
        for q in (slice(100,104),slice(107,111)):
            f[q] /= np.linalg.norm(f[q])
        tw = rng.normal(size=(4,3)).astype(np.float32) if j else np.zeros((4,3),np.float32)
        left, right = episode(),episode()
        left._reset_pose = f,tw
        left.d.qpos[:] = -500
        left.d.qvel[:] = 500
        CarryEnv._set_main_pose(right,f,tw)
        actual.append(left)
        expected.append(right)
    _normal_poses(actual)
    _normal_poses([])
    for old,new in zip(expected,actual):
        assert new._reset_pose is None
        np.testing.assert_array_equal(new.d.qpos,old.d.qpos)
        np.testing.assert_allclose(new.d.qvel,old.d.qvel,atol=2e-6,rtol=2e-6)


@pytest.mark.skipif(not os.environ.get("CARRY_SIM_ROOT") or not os.getenv("CARRY_WARP_DEVICE","").startswith("cuda"),
                    reason="requires real MuJoCo Warp CUDA and official carry assets")
def test_real_sparse_and_mixed_resets_keep_the_next_physics_tick_aligned():
    from scripts.check_v7_gpu_reset import check
    root = Path(os.environ["CARRY_SIM_ROOT"])
    motion = Path(os.getenv("CARRY_MOTION_DIR", root/"data/carrybox"))
    assert len(check(root,motion,os.environ["CARRY_WARP_DEVICE"],os.getenv("CARRYWM_CACHE_DIR")))==3
