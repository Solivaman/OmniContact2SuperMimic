"""Online recovery must produce the exact original V7 plans across scene geometries."""
import os
from pathlib import Path

import numpy as np
import pytest

from carrywm import recovery_fk_v7
from carrywm.data import load_demo
from carrywm.geometry_v7 import generate_v7_clip
from carrywm.sim import _generator


@pytest.fixture(scope="module")
def sim_root():
    for value in (os.environ.get("CARRY_SIM_ROOT"), os.environ.get("SIM2SIM_ROOT"),
                  "../OmniContact_sim2sim", "../papers/OmniContact/code-sim2sim"):
        if value:
            root = Path(value).resolve()
            if (root / "g1_description/g1_29dof.xml").is_file() and list((root / "data/carrybox").glob("*.npz")):
                return root
    pytest.skip("Requires the official carrybox sim2sim assets")


@pytest.mark.parametrize("example,half,yaw", [
    (0, (.15, .15, .15), 0.),
    (2, (.12, .17, .20), .4),
    (4, (.18, .14, .12), -.6),
])
def test_recovery_reproduces_entire_official_clip(sim_root, example, half, yaw):
    paths = sorted((sim_root / "data/carrybox").glob("*.npz"))
    frames = load_demo(paths[example % len(paths)]).frames
    box = frames[0, 104:111].copy()
    box[3:] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
    inputs = (sim_root, frames[0, 97:104], box, np.asarray(half), frames[-1, 104:107])
    official_ik = type(_generator(sim_root))._solve_fixed_feet_hand_ik
    official_fk = official_ik.__globals__["KINEMATICS"]
    assert recovery_fk_v7._recovery_type(sim_root) is not None
    expected = generate_v7_clip(*inputs)
    actual = recovery_fk_v7.generate_recovery_clip(*inputs)
    for old, new in zip(expected, actual, strict=True):
        assert old.dtype == new.dtype and old.shape == new.shape
        assert old.tobytes() == new.tobytes()
    assert official_ik.__globals__["KINEMATICS"] is official_fk
    assert "forward" not in vars(official_fk)


def test_unsupported_source_uses_original_generator(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(recovery_fk_v7, "_recovery_type", lambda root: None)
    monkeypatch.setattr(recovery_fk_v7, "generate_v7_clip", lambda *args: sentinel)
    assert recovery_fk_v7.generate_recovery_clip(tmp_path, None, None, None, None) is sentinel
