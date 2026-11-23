"""Physical G1 carry checks for side selection, interpolation and reset admission."""
import os
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from carrywm.data import load_demo
from carrywm.geometry_v7 import (CONTACT_GAP, LEG_CLEARANCE, MAX_PENETRATION, _faces,
                                 _opposed_side_error, _probe, generate_v7_clip, valid_start_mask)
from carrywm.sim import _matrix


@pytest.fixture(scope="module")
def scene():
    locations = (os.environ.get("SIM2SIM_ROOT"), os.environ.get("CARRY_SIM_ROOT"),
                 os.environ.get("OMNICONTACT_SIM2SIM"), "../OmniContact_sim2sim",
                 "../papers/OmniContact/code-sim2sim")
    for location in filter(None, locations):
        root = Path(location).resolve()
        path = root / "data/carrybox/carrybox_motion_5_with_contact.npz"
        if path.is_file() and (root / "g1_description/g1_29dof.xml").is_file():
            return root, load_demo(path)
    pytest.skip("Set SIM2SIM_ROOT to the official G1 model and carrybox example motions")


def box_rotation(yaw, roll):
    cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([cz * cr, cz * sr, sz * sr, sz * cr])


@pytest.mark.parametrize("yaw,roll", [(0., 0.), (.6, 0.), (1.2, np.pi / 2)])
def test_accept_opposed_lateral_faces_only(yaw, roll):
    half = np.array([.13, .18, .11])
    box = np.r_[.7, .3, .6, box_rotation(yaw, roll)]
    rot = _matrix(box[3:])
    face, _, heading = _faces(box, half, np.zeros(3))
    left = face["local_left"]
    gripped_axis = np.argmax(abs(left))
    vertical_axis = np.argmax(abs(rot[2]))
    assert gripped_axis != vertical_axis
    assert abs(left[gripped_axis]) == pytest.approx(half[gripped_axis])
    assert (rot @ left) @ np.array([-np.sin(heading), np.cos(heading), 0]) > 0
    top = np.eye(3)[vertical_axis] * half[vertical_axis]
    pairs = np.array([[left, -left], [left, left], [top, -top], [1.8 * left, -left]])
    errors = _opposed_side_error(pairs, half, np.broadcast_to(rot, (4, 3, 3)))
    assert errors[0] == pytest.approx(0.)
    assert (errors[1:] > .035).all()


@pytest.mark.parametrize("yaw,roll", [(.6, 0.), (1.2, np.pi / 2)])
def test_intermediate_grasp_and_lowering_on_rotated_boxes(scene, yaw, roll):
    root, demo = scene
    initial = demo.frames[0]
    box = initial[104:111].copy()
    box[3:] = box_rotation(yaw, roll)
    frames, phase = generate_v7_clip(root, initial[97:104], box, demo.half, demo.frames[-1, 104:107])
    valid = valid_start_mask(root, frames, demo.half, "cfgen", phase)
    probe = _probe(root, tuple(demo.half))
    face = _faces(box, demo.half, initial[97:100])[0]
    target = np.array([face["local_left"], -face["local_left"]])
    for phase_id in (21, 24):
        indices = np.flatnonzero(phase.ravel() == phase_id)
        middle = indices[len(indices) // 4:3 * len(indices) // 4]
        assert len(middle) > 8
        assert valid[middle].mean() >= .8
        # Test between the IK knots (spaced eight frames apart) as well as at them.
        for t in middle[::5]:
            f = frames[t]
            assert (f[37:39] > .5).all()
            probe.data.qpos[probe.box_qpos:probe.box_qpos + 7] = f[104:111]
            probe.pose(f[97:100], f[100:104], f[39:68])
            local = (probe.pad_positions() - f[104:107]) @ _matrix(f[107:111])
            assert np.linalg.norm(local - target, axis=1).max() < .012
            left, right, legs = probe.distances()
            assert -MAX_PENETRATION <= min(left, right) <= max(left, right) <= CONTACT_GAP
            assert legs >= LEG_CLEARANCE


def test_reject_body_intersection_and_uncontacted_air_grasp(scene):
    root, demo = scene
    probe = _probe(root, tuple(demo.half))
    f = demo.frames[0]
    probe.pose(f[97:100], f[100:104], f[39:68])
    knee_id = probe.model.body("left_knee_link").id
    knee_geom = probe.leg_geoms[probe.model.geom_bodyid[probe.leg_geoms] == knee_id][0]
    points = [probe.data.geom_xpos[g].copy() for g in (knee_geom, probe.hand_geoms[0][0])]
    frames = np.repeat(f[None], 4, axis=0)
    frames[:, 37:39] = 0.
    frames[:, 107:111] = [1., 0., 0., 0.]
    frames[:2, 104:107] = points
    frames[2:, 104:107] = [4., 4., 1.]
    frames[3, 37:39] = 1.
    for row, collider in zip(frames, (knee_geom, probe.hand_geoms[0][0])):
        probe.data.qpos[probe.box_qpos:probe.box_qpos + 7] = row[104:111]
        probe.pose(row[97:100], row[100:104], row[39:68])
        assert mj.mj_geomDistance(probe.model, probe.data, int(collider), probe.box_geom, .1, None) < -.012
    np.testing.assert_array_equal(valid_start_mask(root, frames, demo.half, "npz"),
                                  [False, False, True, False])


def test_npz_filter_reads_original_demo_without_modifying_it(scene):
    root, demo = scene
    frozen = demo.frames.view()
    frozen.flags.writeable = False
    before = frozen.tobytes()
    mask = valid_start_mask(root, frozen, demo.half, "npz")
    assert mask.shape == (len(frozen),) and mask.dtype == np.bool_
    assert mask.any() and not mask.all()
    assert frozen.tobytes() == before
