"""Physical G1 carry checks for side selection, interpolation and reset admission."""
import os
from pathlib import Path
from unittest.mock import patch

import mujoco as mj
import numpy as np
import pytest

from carrywm.data import load_demo
from carrywm.geometry_v7 import (CONTACT_GAP, LEG_CLEARANCE, MAX_PENETRATION, _faces,
                                 _opposed_side_error, _probe, generate_v7_clip, valid_start_mask)
from carrywm.sim import _matrix, _valid_recovery_grasp


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
    first_grasp = np.flatnonzero((frames[:, 37:39] > .5).all(axis=1))[0]
    assert _valid_recovery_grasp(root, frames, demo.half, phase) == valid[first_grasp]
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


def test_recovery_check_preserves_face_and_foot_anchors(scene):
    root, demo = scene
    init = demo.frames[0]
    frames, phases = generate_v7_clip(root, init[97:104], init[104:111], demo.half,
                                      demo.frames[-1, 104:107])
    mask = valid_start_mask(root, frames, demo.half, "cfgen", phases)
    probe = _probe(root, tuple(demo.half))
    for planted_phase in (21, 24):
        grasp = np.flatnonzero(mask & (phases.ravel() == planted_phase)
                               & (frames[:, 37:39] > .5).all(axis=1))[0]
        # The initial pose defines the target faces. Frame 1 defines the planted
        # feet. Frame 48 is the first grasp, after a long noncontact lead-in.
        base = np.repeat(frames[grasp:grasp + 1], 96, axis=0)
        base[0] = frames[0]
        base[:48, 37:39] = 0.
        phase = np.full((len(base), 1), planted_phase)
        phase[0] = 11
        slipped = base.copy()
        slipped[1, 97] += .1
        wrong_side = base.copy()
        wrong_side[0, 97:99] = 2 * base[0, 104:106] - base[0, 97:99]
        for clip, expected in ((base, True), (slipped, False), (wrong_side, False)):
            assert bool(valid_start_mask(root, clip, demo.half, "cfgen", phase)[48]) == expected
            with patch.object(probe, "distances", wraps=probe.distances) as inspections:
                assert _valid_recovery_grasp(root, clip, demo.half, phase) == expected
            assert inspections.call_count <= 3
    base[:, 37:39] = 0.
    assert not _valid_recovery_grasp(root, base, demo.half, phase)


@pytest.mark.parametrize("root_z,waist", [(.805, .10), (.75, -.08)])
def test_phase24_ik_projects_only_invalid_seed_components(scene, monkeypatch, root_z, waist):
    import scipy.optimize
    from carrywm.geometry_v7 import _v7_type
    from carrywm.sim import _generator

    root, demo = scene
    base = _generator(root)
    v7 = _v7_type(root)()
    frame = demo.frames[0]
    pelvis = frame[97:100].copy()
    pelvis[2] = root_z
    joints = frame[39:68].copy()
    joints[14] = waist
    inputs = dict(pelvis_pos=pelvis, pelvis_quat=frame[100:104], seed_joint_pos=joints,
                  ik_joint_indices=np.arange(29), waist_pitch_min_rad=.06,
                  left_ankle_pos=frame[21:24], left_ankle_quat=frame[24:28],
                  right_ankle_pos=frame[28:31], right_ankle_quat=frame[31:35],
                  left_palm_pos=frame[:3], right_palm_pos=frame[7:10])
    measured = []
    solve = scipy.optimize.least_squares

    def measure(fun, x0, *, bounds, **kwargs):
        measured.append((x0.copy(), *[b.copy() for b in bounds]))
        return solve(fun, x0, bounds=bounds, **kwargs)

    monkeypatch.setattr(scipy.optimize, "least_squares", measure)
    with pytest.raises(ValueError, match="(?i)(initial guess|x0)"):
        base._solve_fixed_feet_hand_ik(**inputs)
    result = v7._solve_fixed_feet_hand_ik(**inputs)
    original, lower, upper = measured[0]
    corrected, actual_lower, actual_upper = measured[1]
    np.testing.assert_array_equal(actual_lower, lower)
    np.testing.assert_array_equal(actual_upper, upper)
    np.testing.assert_array_equal(corrected, np.clip(original, lower, upper))
    assert np.isfinite(np.concatenate(result)).all()
    assert pelvis[2] == np.float32(root_z) and joints[14] == np.float32(waist)
