"""V7 carry references and kinematic reset checks, in MuJoCo order and metres."""
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

import mujoco as mj
import numpy as np

from .data import generate_clip
from .sim import _generator, _matrix, _scene


# Reset tolerances apply to the geometric state before a physics step. Contact
# force, sustained support and task success are measured separately in simulation.
FACE_TOL = .035
CONTACT_GAP = .005
MAX_PENETRATION = .012
SUPPORT_DRIFT = .03
PAD_DRIFT = .045
LEG_CLEARANCE = .002


@lru_cache(maxsize=4)
def _pad_offsets(root):
    tree = ElementTree.parse(Path(root) / "g1_description" / "g1_29dof.xml")
    wrist_offsets, palm_offsets = [], []
    for side, sign in (("left", -1.), ("right", 1.)):
        wrist = tree.find(f".//body[@name='{side}_wrist_yaw_link']")
        normal = np.array([0., sign, 0.])
        points = []
        for geom in wrist.findall("geom[@type='box']"):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            rot = _matrix(np.fromstring(geom.get("quat", "1 0 0 0"), sep=" "))
            half = np.fromstring(geom.get("size"), sep=" ")
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            # Support point toward the box. A tie on a flat edge/face chooses
            # its middle, keeping the representative point away from the rim.
            local_normal = rot.T @ normal
            extreme = np.where(np.abs(local_normal) > 1e-8, np.sign(local_normal), 0.)
            points.append(pos + rot @ (half * extreme))
        if not points:
            raise ValueError(f"No contact boxes on {side} G1 wrist")
        pad = points[np.argmax(np.asarray(points) @ normal)]
        palm = wrist.find(f"body[@name='{side}_palm_link']")
        origin = np.fromstring(palm.get("pos", "0 0 0"), sep=" ")
        palm_rot = _matrix(np.fromstring(palm.get("quat", "1 0 0 0"), sep=" "))
        wrist_offsets.append(pad)
        palm_offsets.append(palm_rot.T @ (pad - origin))
    return np.asarray(wrist_offsets), np.asarray(palm_offsets)


def palm_surface_offsets(root):
    """Left/right inner collision-surface points expressed in each palm link."""
    return _pad_offsets(root)[1].copy()


def _faces(box, half, pelvis):
    """Select opposed sides in the oriented box; any local axis can point up."""
    rot = _matrix(box[3:7])
    vertical = int(np.argmax(np.abs(rot[2])))
    sides = [axis for axis in range(3) if axis != vertical]
    delta = np.array([*(box[:2] - pelvis[:2]), 0.])
    approach_axis = max(sides, key=lambda a: abs(rot[:, a] @ delta))
    forward = rot[:, approach_axis] * (1. if rot[:, approach_axis] @ delta >= 0 else -1.)
    yaw = float(np.arctan2(forward[1], forward[0]))
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    grip_axis = next(a for a in sides if a != approach_axis)
    local_left = np.zeros(3)
    local_left[grip_axis] = half[grip_axis] * (1. if left @ rot[:, grip_axis] >= 0 else -1.)
    face = dict(pos=box[:3] - forward * half[approach_axis], n=-forward,
                w=half[grip_axis], local_left=local_left)
    return face, np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)], dtype=np.float32), yaw


class _Probe:
    """One private CPU geometry workspace per scene size. Never steps physics."""
    def __init__(self, root, half):
        self.model = model = _scene(root, half, headless=True)
        self.data = mj.MjData(model)
        self.box_geom = model.geom("box_geom").id
        self.box_qpos = int(model.joint("box").qposadr[0])
        self.wrists = [model.body(side+"_wrist_yaw_link").id for side in ("left", "right")]
        self.ankles = [model.body(side+"_ankle_pitch_link").id for side in ("left", "right")]
        self.pads = _pad_offsets(root)[0]
        self.hand_geoms = [self.collision_geoms([body]) for body in self.wrists]
        legs = [model.body(f"{side}_{name}_link").id for side in ("left", "right")
                for name in ("hip_yaw", "knee")]
        self.leg_geoms = self.collision_geoms(legs)
        self.active = np.array([2, *(6+i for i in range(29) if i not in (12, 13))])
        self.limits = model.jnt_range[model.actuator_trnid[:, 0]]

    def collision_geoms(self, bodies):
        m = self.model
        can_collide = ((m.geom_contype & m.geom_conaffinity[self.box_geom]) |
                       (m.geom_conaffinity & m.geom_contype[self.box_geom])) != 0
        ids = np.flatnonzero(np.isin(m.geom_bodyid, bodies) & can_collide)
        if not len(ids):
            raise ValueError(f"No box-colliding geometry on G1 bodies {bodies}")
        return ids

    def pose(self, pelvis, quat, joints):
        d = self.data
        d.qpos[:3], d.qpos[3:7], d.qpos[7:36] = pelvis, quat, joints
        mj.mj_kinematics(self.model, d)

    def pad_positions(self):
        d = self.data
        return d.xpos[self.wrists] + (d.xmat[self.wrists].reshape(2,3,3) @ self.pads[...,None])[...,0]

    def distances(self):
        """Signed distances for both hand colliders and the thigh/shin colliders."""
        m, d = self.model, self.data
        # mj_geomDistance uses the actual rotated box and convex MJCF meshes.
        # A cutoff larger than the reset margin avoids narrowphase for far pairs.
        return [min(mj.mj_geomDistance(m, d, int(g), self.box_geom, .1, None) for g in geoms)
                for geoms in (*self.hand_geoms, self.leg_geoms)]

    def correct(self, pelvis, quat, joints, targets, fixed_feet, foot_rotations):
        """Small damped IK correction at a lift/lower keyframe; keep root xy/yaw."""
        q, p = joints.astype(float).copy(), pelvis.astype(float).copy()
        m, d = self.model, self.data
        for _ in range(12):
            self.pose(p, quat, q)
            points = np.vstack((self.pad_positions(), d.xpos[self.ankles]))
            errors = np.vstack((targets, fixed_feet)) - points
            if np.linalg.norm(errors, axis=1).max() < .002:
                break
            mj.mj_comPos(m, d)
            rows, residual = [], []
            for i, body in enumerate((*self.wrists, *self.ankles)):
                jp, jr = np.empty((3,m.nv)), np.empty((3,m.nv))
                mj.mj_jac(m, d, jp, jr, points[i], body)
                rows.append(jp[:,self.active])
                residual.append(errors[i])
                if i >= 2:
                    # Express ankle orientation error in world angular coordinates.
                    actual = d.xmat[body].reshape(3,3)
                    dr = .5 * np.cross(actual.T, foot_rotations[i-2].T).sum(0)
                    rows.append(.12 * jr[:,self.active])
                    residual.append(.12 * dr)
                else:
                    # The inner palm surface should face the opposing hand.
                    desired = targets[1-i] - targets[i]
                    desired /= max(np.linalg.norm(desired), 1e-8)
                    normal = d.xmat[body].reshape(3,3)[:,1] * (-1. if i == 0 else 1.)
                    rows.append(.10 * (np.eye(3) - np.outer(normal, normal)) @ jr[:,self.active])
                    residual.append(.10 * np.cross(normal, desired))
            jac = np.vstack(rows)
            inc = np.linalg.solve(jac.T @ jac + .001 * np.eye(len(self.active)),
                                  jac.T @ np.concatenate(residual))
            inc *= min(1., .15 / max(1e-8, np.abs(inc).max()))
            p[2] = np.clip(p[2] + inc[0], pelvis[2]-.08, pelvis[2]+.08)
            active_joint = self.active[1:] - 6
            q[active_joint] = np.clip(q[active_joint]+inc[1:],
                                      self.limits[active_joint,0], self.limits[active_joint,1])
        return q, p


@lru_cache(maxsize=2)
def _probe(root, half):
    return _Probe(root, half)


class _CarryGeometry:
    """V7 overrides the official generator's grasp placement and interpolation."""
    def _get_target_face_info(self, obj_pos, obj_quat, pelvis_pos, dims):
        return _faces(np.r_[obj_pos, obj_quat], dims, pelvis_pos)

    def _grasp_targets(self, obj_pos, face, target_yaw_quat):
        self.local_left = face["local_left"]
        delta = self.initial_box_rot @ self.local_left
        return dict(lw_contact=obj_pos + delta, rw_contact=obj_pos - delta)

    def _append_joint_fk_phase(self, b, phase, **args):
        if phase in (21, 24):
            # Upstream creates these sections by independently interpolating
            # box, pelvis and q. Add FK-checked IK knots inside the interval.
            positions = args["pelvis_seq"].copy()
            joints = args["qseq"].copy()
            box_pos = args["obj_pos"]
            axis = _matrix(args["obj_quat"]) @ self.local_left
            feet = np.stack([b.last("la_p"), b.last("ra_p")])
            feet_rot = _matrix(np.stack([b.last("la_q"), b.last("ra_q")]))
            knots = np.unique(np.r_[np.arange(0, len(joints), 8), len(joints)-1])
            corrections = np.zeros((len(knots), 30))
            # Knot zero is the exact preceding phase boundary.
            for k, t in enumerate(knots[1:], start=1):
                targets = np.stack((box_pos[t]+axis, box_pos[t]-axis))
                q, p = self.probe.correct(positions[t], args["pelvis_quat_seq"][t],
                                          joints[t], targets, feet, feet_rot)
                corrections[k] = np.r_[q - joints[t], p[2] - positions[t,2]]
            times = np.arange(len(joints))
            delta = np.column_stack([np.interp(times, knots, col) for col in corrections.T])
            joints += delta[:,:29]
            positions[:,2] += delta[:,29]
            args["qseq"], args["pelvis_seq"] = joints, positions
        return super()._append_joint_fk_phase(b, phase, **args)


@lru_cache(maxsize=2)
def _v7_type(root):
    return type("V7CarryBox", (_CarryGeometry, type(_generator(root))), {})


def generate_v7_clip(root, pose, box, half, goal):
    root = Path(root).resolve()
    gen = _v7_type(root)()
    gen.cfg["ik_joint_indices"] = np.array([j for j in range(29) if j not in (12,13)])
    gen.probe = _probe(root, tuple(float(x) for x in half))
    gen._left_palm_center, gen._right_palm_center = gen.probe.pads.astype(np.float32)
    gen.initial_box_rot = _matrix(box[3:7])
    return generate_clip(gen, pose, box, half, goal)


def _opposed_side_error(local, half, rot):
    """Distance to a pair of opposite lateral face rectangles, in metres."""
    count = len(local)
    vertical = np.argmax(np.abs(rot[:,2]), axis=1)
    best = np.full(count, np.inf)
    for axis in range(3):
        for sign in (1., -1.):
            delta = np.maximum(np.abs(local) - half, 0.)
            delta[:,:,axis] = local[:,:,axis] - half[axis]*np.array([sign,-sign])
            error = np.linalg.norm(delta, axis=2).max(axis=1)
            best = np.minimum(best, np.where(vertical == axis, np.inf, error))
    return best


def valid_start_mask(root, frames, half, source, phase=None):
    """Return geometry-admissible physical starts. No cross-episode history is used."""
    root = Path(root).resolve()
    half = np.asarray(half, dtype=float)
    probe = _probe(root, tuple(half))
    d = probe.data
    rot = _matrix(frames[:,107:111])
    pads = np.empty((len(frames), 2, 3))
    feet = np.empty_like(pads)
    distances = np.empty((len(frames), 3))
    for i, f in enumerate(frames):
        d.qpos[probe.box_qpos:probe.box_qpos+7] = f[104:111]
        probe.pose(f[97:100], f[100:104], f[39:68])
        pads[i] = probe.pad_positions()
        feet[i] = d.xpos[probe.ankles]
        distances[i] = probe.distances()
    local = (pads - frames[:,None,104:107]) @ rot
    hand, legs = distances[:,:2], distances[:,2]
    bilateral = (frames[:,37:39] > .5).all(axis=1)
    admissible = (legs >= LEG_CLEARANCE) & (hand.min(axis=1) >= -MAX_PENETRATION)
    opposed = _opposed_side_error(local, half, rot) <= FACE_TOL
    admissible &= ~bilateral | (opposed & (hand.max(axis=1) <= CONTACT_GAP))
    if phase is not None and source in ("cfgen", "recovery"):
        face = _faces(frames[0,104:111], half, frames[0,97:100])[0]
        target = np.stack((face["local_left"], -face["local_left"]))
        drift = np.linalg.norm(local-target, axis=2).max(axis=1)
        admissible &= ~bilateral | (drift <= PAD_DRIFT)
        # These two intervals require planted feet; carry-walking permits steps.
        phase = np.asarray(phase).ravel()
        for stage in (21, 24):
            idx = np.flatnonzero(phase == stage)
            if len(idx):
                shift = np.linalg.norm(feet[idx] - feet[idx[0]], axis=2).max(axis=1)
                admissible[idx] &= shift <= SUPPORT_DRIFT
    return admissible
