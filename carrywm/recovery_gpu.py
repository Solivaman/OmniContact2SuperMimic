"""Batch V7 recovery reference poses after its original CPU planner has solved the IK.

Workers emit joint, pelvis, box and contact trajectories plus exact link poses at
phase boundaries. The owner of the CUDA simulator completes the intervening link
poses together, before any frame is passed to the collector or registered in replay.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from .data import LINKS, generate_clip
from .geometry_v7 import _CarryGeometry, _matrix, _probe
from .model import quat_mul
from .recovery_fk_v7 import _recovery_type, _with_globals
from .sim import _v7_recovery_plan, _valid_recovery_grasp


@dataclass
class RecoverySkeleton:
    frames: np.ndarray
    phase: np.ndarray
    gpu_frames: torch.Tensor | None = None


def _rotate_one(q, v):
    """Original quat_apply arithmetic for a single 3-vector, without np.cross dispatch."""
    u, w = q[1:], q[0]
    cross = np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]])
    return (v * (2. * w**2 - 1.) + cross * w * 2.
            + u * (u[None, :] @ v[:, None]) * 2.).reshape(3)


def _joint_block(self, builder, phase, *, obj_pos, obj_quat, qseq, pelvis_seq, pelvis_quat_seq, contact):
    builder.append_skeleton(phase, obj_pos, obj_quat, qseq, pelvis_seq, pelvis_quat_seq, contact)


def _loco_inputs(pos, quat, joints):
    # In the released carry generator, locomotion reads only the base rows of
    # the returned blocks. All link poses are reconstructed by the builder.
    return dict(base_p=pos, base_q=quat, joint_pos_mj=joints)


def _loco_block(builder, phase, *, fk_refs, object_pos, object_quat, contact):
    builder.append_skeleton(phase, object_pos, object_quat, fk_refs["joint_pos_mj"],
                            fk_refs["base_p"], fk_refs["base_q"], contact)


@lru_cache(maxsize=2)
def _skeleton_type(root):
    recovery = _recovery_type(root)
    if recovery is None:
        return None
    source = recovery.__bases__[1]
    generate = source.generate
    builder = source._new_builder.__globals__["_TrajBuilder"]

    class SkeletonBuilder(builder):
        @staticmethod
        def _fk_refs(base_p, base_q, joints):
            # The following IK phase reads last(), and the final pose writer
            # uses frame zero to choose the same quaternion hemisphere.
            idx = [0, -1] if len(joints) > 1 else [0]
            edges = builder._fk_refs(base_p[idx], base_q[idx], joints[idx])
            refs = {}
            for key, values in edges.items():
                refs[key] = np.repeat(values[-1:], len(joints), axis=0)
                refs[key][0] = values[0]
            return refs

        def append_skeleton(self, phase, obj_p, obj_q, joints, base_p, base_q, contact):
            # All G1 carry blocks supply base_p/base_q/joints. The upstream append
            # consequently derives its link fields exclusively through _fk_refs.
            n = len(joints) if np.ndim(joints) == 2 else len(base_p)
            p = self._bcast(base_p, n, (3,))
            q = self._bcast(base_q, n, (4,))
            self.append(phase, lw_p=p, lw_q=q, rw_p=p, rw_q=q, torso_p=p, torso_yaw_q=q,
                        torso_pitch_deg=np.zeros(n, np.float32), la_p=p, la_q=q, ra_p=p, ra_q=q,
                        base_p=p, base_q=q, dof_pos=joints, obj_p=obj_p, obj_q=obj_q, contact=contact)

    # Bind the installed version's original locomotion equations, interpolation,
    # object transport, and timing. Only override generation of redundant FK rows.
    def specialize(fn):
        deps = {}
        for key in fn.__code__.co_names:
            obj = fn.__globals__.get(key)
            if key in ("_fk_reference_sequence", "_fk_reference_sequence_from_joints"):
                deps[key] = _loco_inputs
            elif key == "_append_fk_block":
                deps[key] = _loco_block
            elif key.startswith("_append_") and getattr(obj, "__module__", "").endswith("CFgen_meta1_loco"):
                deps[key] = specialize(obj)
        return _with_globals(fn, **deps)

    class SkeletonSource(source):
        _new_builder = _with_globals(source._new_builder, _TrajBuilder=SkeletonBuilder)
        _solve_fixed_feet_hand_ik = _with_globals(source._solve_fixed_feet_hand_ik, quat_apply=_rotate_one)
        _append_joint_fk_phase = _joint_block

    SkeletonSource.generate = specialize(generate)
    return type("V7RecoverySkeletonSource", (_CarryGeometry, SkeletonSource), {})


def prepare_recovery(root, pose, box, half, goal):
    """CPU worker. Admission uses the original q, base, box and MuJoCo collision geometry."""
    root = Path(root).resolve()
    try:
        source = _skeleton_type(root)
    except (KeyError, AttributeError):
        source = None
    if source is None:
        return _v7_recovery_plan(root, pose, box, half, goal)
    gen = source()
    gen.cfg["ik_joint_indices"] = np.array([j for j in range(29) if j not in (12, 13)])
    gen.probe = _probe(root, tuple(float(x) for x in half))
    gen._left_palm_center, gen._right_palm_center = gen.probe.pads.astype(np.float32)
    gen.initial_box_rot = _matrix(box[3:7])
    try:
        frames, phase = generate_clip(gen, pose, box, half, goal)
        if _valid_recovery_grasp(root, frames, half, phase):
            return RecoverySkeleton(frames, phase)
    except KeyError:
        # Some generator versions request link fields beyond the G1 skeleton.
        return _v7_recovery_plan(root, pose, box, half, goal)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return None


def _rotate(q, v):
    t = 2 * torch.linalg.cross(q[..., 1:], v)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t)


class BatchedRecoveryFK:
    """G1 rigid-body tree in double precision; one GPU batch spans multiple physical worlds."""
    def __init__(self, root, device):
        self.device = torch.device(device)
        model = mujoco.MjModel.from_xml_path(str(Path(root) / "g1_description/g1_29dof.xml"))
        if (model.nq != 36 or model.body_jntnum[1] != 1 or np.any(model.body_jntnum > 1)
                or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError("Recovery FK requires the official free-root 29-hinge G1")
        self.links = torch.tensor([model.body(n).id for n in LINKS], device=self.device)
        active = np.maximum(model.body_jntadr, 0)
        hinge = np.where((model.body_jntnum == 1) & (active > 0), model.jnt_qposadr[active]-7, 29)
        self.hinge = torch.as_tensor(hinge, device=self.device)
        def tensor(values):
            return torch.as_tensor(np.array(values, copy=True), dtype=torch.float64, device=self.device)
        self.axis = tensor(model.jnt_axis[active])
        self.anchor = tensor(np.where((hinge < 29)[:, None], model.jnt_pos[active], 0.))
        self.offset = tensor(model.body_pos)
        self.rest = tensor(model.body_quat)
        self.q0 = tensor(model.qpos0[7:])
        parents = model.body_parentid.copy()
        self.parents = []
        while np.any(parents > 0):
            self.parents.append(torch.as_tensor(parents, device=self.device))
            parents = parents[parents]

    def _body_poses(self, joint_and_root, links):
        """Return unprojected world-frame link positions and unit quaternions."""
        joints, pos, root_q = joint_and_root.split([29, 3, 4], -1)
        angle = F.pad(joints-self.q0, (0, 1))[:, self.hinge, None] * .5
        step_q = torch.cat((angle.cos(), angle.sin() * self.axis), -1)
        q = quat_mul(self.rest.expand_as(step_q), step_q)
        anchor = self.anchor.expand_as(q[..., 1:])
        p = self.offset + _rotate(self.rest.expand_as(q), anchor - _rotate(step_q, anchor))
        p[:, 1], q[:, 1] = pos, F.normalize(root_q, dim=-1)
        for parent in self.parents:
            pq = q[:, parent]
            p, q = p[:, parent] + _rotate(pq, p), quat_mul(pq, q)
        return p[:, links], F.normalize(q[:, links], dim=-1)

    def _link_poses(self, joint_and_root):
        p, q = self._body_poses(joint_and_root, self.links)
        # Official references discard torso roll and reconstruct yaw * pitch.
        w, x, y, z = q[:, 2].unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))*.5
        pitch = torch.asin(torch.clamp(2*(w*y-z*x), -1., 1.))*.5
        cy, sy, cp, sp = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin()
        q[:, 2] = torch.stack((cy*cp, -sy*sp, cy*sp, sy*cp), dim=-1)
        return torch.cat((p, q), -1).float()

    @torch.no_grad()
    def complete(self, plans, *, keep_on_device=False):
        if not plans:
            return
        raw = np.concatenate([np.concatenate((plan.frames[:, 39:68], plan.frames[:, 97:104]), -1)
                              for plan in plans])
        # Bound temporary memory independently of trajectory lengths and drop count.
        pieces = [self._link_poses(torch.as_tensor(raw[i:i+4096], device=self.device,
                                                   dtype=torch.float64))
                  for i in range(0, len(raw), 4096)]
        poses_gpu = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        lengths = [len(plan.frames) for plan in plans]
        blocks = poses_gpu.split(lengths, dim=0)
        first_orig = torch.as_tensor(
            np.stack([plan.frames[0, :35].reshape(5, 7)[:, 3:] for plan in plans]),
            dtype=torch.float32,
            device=self.device,
        )
        if keep_on_device:
            tail_host = np.concatenate(
                [np.ascontiguousarray(plan.frames[:, 35:], dtype=np.float32) for plan in plans],
                axis=0,
            )
            tails_gpu = torch.as_tensor(tail_host, dtype=torch.float32, device=self.device).split(lengths, dim=0)
        fixed_blocks = []
        for idx, (plan, block) in enumerate(zip(plans, blocks)):
            block = block.clone()
            quat = block[:, :, 3:]
            s0 = torch.where((quat[:1] * first_orig[idx:idx+1]).sum(-1) < 0, -1., 1.)
            if len(quat) > 1:
                s_next = torch.where((quat[:-1] * quat[1:]).sum(-1) < 0, -1., 1.)
                signs = torch.cat((s0, s_next), dim=0)
            else:
                signs = s0
            quat.mul_(torch.cumprod(signs, dim=0).unsqueeze(-1))
            if keep_on_device:
                plan.gpu_frames = torch.cat((block.reshape(-1, 35), tails_gpu[idx]), dim=1)
            else:
                fixed_blocks.append(block.reshape(-1, 35))
                plan.gpu_frames = None
        if not keep_on_device:
            host_blocks = torch.cat(fixed_blocks, dim=0).cpu().numpy()
            start = 0
            for plan, n in zip(plans, lengths):
                plan.frames[:, :35] = host_blocks[start:start + n]
                start += n
