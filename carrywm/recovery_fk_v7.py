"""Position-only MuJoCo FK for V7's online carry recovery planner."""
from functools import lru_cache
from pathlib import Path
from types import FunctionType

import mujoco as mj
import numpy as np

from .data import generate_clip
from .geometry_v7 import _CarryGeometry, _matrix, _probe, generate_v7_clip
from .sim import _generator


class _PoseKinematics:
    """The official IK reads wrist, palm, torso and ankle poses. It never reads forces."""

    def __init__(self, original):
        self.original = original
        self.model = original.model
        self.joint_names = original.joint_names
        self.data = mj.MjData(self.model)
        self.joint_qpos = np.asarray(original.joint_qpos_indices)
        self.bodies = ("left_wrist_yaw_link", "right_wrist_yaw_link",
                       "left_palm_link", "right_palm_link", "torso_link",
                       "left_ankle_pitch_link", "right_ankle_pitch_link")
        self.ids = [self.model.body(name).id for name in self.bodies]

    def forward(self, joint_pos, base_pos=None, base_quat=None, *args, **kwargs):
        if base_pos is None or base_quat is None or args or kwargs:
            return self.original.forward(joint_pos, base_pos, base_quat, *args, **kwargs)
        self.data.qpos[:3] = base_pos
        self.data.qpos[3:7] = base_quat
        self.data.qpos[self.joint_qpos] = joint_pos
        mj.mj_kinematics(self.model, self.data)
        # Advanced indexing takes a snapshot. The next FK call can safely reuse MjData.
        return {name: {"pos": p, "quat": q}
                for name, p, q in zip(self.bodies, self.data.xpos[self.ids], self.data.xquat[self.ids])}


def _with_globals(fn, **overrides):
    """Bind the upstream function to private dependencies; leave its module untouched."""
    bound = FunctionType(fn.__code__, {**fn.__globals__, **overrides},
                         fn.__name__, fn.__defaults__, fn.__closure__)
    bound.__kwdefaults__ = fn.__kwdefaults__
    return bound


@lru_cache(maxsize=2)
def _recovery_type(root):
    official = type(_generator(root))
    # The upstream solver and reference builder use a module-global KINEMATICS
    # instead of accepting FK as an argument. Specialize just these methods.
    try:
        solver = official._solve_fixed_feet_hand_ik
        append = official._append_joint_fk_phase
        new_builder = official._new_builder
        builder = new_builder.__globals__["_TrajBuilder"]
        k = solver.__globals__["KINEMATICS"]
        if (not k.has_free_joint or k.model.nq != 36
                or append.__globals__.get("KINEMATICS") is not k
                or builder._fk_refs.__globals__.get("KINEMATICS") is not k):
            return None
        poses = _PoseKinematics(k)

        class RecoveryBuilder(builder):
            _fk_refs = staticmethod(_with_globals(builder._fk_refs, KINEMATICS=poses))
            _infer_base_from_torso = staticmethod(_with_globals(builder._infer_base_from_torso, KINEMATICS=poses))

        class RecoverySource(official):
            _solve_fixed_feet_hand_ik = _with_globals(solver, KINEMATICS=poses)
            _append_joint_fk_phase = _with_globals(append, KINEMATICS=poses)
            _new_builder = _with_globals(new_builder, _TrajBuilder=RecoveryBuilder)
    except (KeyError, AttributeError):
        return None
    return type("V7RecoveryCarryBox", (_CarryGeometry, RecoverySource), {})


def generate_recovery_clip(root, pose, box, half, goal):
    root = Path(root).resolve()
    source = _recovery_type(root)
    if source is not None:
        gen = source()
        gen.cfg["ik_joint_indices"] = np.array([j for j in range(29) if j not in (12, 13)])
        gen.probe = _probe(root, tuple(float(x) for x in half))
        gen._left_palm_center, gen._right_palm_center = gen.probe.pads.astype(np.float32)
        gen.initial_box_rot = _matrix(box[3:7])
        try:
            return generate_clip(gen, pose, box, half, goal)
        except KeyError:
            # A different upstream generator may request extra bodies or velocities.
            pass
    return generate_v7_clip(root, pose, box, half, goal)
