#!/usr/bin/env python3
"""Evaluate a V7-family actor and the official OmniContact policies in MuJoCo."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
import torch
import yaml

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from official_cfgen_v72_adapter import (
    generate_official_cfgen_raw,
    pack_official_cfgen_direct,
    reconstruct_clip_from_5links_ik,
    verify_official_fk_consistency,
)
from carrywm.data import LAB2MJ, generate_clip, raw_window
from carrywm.model import BOX_GOAL_DIM
from carrywm.recovery_fk_v7 import generate_recovery_clip
from carrywm.sim import DEFAULT_CFGEN_GOAL_NOISE_STD_M, _matrix, _scene
from carrywm.train import _CpuEvalNet, _init_eval_worker, tensor


DEFAULT_PELVIS_INIT_Z = 0.778
MIN_ROBOT_CLEARANCE = 0.65
FALL_MIN_HEIGHT_M = 0.15
FALL_UPRIGHT_MIN_R22 = 0.30
OFFICIAL_GOAL_TOL_M = 0.20


def _clean_json(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return None if isinstance(val, float) and not math.isfinite(val) else val
    if isinstance(obj, np.ndarray):
        return _clean_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    return obj


def _make_mj_env_consts(sim_root: Path):
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ

    cfg = yaml.safe_load((sim_root / "policy/omnicontact/config/OmniContact.yaml").read_text())
    return {
        "cfg": cfg,
        "default_qpos_mj": np.asarray(DEFAULT_JOINT_POS_MJ, dtype=np.float64).copy(),
        "default_angles_lab": np.asarray(cfg["default_angles_lab"], dtype=np.float32),
        "kp_mj": np.asarray(cfg["kp_lab"], dtype=np.float64)[LAB2MJ],
        "kd_mj": np.asarray(cfg["kd_lab"], dtype=np.float64)[LAB2MJ],
        "tau_limit_mj": np.asarray(cfg["tau_limit_lab"], dtype=np.float64)[LAB2MJ],
        "action_scale_lab": np.asarray(cfg["action_scale_lab"], dtype=np.float32),
        "mj2lab": np.asarray(cfg["mj2lab"], dtype=np.int32),
        "lab2mj": np.asarray(cfg["lab2mj"], dtype=np.int32),
    }


def sample_official_episodes(n: int, seed: int, *, randomize_box_size: bool) -> list[dict]:
    """Generate official Carry Box benchmark specs matching paper.tex Appendix sec:append_eval_protocol."""
    rng = np.random.default_rng(seed)
    episodes = []
    for idx in range(n):
        if randomize_box_size:
            full_dims = rng.uniform(0.20, 0.50, size=3).astype(np.float32)
            half = 0.5 * full_dims
        else:
            half = np.array([0.15, 0.15, 0.15], dtype=np.float32)

        foot_r = float(max(half[0], half[1]))
        robot_clear = foot_r + MIN_ROBOT_CLEARANCE
        pair_clear = 2.0 * foot_r + 0.60

        while True:
            ix, iy = rng.uniform(-5.0, 5.0, size=2)
            if math.hypot(ix, iy) >= robot_clear:
                break
        iz = float(rng.uniform(float(half[2]), 0.80))

        while True:
            gx, gy = rng.uniform(-5.0, 5.0, size=2)
            if math.hypot(gx, gy) >= robot_clear and math.hypot(gx - ix, gy - iy) >= pair_clear:
                break
        gz = float(rng.uniform(float(half[2]), 0.80))

        init_pos = np.array([ix, iy, iz], dtype=np.float32)
        goal_pos = np.array([gx, gy, gz], dtype=np.float32)
        episodes.append(
            {
                "ep_id": idx,
                "label": f"rand_{idx:03d}",
                "half": half.astype(float).tolist(),
                "init_pos": init_pos.astype(float).tolist(),
                "init_quat": [1.0, 0.0, 0.0, 0.0],
                "goal_pos": goal_pos.astype(float).tolist(),
                "root_pos": [0.0, 0.0, DEFAULT_PELVIS_INIT_Z],
                "root_quat": [1.0, 0.0, 0.0, 0.0],
                "robot_joint_pos_mj": None,
                "table1_pos": (init_pos - np.array([0.0, 0.0, half[2] + 0.01], dtype=np.float32)).astype(float).tolist(),
                "table2_pos": (goal_pos - np.array([0.0, 0.0, half[2] + 0.01], dtype=np.float32)).astype(float).tolist(),
            }
        )
    return episodes


def load_official_demo_episodes(sim_root: Path) -> list[dict]:
    """Load the 13 carrybox_*.npz scenes distributed inside OmniContact_sim2sim/data/carrybox."""
    data_dir = Path(sim_root) / "data" / "carrybox"
    files = sorted(data_dir.glob("carrybox_motion_*_with_contact.npz"))
    episodes = []
    for idx, path in enumerate(files):
        with np.load(path, allow_pickle=False) as d:
            body_pos = np.asarray(d["body_pos_w"], dtype=np.float32)
            body_quat = np.asarray(d["body_quat_w"], dtype=np.float32)
            base_pos = np.asarray(d["base_pos_w"][0], dtype=np.float32) if "base_pos_w" in d else body_pos[0, 0]
            base_quat = np.asarray(d["base_quat_w"][0], dtype=np.float32) if "base_quat_w" in d else body_quat[0, 0]
            joint_mj = np.asarray(d["joint_pos"][0, LAB2MJ], dtype=np.float32)
            obj0 = np.asarray(d["object_pos_w"][0], dtype=np.float32)
            obj_q0 = np.asarray(d["object_quat_w"][0], dtype=np.float32)
            goal = np.asarray(d["object_pos_w"][-1], dtype=np.float32)
            t1 = np.asarray(d["table1_pos_w"], dtype=np.float32).reshape(-1, 3)[0]
            t2 = np.asarray(d["table2_pos_w"], dtype=np.float32).reshape(-1, 3)[0]
        episodes.append(
            {
                "ep_id": idx,
                "label": path.stem,
                "half": [0.15, 0.15, 0.15],
                "init_pos": obj0.astype(float).tolist(),
                "init_quat": obj_q0.astype(float).tolist(),
                "goal_pos": goal.astype(float).tolist(),
                "root_pos": base_pos.astype(float).tolist(),
                "root_quat": base_quat.astype(float).tolist(),
                "robot_joint_pos_mj": joint_mj.astype(float).tolist(),
                "table1_pos": t1.astype(float).tolist(),
                "table2_pos": t2.astype(float).tolist(),
            }
        )
    return episodes


def load_case1_test_episodes(motion_dir: Path) -> list[dict]:
    """Load the held-out 58 case1_carry test scenes as an official MoCap fairness subset."""
    from carrywm.data import load_demo, split_paths

    paths = split_paths(Path(motion_dir), "test")
    episodes = []
    for idx, path in enumerate(paths):
        sc = load_demo(path)
        episodes.append(
            {
                "ep_id": idx,
                "label": path.stem,
                "half": np.asarray(sc.half, dtype=float).tolist(),
                "init_pos": np.asarray(sc.frames[0, 104:107], dtype=float).tolist(),
                "init_quat": np.asarray(sc.frames[0, 107:111], dtype=float).tolist(),
                "goal_pos": np.asarray(sc.frames[-1, 104:107], dtype=float).tolist(),
                "root_pos": np.asarray(sc.frames[0, 97:100], dtype=float).tolist(),
                "root_quat": np.asarray(sc.frames[0, 100:104], dtype=float).tolist(),
                "robot_joint_pos_mj": np.asarray(sc.frames[0, 39:68], dtype=float).tolist(),
                "table1_pos": np.asarray(sc.tables[0], dtype=float).tolist(),
                "table2_pos": np.asarray(sc.tables[1], dtype=float).tolist(),
            }
        )
    return episodes


class _OfficialOnnxAdapter:
    """Exact observation builder matching OmniContact_sim2sim/policy/omnicontact/OmniContact.py."""

    _HISTORY_SLICES = (
        (0, 15),
        (15, 18),
        (18, 21),
        (21, 50),
        (50, 79),
        (79, 108),
        (108, 111),
        (111, 117),
        (117, 141),
    )
    _FUTURE_OFFSETS = np.array([0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50], dtype=np.int32)
    _BBOX_CORNERS = np.array(
        [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ],
        dtype=np.float32,
    )

    def __init__(self, sim_root: Path, onnx_path: Path, consts: dict):
        import onnxruntime as ort
        from common.mujoco_kinematics import MujocoKinematics
        from common import utils as cu

        self.cu = cu
        self.consts = consts
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.num_obs = int(self.sess.get_inputs()[0].shape[-1])
        self.single_obs_dim = (self.num_obs - 11 * 49) // 5
        if self.single_obs_dim != 141 or self.num_obs != 1244:
            # Keep flexible for variants like 1343 vs 1244, while tracking single_obs_dim.
            pass
        self.kin = MujocoKinematics(str(sim_root / "g1_description" / "g1_29dof.xml"))
        self.hist = np.zeros((5, 141), dtype=np.float32)
        self.last_action_lab = np.zeros(29, dtype=np.float32)
        self.traj = None
        self.half = None

    def reset_episode(self, traj: dict, half: np.ndarray):
        self.traj = traj
        self.half = np.asarray(half, dtype=np.float32)
        self.hist.fill(0.0)
        self.last_action_lab.fill(0.0)

    def set_plan(self, traj: dict):
        self.traj = traj

    def ref_obj_at(self, step_idx: int) -> np.ndarray:
        arr = np.asarray(self.traj["ref_object_pos"], dtype=np.float32)
        return arr[min(max(step_idx - 1, 0), len(arr) - 1)]

    def ref_obj_frame(self, frame_idx: int) -> np.ndarray:
        arr = np.asarray(self.traj["ref_object_pos"], dtype=np.float32)
        return arr[min(max(frame_idx, 0), len(arr) - 1)]

    def ref_torso_frame(self, frame_idx: int) -> np.ndarray:
        arr = np.asarray(self.traj["ref_torso_future_pos"], dtype=np.float32)
        return arr[min(max(frame_idx, 0), len(arr) - 1)]

    def plan_length(self) -> int:
        return int(len(self.traj["ref_left_wrist_pos"]))

    def act(self, state84: np.ndarray, step_idx: int) -> np.ndarray:
        cu = self.cu
        q_mj = state84[7:36]
        dq_mj = state84[42:71]
        base_pos = state84[0:3]
        base_quat = state84[3:7]
        ang_vel = state84[39:42]
        obj_pos = state84[71:74]
        obj_quat = state84[74:78]

        fk = self.kin.forward(q_mj, base_pos, base_quat)
        torso_pos = fk["torso_link"]["pos"].astype(np.float32)
        torso_quat = fk["torso_link"]["quat"].astype(np.float32)
        ee_names = ("left_palm_link", "right_palm_link", "left_ankle_pitch_link", "right_ankle_pitch_link", "mid360_link")
        ee_pos = np.array([cu.quat_rotate_inverse(torso_quat, fk[n]["pos"] - torso_pos) for n in ee_names], dtype=np.float32).reshape(-1)
        grav = cu.get_gravity_orientation(base_quat).astype(np.float32)

        mj2lab = self.consts["mj2lab"]
        qj = (q_mj[mj2lab] - self.consts["default_angles_lab"]).astype(np.float32)
        dqj = dq_mj[mj2lab].astype(np.float32)

        heading = cu.yaw_quat(torso_quat).astype(np.float32)
        heading_conj = cu.quat_conjugate(heading).astype(np.float32)

        obj_pos_rel, obj_rot_rel = cu.subtract_frame_transforms(torso_pos, heading, obj_pos, obj_quat)
        obj_pos_rel = np.asarray(obj_pos_rel, dtype=np.float32)
        norm_o = float(np.linalg.norm(obj_pos_rel))
        if norm_o > 4.0:
            obj_pos_rel = obj_pos_rel * (4.0 / norm_o)
        obj_rot_6d = cu.matrix_from_quat(obj_rot_rel)[:, :2].reshape(-1).astype(np.float32)

        bbox_off = self._BBOX_CORNERS * self.half[None, :]
        bbox_world = cu.quat_apply_batch(obj_quat, bbox_off) + obj_pos[None, :]
        bbox_rel = cu.quat_apply_batch(heading_conj, bbox_world - torso_pos[None, :])
        bn = np.linalg.norm(bbox_rel, axis=-1, keepdims=True)
        bbox_rel = bbox_rel * np.minimum(1.0, 4.0 / np.maximum(bn, 1e-8))

        last_idx = self.plan_length() - 1
        idx = np.minimum(step_idx + self._FUTURE_OFFSETS, last_idx)

        def build_rel(p_key: str, q_key: str):
            p_sel = np.asarray(self.traj[p_key][idx], dtype=np.float32)
            q_sel = np.asarray(self.traj[q_key][idx], dtype=np.float32)
            rel_p = cu.quat_apply_batch(heading_conj, p_sel - torso_pos[None, :])
            rel_q = cu.quat_mul_left_batch(heading_conj, q_sel)
            return rel_p, cu.quat_to_6d_batch(rel_q)

        fl_p, fl_6 = build_rel("ref_left_wrist_pos", "ref_left_wrist_quat")
        fr_p, fr_6 = build_rel("ref_right_wrist_pos", "ref_right_wrist_quat")
        ft_p, ft_6 = build_rel("ref_torso_future_pos", "ref_torso_future_quat")
        fla_p, fla_6 = build_rel("ref_left_ankle_future_pos", "ref_left_ankle_future_quat")
        fra_p, fra_6 = build_rel("ref_right_ankle_future_pos", "ref_right_ankle_future_quat")
        f_con = np.asarray(self.traj["ref_contact"][idx], dtype=np.float32).reshape(-1)
        tracking = np.concatenate(
            [np.concatenate([fl_p, fl_6, fr_p, fr_6, ft_p, ft_6, fla_p, fla_6, fra_p, fra_6], axis=-1).reshape(-1), f_con],
            axis=-1,
        ).astype(np.float32)

        curr_prop = np.concatenate(
            [
                ee_pos,
                ang_vel.astype(np.float32).reshape(-1),
                grav,
                qj,
                dqj,
                self.last_action_lab,
                obj_pos_rel.astype(np.float32),
                obj_rot_6d,
                bbox_rel.reshape(-1).astype(np.float32),
            ]
        ).astype(np.float32)

        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = curr_prop
        hist_flat = np.concatenate([self.hist[:, a:b].reshape(-1) for a, b in self._HISTORY_SLICES])
        full_obs = np.concatenate([tracking, hist_flat]).astype(np.float32)
        if len(full_obs) != self.num_obs:
            # Handle trailing extra channels if an older ONNX expects >1244 dims.
            if len(full_obs) < self.num_obs:
                full_obs = np.pad(full_obs, (0, self.num_obs - len(full_obs)))
            else:
                full_obs = full_obs[: self.num_obs]

        feed = {self.input_names[0]: full_obs[None, ...]}
        if len(self.input_names) > 1:
            feed[self.input_names[1]] = np.array([[0.0]], dtype=np.float32)
        act_lab = self.sess.run(None, feed)[0].squeeze().astype(np.float32)
        self.last_action_lab = act_lab.copy()
        target_mj = (act_lab * self.consts["action_scale_lab"] + self.consts["default_angles_lab"])[self.consts["lab2mj"]]
        return target_mj.astype(np.float64)


class _V72ActorRunner:
    """Run a V7-family actor using its training-time observation and reference encoding."""

    def __init__(self, net: _CpuEvalNet):
        self.net = net
        self.geo = net.geo
        self.hist = None
        self.prev_action = np.zeros(29, dtype=np.float32)
        self.ctx = None
        self.clip = None

    def reset_episode(self, clip: np.ndarray, ctx: np.ndarray):
        self.clip = np.asarray(clip, dtype=np.float32)
        self.ctx = np.asarray(ctx, dtype=np.float32)
        self.hist = None
        self.prev_action = np.zeros(29, dtype=np.float32)

    def set_plan(self, clip: np.ndarray):
        self.clip = np.asarray(clip, dtype=np.float32)

    def ref_obj_at(self, step_idx: int) -> np.ndarray:
        return self.clip[min(max(step_idx - 1, 0), len(self.clip) - 1), 104:107]

    def ref_obj_frame(self, frame_idx: int) -> np.ndarray:
        return self.clip[min(max(frame_idx, 0), len(self.clip) - 1), 104:107]

    def ref_torso_frame(self, frame_idx: int) -> np.ndarray:
        return self.clip[min(max(frame_idx, 0), len(self.clip) - 1), 14:17]

    def plan_length(self) -> int:
        return int(len(self.clip))

    @torch.no_grad()
    def act(self, state84: np.ndarray, step_idx: int) -> np.ndarray:
        dev = self.net.device
        s_t = tensor(state84, dev)[None]
        c_t = tensor(self.ctx, dev)[None]
        obs = self.geo.observation(s_t, tensor(self.prev_action, dev)[None], c_t)
        if self.hist is None:
            self.hist = obs[:, None].expand(-1, 10, -1).clone()
        else:
            self.hist = torch.cat((self.hist[:, 1:], obs[:, None]), dim=1)
        win = raw_window(self.clip, min(step_idx, len(self.clip) - 1))
        goal = self.net.reference(s_t, tensor(win, dev)[None])
        action = np.clip(self.net.actor(goal, self.hist)[0].float().cpu().numpy(), -10.0, 10.0).astype(np.float32)
        self.prev_action = action.copy()
        return self.geo.default.cpu().numpy().astype(np.float64) + self.geo.scale.cpu().numpy().astype(np.float64) * action.astype(np.float64)


def _setup_mj_state(m, d, ep: dict, default_qpos_mj: np.ndarray):
    half = np.asarray(ep["half"], dtype=float)
    bp = int(m.joint("box").qposadr[0])
    bv = int(m.joint("box").dofadr[0])
    table_ids = [int(m.body(f"plane_{j}_holder").mocapid[0]) for j in (1, 2)]

    mj.mj_resetData(m, d)
    d.qpos[:3] = ep["root_pos"]
    d.qpos[3:7] = ep["root_quat"]
    d.qpos[7:36] = ep["robot_joint_pos_mj"] if ep["robot_joint_pos_mj"] is not None else default_qpos_mj
    d.qvel[:35] = 0.0
    d.qpos[bp : bp + 3] = ep["init_pos"]
    d.qpos[bp + 3 : bp + 7] = ep["init_quat"]
    d.qvel[bv : bv + 6] = 0.0
    d.mocap_pos[table_ids[0]] = ep["table1_pos"]
    d.mocap_pos[table_ids[1]] = ep["table2_pos"]
    mj.mj_forward(m, d)
    return bp, bv, half


def _box_hand_contacts(m, d, box_geom: int, wrist_body_ids: tuple[int, int]) -> bool:
    wrench = np.empty(6)
    f = np.zeros((2, 3))
    for ci, con in enumerate(d.contact):
        g1, g2 = int(con.geom1), int(con.geom2)
        if box_geom not in (g1, g2):
            continue
        other = g2 if g1 == box_geom else g1
        bid = int(m.geom_bodyid[other])
        if bid not in wrist_body_ids:
            continue
        mj.mj_contactForce(m, d, ci, wrench)
        if wrench[0] <= 0.0:
            continue
        wf = np.asarray(con.frame).reshape(3, 3).T @ wrench[:3]
        slot = 0 if bid == wrist_body_ids[0] else 1
        f[slot] += wf if other == g2 else -wf
    return bool((np.linalg.norm(f, axis=1) > 5.0).any())


def _is_fallen(state84: np.ndarray) -> bool:
    r22 = float(_matrix(state84[3:7])[2, 2])
    return bool(float(state84[2]) < FALL_MIN_HEIGHT_M or r22 < FALL_UPRIGHT_MIN_R22)


def _run_episode_with_runner(
    sim_root: Path,
    ep: dict,
    consts: dict,
    runner,
    planner_mode: str,
    init_plan,
    *,
    allow_replan: bool,
    max_replans: int = 2,
) -> dict:
    half = tuple(float(x) for x in ep["half"])
    m = _scene(sim_root, half, headless=True)
    d = mj.MjData(m)
    bp, bv, half_arr = _setup_mj_state(m, d, ep, consts["default_qpos_mj"])
    box_geom = m.geom("box_geom").id
    floor_geom = m.geom("floor").id
    tables_geom = [m.geom(f"plane_{j}").id for j in (1, 2)]
    wrist_ids = (m.body("left_wrist_yaw_link").id, m.body("right_wrist_yaw_link").id)
    torso_body_id = m.body("torso_link").id
    table_ids = [int(m.body(f"plane_{j}_holder").mocapid[0]) for j in (1, 2)]

    init_box = np.asarray(ep["init_pos"], dtype=np.float32)
    goal_box = np.asarray(ep["goal_pos"], dtype=np.float32)
    root_p = np.asarray(ep["root_pos"], dtype=np.float64)
    rq = np.asarray(ep["root_quat"], dtype=np.float64)
    init_dist_m = round(float(math.hypot(float(init_box[0]) - float(root_p[0]), float(init_box[1]) - float(root_p[1]))), 3)
    ryaw = math.atan2(2.0 * (rq[0] * rq[3] + rq[1] * rq[2]), 1.0 - 2.0 * (rq[2] * rq[2] + rq[3] * rq[3]))
    byaw = math.atan2(float(init_box[1]) - float(root_p[1]), float(init_box[0]) - float(root_p[0]))
    init_heading_deg = round(abs(math.degrees(math.atan2(math.sin(byaw - ryaw), math.cos(byaw - ryaw)))), 1)
    ctx = np.concatenate(
        (
            half_arr.astype(np.float32),
            np.asarray(m.body("box").mass, dtype=np.float32),
            np.asarray(m.body("box").inertia, dtype=np.float32),
            np.asarray(m.geom_friction[box_geom], dtype=np.float32),
            np.asarray(d.mocap_pos[table_ids].ravel(), dtype=np.float32),
        )
    ).astype(np.float32)

    if init_plan is None:
        return {
            "ep_id": ep["ep_id"],
            "label": ep["label"],
            "box_size_cm": [round(float(200.0 * x), 1) for x in half_arr],
            "max_box_cm": round(float(200.0 * max(half_arr)), 1),
            "travel_dist_m": round(float(np.linalg.norm(goal_box[:2] - init_box[:2])), 3),
            "init_dist_m": init_dist_m,
            "init_heading_deg": init_heading_deg,
            "init_z_m": round(float(init_box[2]), 3),
            "goal_z_m": round(float(goal_box[2]), 3),
            "plan_failed": True,
            "official_success": False,
            "strict_success": False,
            "dist_only_success": False,
            "final_goal_err_m": float(np.linalg.norm(init_box - goal_box)),
            "open_loop_goal_err_m": float(np.linalg.norm(init_box - goal_box)),
            "E_obj_m": None,
            "open_loop_E_obj_m": None,
            "E_torso_m": None,
            "open_loop_E_torso_m": None,
            "action_rate_mean_rad_s": None,
            "action_rate_rms_rad_s": None,
            "hand_contact_ratio": 0.0,
            "open_loop_hand_contact_ratio": 0.0,
            "box_stable": False,
            "lifted": False,
            "fallen": False,
            "dropped": False,
            "replans": 0,
            "steps": 0,
            "fail_reason": "plan_failed",
        }

    if isinstance(runner, _OfficialOnnxAdapter):
        runner.reset_episode(init_plan, half_arr)
    else:
        runner.reset_episode(init_plan, ctx)

    kp, kd, tau_lim = consts["kp_mj"], consts["kd_mj"], consts["tau_limit_mj"]
    lifted = False
    carried_steps = 0
    lost_steps = 0
    dropped = False
    fallen = False
    replans = 0
    cooldown = 0
    step_in_plan = 0
    total_steps = 0
    # Guard against runaway replan loops with an absolute cap.
    max_total_steps = max(2500, runner.plan_length() * (max_replans + 1) + 100)

    def read_state() -> np.ndarray:
        return np.concatenate((d.qpos[:36], d.qvel[:35], d.qpos[bp : bp + 7], d.qvel[bv : bv + 6])).astype(np.float32)

    open_loop_snapshot = None
    prev_target_q = None
    act_abs_sum = 0.0
    act_sq_sum = 0.0
    act_samples = 0
    e_obj_sum = 0.0
    e_torso_sum = 0.0
    hand_steps = 0

    while total_steps < max_total_steps:
        state84 = read_state()
        target_q_mj = runner.act(state84, step_in_plan)
        if prev_target_q is not None:
            delta_rate = (target_q_mj - prev_target_q) / 0.02
            act_abs_sum += float(np.mean(np.abs(delta_rate)))
            act_sq_sum += float(np.mean(delta_rate * delta_rate))
            act_samples += 1
        prev_target_q = target_q_mj.copy()
        for _ in range(4):
            tau = kp * (target_q_mj - d.qpos[7:36]) - kd * d.qvel[6:35]
            d.ctrl[:] = np.clip(tau, -tau_lim, tau_lim)
            mj.mj_step(m, d)
        mj.mj_forward(m, d)
        total_steps += 1
        state84 = read_state()

        hand = _box_hand_contacts(m, d, box_geom, wrist_ids)
        hand_steps += int(hand)
        box_pos = state84[71:74]
        torso_pos = np.asarray(d.xpos[torso_body_id], dtype=np.float32)
        e_obj_sum += float(np.linalg.norm(box_pos - runner.ref_obj_frame(step_in_plan + 1)))
        e_torso_sum += float(np.linalg.norm(torso_pos - runner.ref_torso_frame(step_in_plan + 1)))
        ground = any(
            int(c.geom1) in (box_geom, floor_geom) and int(c.geom2) in (box_geom, floor_geom)
            for c in d.contact
        )
        t1_hit = any(
            int(c.geom1) in (box_geom, tables_geom[0]) and int(c.geom2) in (box_geom, tables_geom[0])
            for c in d.contact
        )
        t2_hit = any(
            int(c.geom1) in (box_geom, tables_geom[1]) and int(c.geom2) in (box_geom, tables_geom[1])
            for c in d.contact
        )
        disp = box_pos - init_box
        cleared = float(disp[2]) > 0.025 or float(np.linalg.norm(disp[:2])) > 0.10
        held_free = hand and not (ground or t1_hit or t2_hit) and cleared
        carried_steps = carried_steps + 1 if held_free else 0
        lifted = lifted or (carried_steps >= 2)
        lost_steps = lost_steps + 1 if (lifted and not hand) else 0

        goal_err_now = float(np.linalg.norm(box_pos - goal_box))
        lin_spd = float(np.linalg.norm(state84[78:81]))
        ang_spd = float(np.linalg.norm(state84[81:84]))
        if ((ground and float(disp[2]) < -0.08 and not hand) or (lost_steps >= 3 and (ground or t1_hit or t2_hit))) and goal_err_now >= OFFICIAL_GOAL_TOL_M:
            dropped = True

        fallen = _is_fallen(state84)
        plan_done = step_in_plan >= runner.plan_length() - 1

        if open_loop_snapshot is None and (plan_done or fallen):
            open_loop_snapshot = {
                "final_goal_err_m": goal_err_now,
                "fallen": fallen,
                "lifted": lifted,
                "dropped": dropped,
                "box_stable": bool(lin_spd < 0.1 and ang_spd < 0.2),
                "steps": total_steps,
                "E_obj_m": e_obj_sum / max(total_steps, 1),
                "E_torso_m": e_torso_sum / max(total_steps, 1),
                "hand_contact_ratio": hand_steps / max(total_steps, 1),
            }
            if not allow_replan:
                break

        if fallen:
            break

        if allow_replan and replans < max_replans:
            if cooldown > 0:
                cooldown -= 1
            else:
                obj_static = (lin_spd <= 0.05 and ang_spd <= 0.05)
                ref_dev = float(np.linalg.norm(box_pos - runner.ref_obj_at(step_in_plan + 1)))
                need_replan = obj_static and ((ref_dev > 1.5) or (plan_done and goal_err_now > OFFICIAL_GOAL_TOL_M))
                if need_replan:
                    try:
                        if planner_mode == "official_cfgen":
                            new_traj, _ = generate_official_cfgen_raw(
                                sim_root, state84[:7], state84[71:78], half_arr, goal_box
                            )
                            new_plan = new_traj if isinstance(runner, _OfficialOnnxAdapter) else pack_official_cfgen_direct(new_traj)
                        elif planner_mode == "v7_cfgen":
                            frames, _ = generate_recovery_clip(sim_root, state84[:7], state84[71:78], half_arr, goal_box)
                            new_plan = frames
                        else:
                            raise ValueError(planner_mode)
                        runner.set_plan(new_plan)
                        step_in_plan = 0
                        cooldown = 80
                        replans += 1
                        continue
                    except Exception:
                        break
                elif plan_done:
                    break
        elif plan_done:
            break

        step_in_plan += 1

    final_state = read_state()
    final_goal_err = float(np.linalg.norm(final_state[71:74] - goal_box))
    final_fallen = bool(fallen or _is_fallen(final_state))
    final_lin = float(np.linalg.norm(final_state[78:81]))
    final_ang = float(np.linalg.norm(final_state[81:84]))
    box_stable = bool(final_lin < 0.1 and final_ang < 0.2)

    if open_loop_snapshot is None:
        open_loop_snapshot = {
            "final_goal_err_m": final_goal_err,
            "fallen": final_fallen,
            "lifted": lifted,
            "dropped": dropped,
            "box_stable": box_stable,
            "steps": total_steps,
            "E_obj_m": e_obj_sum / max(total_steps, 1),
            "E_torso_m": e_torso_sum / max(total_steps, 1),
            "hand_contact_ratio": hand_steps / max(total_steps, 1),
        }

    ol = open_loop_snapshot
    ol_off_succ = bool((not ol["fallen"]) and (ol["final_goal_err_m"] < OFFICIAL_GOAL_TOL_M))
    ol_strict_succ = bool(ol_off_succ and ol["lifted"] and ol["box_stable"])

    cl_off_succ = bool((not final_fallen) and (final_goal_err < OFFICIAL_GOAL_TOL_M))
    cl_strict_succ = bool(cl_off_succ and lifted and box_stable)

    e_obj_mean = e_obj_sum / max(total_steps, 1)
    e_torso_mean = e_torso_sum / max(total_steps, 1)
    act_rate_mean = act_abs_sum / max(act_samples, 1) if act_samples else None
    act_rate_rms = math.sqrt(act_sq_sum / max(act_samples, 1)) if act_samples else None
    hand_ratio = hand_steps / max(total_steps, 1)

    if cl_off_succ:
        reason = "success"
    elif final_fallen and not lifted:
        reason = "approach_or_lift_fall"
    elif final_fallen:
        reason = "carry_fall"
    elif not lifted:
        reason = "never_lifted"
    elif dropped:
        reason = "carry_drop"
    else:
        reason = "place_miss"

    return {
        "ep_id": ep["ep_id"],
        "label": ep["label"],
        "box_size_cm": [round(float(200.0 * x), 1) for x in half_arr],
        "max_box_cm": round(float(200.0 * max(half_arr)), 1),
        "travel_dist_m": round(float(np.linalg.norm(goal_box[:2] - init_box[:2])), 3),
        "init_dist_m": init_dist_m,
        "init_heading_deg": init_heading_deg,
        "init_z_m": round(float(init_box[2]), 3),
        "goal_z_m": round(float(goal_box[2]), 3),
        "plan_failed": False,
        "open_loop_official_success": ol_off_succ,
        "open_loop_strict_success": ol_strict_succ,
        "open_loop_goal_err_m": round(float(ol["final_goal_err_m"]), 4),
        "open_loop_E_obj_m": round(float(ol["E_obj_m"]), 4),
        "open_loop_E_torso_m": round(float(ol["E_torso_m"]), 4),
        "open_loop_hand_contact_ratio": round(float(ol["hand_contact_ratio"]), 4),
        "official_success": cl_off_succ,
        "strict_success": cl_strict_succ,
        "dist_only_success": bool(final_goal_err < OFFICIAL_GOAL_TOL_M),
        "final_goal_err_m": round(final_goal_err, 4),
        "E_obj_m": round(float(e_obj_mean), 4),
        "E_torso_m": round(float(e_torso_mean), 4),
        "action_rate_mean_rad_s": round(float(act_rate_mean), 3) if act_rate_mean is not None else None,
        "action_rate_rms_rad_s": round(float(act_rate_rms), 3) if act_rate_rms is not None else None,
        "hand_contact_ratio": round(float(hand_ratio), 4),
        "box_stable": box_stable,
        "lifted": bool(lifted),
        "fallen": final_fallen,
        "dropped": bool(dropped),
        "replans": replans,
        "steps": total_steps,
        "fail_reason": reason,
    }


def _worker_eval_chunk(spec):
    (
        sim_root,
        checkpoint_path,
        actor_state,
        goal_mode,
        drop_penalty,
        enc_w,
        dec_w,
        methods,
        episodes,
        ik_subset,
    ) = spec

    sim_root = Path(sim_root)
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    consts = _make_mj_env_consts(sim_root)

    runners = {}
    if "official_transformer" in methods:
        runners["official_transformer"] = _OfficialOnnxAdapter(
            sim_root, sim_root / "policy/omnicontact/model/omnicontact_transformer.onnx", consts
        )
    if "official_mlp" in methods:
        runners["official_mlp"] = _OfficialOnnxAdapter(
            sim_root, sim_root / "policy/omnicontact/model/omnicontact_mlp.onnx", consts
        )
    net = None
    if any(m in methods for m in ("v72_official_cfgen", "v72_v7_cfgen", "v72_official_5link_ik")):
        net = _CpuEvalNet(sim_root, goal_mode, drop_penalty, enc_w, dec_w, actor_state)
        for key in ("v72_official_cfgen", "v72_v7_cfgen", "v72_official_5link_ik"):
            if key in methods:
                runners[key] = _V72ActorRunner(net)

    from carrywm.sim import _generator

    results = {m: [] for m in methods}
    fk_stats = []
    ik_stats = []

    for ep in episodes:
        root7 = np.array(ep["root_pos"] + ep["root_quat"], dtype=np.float32)
        box7 = np.array(ep["init_pos"] + ep["init_quat"], dtype=np.float32)
        half = np.asarray(ep["half"], dtype=np.float32)
        goal = np.asarray(ep["goal_pos"], dtype=np.float32)

        off_traj = None
        off_clip = None
        off_plan_err = False
        try:
            off_traj, _ = generate_official_cfgen_raw(sim_root, root7, box7, half, goal)
            off_clip = pack_official_cfgen_direct(off_traj)
            if not fk_stats or len(fk_stats) < 8:
                fk_stats.append(verify_official_fk_consistency(sim_root, off_traj))
        except Exception:
            off_plan_err = True

        if "official_transformer" in methods:
            results["official_transformer"].append(
                _run_episode_with_runner(
                    sim_root, ep, consts, runners["official_transformer"],
                    "official_cfgen", off_traj, allow_replan=True,
                )
            )
        if "official_mlp" in methods:
            results["official_mlp"].append(
                _run_episode_with_runner(
                    sim_root, ep, consts, runners["official_mlp"],
                    "official_cfgen", off_traj, allow_replan=True,
                )
            )
        if "v72_official_cfgen" in methods:
            results["v72_official_cfgen"].append(
                _run_episode_with_runner(
                    sim_root, ep, consts, runners["v72_official_cfgen"],
                    "official_cfgen", off_clip, allow_replan=True,
                )
            )
        if "v72_v7_cfgen" in methods:
            from carrywm.geometry_v7 import generate_v7_clip

            v7_clip = None
            try:
                v7_clip, _ = generate_v7_clip(sim_root, root7, box7, half, goal)
            except Exception:
                v7_clip = None
            results["v72_v7_cfgen"].append(
                _run_episode_with_runner(
                    sim_root, ep, consts, runners["v72_v7_cfgen"],
                    "v7_cfgen", v7_clip, allow_replan=True,
                )
            )
        if "v72_official_5link_ik" in methods and ep["ep_id"] < ik_subset:
            ik_clip = None
            if off_traj is not None and not off_plan_err:
                try:
                    ik_clip, st = reconstruct_clip_from_5links_ik(
                        sim_root, off_traj, init_base_pos=root7[:3], init_base_quat=root7[3:7]
                    )
                    ik_stats.append(st)
                except Exception:
                    ik_clip = None
            results["v72_official_5link_ik"].append(
                _run_episode_with_runner(
                    sim_root, ep, consts, runners["v72_official_5link_ik"],
                    "official_cfgen", ik_clip, allow_replan=False,
                )
            )

    return results, fk_stats, ik_stats


def _summarize_records(recs: list[dict]) -> dict:
    n = len(recs)
    if not n:
        return {}
    ol_off = [r for r in recs if r.get("open_loop_official_success")]
    ol_st = [r for r in recs if r.get("open_loop_strict_success")]
    cl_off = [r for r in recs if r.get("official_success")]
    cl_st = [r for r in recs if r.get("strict_success")]
    reasons = {}
    for r in recs:
        reasons[r["fail_reason"]] = reasons.get(r["fail_reason"], 0) + 1

    in_box = [r for r in recs if r.get("max_box_cm", 0) <= 35.0]
    big_box = [r for r in recs if r.get("max_box_cm", 0) > 35.0]
    short_d = [r for r in recs if r.get("travel_dist_m", 0) <= 3.0]
    mid_d = [r for r in recs if 3.0 < r.get("travel_dist_m", 0) <= 6.0]
    far_d = [r for r in recs if r.get("travel_dist_m", 0) > 6.0]
    near_d = [r for r in recs if r.get("init_dist_m", 0) <= 1.5]
    far_init_d = [r for r in recs if r.get("init_dist_m", 0) > 1.5]
    small_h = [r for r in recs if r.get("init_heading_deg", 0) <= 30.0]
    big_h = [r for r in recs if r.get("init_heading_deg", 0) > 30.0]

    def rate(arr, key="official_success"):
        return round(float(sum(1 for x in arr if x.get(key)) / len(arr)), 4) if arr else None

    def opt_mean(key: str, digits: int = 4):
        vals = [r[key] for r in recs if r.get(key) is not None and math.isfinite(float(r[key]))]
        return round(float(np.mean(vals)), digits) if vals else None

    ol_succ_rate = round(len(ol_off) / n, 4)
    cl_succ_rate = round(len(cl_off) / n, 4)
    ol_err_m = round(float(np.mean([r.get("open_loop_goal_err_m", r["final_goal_err_m"]) for r in recs])), 4)
    cl_err_m = round(float(np.mean([r["final_goal_err_m"] for r in recs])), 4)
    e_obj_m = opt_mean("E_obj_m", 4)
    ol_e_obj_m = opt_mean("open_loop_E_obj_m", 4)
    e_torso_m = opt_mean("E_torso_m", 4)
    ol_e_torso_m = opt_mean("open_loop_E_torso_m", 4)
    act_mean = opt_mean("action_rate_mean_rad_s", 3)
    act_rms = opt_mean("action_rate_rms_rad_s", 3)
    hand_ratio = opt_mean("hand_contact_ratio", 4)
    box_stable_rate = round(sum(1 for r in recs if r.get("box_stable")) / n, 4)

    return {
        "episodes": n,
        "open_loop_official_succ_rate": ol_succ_rate,
        "open_loop_strict_succ_rate": round(len(ol_st) / n, 4),
        "open_loop_mean_goal_err_m": ol_err_m,
        "official_succ_rate": cl_succ_rate,
        "strict_succ_rate": round(len(cl_st) / n, 4),
        "dist_only_succ_rate": round(sum(1 for r in recs if r["dist_only_success"]) / n, 4),
        "mean_goal_err_m": cl_err_m,
        "R_succ_pct": round(100.0 * ol_succ_rate, 2),
        "R_succ_replan_pct": round(100.0 * cl_succ_rate, 2),
        "E_obj_T_m": cl_err_m,
        "open_loop_E_obj_T_m": ol_err_m,
        "E_obj_m": e_obj_m,
        "open_loop_E_obj_m": ol_e_obj_m,
        "E_torso_m": e_torso_m,
        "open_loop_E_torso_m": ol_e_torso_m,
        "action_rate_mean_rad_s": act_mean,
        "action_rate_rms_rad_s": act_rms,
        "hand_contact_ratio": hand_ratio,
        "box_stable_rate": box_stable_rate,
        "N_hoi": None,
        "N_hoi_note": "OmniContact Appendix E.2 scores N_hoi (0-10) via Gemini-3.1-Pro blind video judge; physical smoothness proxies recorded in action_rate_*/E_torso_m/hand_contact_ratio.",
        "lift_rate": round(sum(1 for r in recs if r["lifted"]) / n, 4),
        "drop_rate": round(sum(1 for r in recs if r["dropped"]) / n, 4),
        "fall_rate": round(sum(1 for r in recs if r["fallen"]) / n, 4),
        "mean_replans": round(float(np.mean([r["replans"] for r in recs])), 3),
        "mean_steps": round(float(np.mean([r["steps"] for r in recs])), 1),
        "fail_reasons": reasons,
        "by_box_size": {
            "le_35cm": {"n": len(in_box), "official_succ": rate(in_box), "open_loop_succ": rate(in_box, "open_loop_official_success"), "lift": rate(in_box, "lifted")},
            "gt_35cm": {"n": len(big_box), "official_succ": rate(big_box), "open_loop_succ": rate(big_box, "open_loop_official_success"), "lift": rate(big_box, "lifted")},
        },
        "by_travel_dist": {
            "le_3m": {"n": len(short_d), "official_succ": rate(short_d)},
            "3_to_6m": {"n": len(mid_d), "official_succ": rate(mid_d)},
            "gt_6m": {"n": len(far_d), "official_succ": rate(far_d)},
        },
        "by_init_dist": {
            "le_1_5m": {"n": len(near_d), "official_succ": rate(near_d), "open_loop_succ": rate(near_d, "open_loop_official_success"), "fall_rate": rate(near_d, "fallen")},
            "gt_1_5m": {"n": len(far_init_d), "official_succ": rate(far_init_d), "open_loop_succ": rate(far_init_d, "open_loop_official_success"), "fall_rate": rate(far_init_d, "fallen")},
        },
        "by_init_heading": {
            "le_30deg": {"n": len(small_h), "official_succ": rate(small_h), "open_loop_succ": rate(small_h, "open_loop_official_success"), "fall_rate": rate(small_h, "fallen")},
            "gt_30deg": {"n": len(big_h), "official_succ": rate(big_h), "open_loop_succ": rate(big_h, "open_loop_official_success"), "fall_rate": rate(big_h, "fallen")},
        },
    }


def _run_suite(
    pool,
    args,
    actor_state,
    cfg,
    episodes: list[dict],
    methods: tuple[str, ...],
    *,
    ik_subset: int = 0,
):
    n_workers = min(args.workers, len(episodes))
    chunks = [episodes[i::n_workers] for i in range(n_workers)]
    specs = []
    for idx, chunk in enumerate(chunks):
        sub = list(methods)
        do_ik = "v72_official_5link_ik" in methods and any(ep["ep_id"] < ik_subset for ep in chunk)
        if not do_ik and "v72_official_5link_ik" in sub:
            sub = [m for m in sub if m != "v72_official_5link_ik"]
        specs.append(
            (
                str(args.sim_root),
                str(args.checkpoint),
                actor_state,
                str(cfg.get("goal_mode", "full")),
                float(cfg.get("drop_penalty", 1.0)),
                int(cfg.get("encoder_width", 2048)),
                int(cfg.get("decoder_width", 2048)),
                tuple(sub),
                chunk,
                ik_subset,
            )
        )

    # Filter v72_official_5link_ik to episodes < ik_subset cleanly after collection.
    all_res = {m: [] for m in methods}
    all_fk = []
    all_ik = []
    for res, fk_s, ik_s in pool.map(_worker_eval_chunk, specs):
        all_fk.extend(fk_s)
        all_ik.extend(ik_s)
        for m, lst in res.items():
            if m == "v72_official_5link_ik" and ik_subset > 0:
                all_res[m].extend([x for x in lst if x["ep_id"] < ik_subset])
            else:
                all_res[m].extend(lst)

    for m in all_res:
        all_res[m].sort(key=lambda r: r["ep_id"])

    summary = {m: _summarize_records(all_res[m]) for m in methods if all_res[m]}
    fk_summary = {
        "samples": len(all_fk),
        "mean_pos_err_m": float(np.mean([x["mean_pos_err_m"] for x in all_fk])) if all_fk else None,
        "max_pos_err_m": float(np.max([x["max_pos_err_m"] for x in all_fk])) if all_fk else None,
    }
    ik_summary = {
        "episodes": len(all_ik),
        "mean_link_err_m": float(np.mean([x["ik_mean_link_err_m"] for x in all_ik])) if all_ik else None,
        "max_link_err_m": float(np.max([x["ik_max_link_err_m"] for x in all_ik])) if all_ik else None,
    }
    return {"summary": summary, "fk_consistency": fk_summary, "ik_5link_reconstruction": ik_summary, "episodes": all_res}


def _print_suite_summary(tag: str, summary: dict, *, show_box_split: bool = False):
    for m, s in summary.items():
        box_part = ""
        if show_box_split:
            box_part = (
                f" <=35cm={100*(s['by_box_size']['le_35cm']['official_succ'] or 0):5.1f}%"
                f" >35cm={100*(s['by_box_size']['gt_35cm']['official_succ'] or 0):5.1f}%"
            )
        e_obj = s.get("E_obj_m")
        e_torso = s.get("E_torso_m")
        act_rms = s.get("action_rate_rms_rad_s")
        print(
            f"  {tag}[{m:22s}]: n={s['episodes']:3d} Rsucc={s['R_succ_pct']:5.1f}% "
            f"Rsucc*={s['R_succ_replan_pct']:5.1f}% strict={100*s['strict_succ_rate']:5.1f}% "
            f"EobjT={s['E_obj_T_m']:.3f}m Eobj={e_obj if e_obj is not None else -1:.3f}m "
            f"Etorso={e_torso if e_torso is not None else -1:.3f}m actRMS={act_rms if act_rms is not None else -1:.2f} "
            f"lift={100*s['lift_rate']:5.1f}% fall={100*s['fall_rate']:4.1f}%{box_part}",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--sim-root", required=True, type=Path)
    p.add_argument("--motion-dir", type=Path, default=None)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--bench-a-episodes", type=int, default=200)
    p.add_argument("--bench-b-episodes", type=int, default=100)
    p.add_argument("--ik-check-episodes", type=int, default=40)
    p.add_argument("--methods", type=str, default=None,
                   help="Optional comma-separated subset of methods to run")
    p.add_argument("--skip-ik", action="store_true",
                   help="Skip v72_official_5link_ik diagnostic subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.bench_a_episodes = 6
        args.bench_b_episodes = 0
        args.ik_check_episodes = 6
        args.workers = min(args.workers, 6)

    t0 = time.monotonic()
    # Full training checkpoints also contain the replay buffer. Map the file
    # so CPU evaluation only reads the actor tensor storage it actually uses.
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = dict(ck.get("config", {}))
    actor_state = {k: v.detach().cpu() for k, v in ck["models"][1].items()}

    all_methods = ("official_mlp", "official_transformer", "v72_official_cfgen", "v72_v7_cfgen", "v72_official_5link_ik")
    selected = tuple(x.strip() for x in args.methods.split(",") if x.strip()) if args.methods else all_methods
    if args.skip_ik:
        selected = tuple(m for m in selected if m != "v72_official_5link_ik")
    for m in selected:
        if m not in all_methods:
            raise ValueError(f"Unknown method {m!r}; expected subset of {all_methods}")

    methods_a = selected
    methods_b = tuple(m for m in selected if m != "v72_official_5link_ik")
    methods_c = selected
    methods_d = selected

    eps_a = sample_official_episodes(args.bench_a_episodes, args.seed, randomize_box_size=True)
    eps_b = sample_official_episodes(args.bench_b_episodes, args.seed + 1000, randomize_box_size=False) if args.bench_b_episodes > 0 else []
    eps_c = load_official_demo_episodes(args.sim_root)
    eps_d = load_case1_test_episodes(args.motion_dir) if args.motion_dir is not None else []

    ctx = mp.get_context("spawn")
    out = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_label": args.checkpoint.parent.name,
        "iteration": int(ck["iteration"]),
        "real_steps": int(ck.get("real_steps", 0)),
        "seed": args.seed,
    }
    del ck
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_eval_worker) as pool:
        t_a = time.monotonic()
        print(f"=== Running Benchmark A: Official Randomized Carry Box ({len(eps_a)} eps, [0.20, 0.50]m) ===", flush=True)
        res_a = _run_suite(pool, args, actor_state, cfg, eps_a, methods_a, ik_subset=args.ik_check_episodes)
        out["benchmark_a_official_random"] = res_a
        print(f"--- Benchmark A done in {time.monotonic() - t_a:.1f}s ---", flush=True)
        _print_suite_summary("A", res_a["summary"], show_box_split=True)
        print(f"  FK check: {res_a['fk_consistency']} | 5-link IK: {res_a['ik_5link_reconstruction']}", flush=True)

        if eps_b and methods_b:
            t_b = time.monotonic()
            print(f"=== Running Benchmark B: Official Sim2Sim 30cm Cube ({len(eps_b)} eps, [0.30, 0.30, 0.30]m) ===", flush=True)
            res_b = _run_suite(pool, args, actor_state, cfg, eps_b, methods_b)
            out["benchmark_b_30cm_cube"] = res_b
            print(f"--- Benchmark B done in {time.monotonic() - t_b:.1f}s ---", flush=True)
            _print_suite_summary("B", res_b["summary"])

        if eps_c and not args.smoke:
            t_c = time.monotonic()
            print(f"=== Running Benchmark C: Official sim2sim data/carrybox Demos ({len(eps_c)} eps) ===", flush=True)
            res_c = _run_suite(pool, args, actor_state, cfg, eps_c, methods_c, ik_subset=len(eps_c))
            out["benchmark_c_sim2sim_demos"] = res_c
            print(f"--- Benchmark C done in {time.monotonic() - t_c:.1f}s ---", flush=True)
            _print_suite_summary("C", res_c["summary"])

        if eps_d:
            eps_d_run = eps_d[:6] if args.smoke else eps_d
            ik_d = min(6, len(eps_d_run)) if args.smoke else min(20, len(eps_d_run))
            t_d = time.monotonic()
            print(f"=== Running Benchmark D: MoCap Fairness Subset ({len(eps_d_run)} eps, case1_carry/test) ===", flush=True)
            res_d = _run_suite(pool, args, actor_state, cfg, eps_d_run, methods_d, ik_subset=ik_d)
            out["benchmark_d_fairness_subset"] = res_d
            print(f"--- Benchmark D done in {time.monotonic() - t_d:.1f}s ---", flush=True)
            _print_suite_summary("D", res_d["summary"])

    out["wall_seconds"] = round(time.monotonic() - t0, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean_json(out), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(f"=== ALL BENCHMARKS SAVED ({out['wall_seconds']}s): {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
