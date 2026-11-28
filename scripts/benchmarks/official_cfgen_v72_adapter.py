"""Pure-kinematics adapter from unmodified official OmniContact CF-Gen to V7.2 122-dim input."""
from functools import lru_cache
from pathlib import Path
import sys

import mujoco as mj
import numpy as np


DT = 0.02
RAW_DIM = 111


def _ensure_sim_root(sim_root: Path):
    sim_root = Path(sim_root).resolve()
    sim_str = str(sim_root)
    if sim_str not in sys.path:
        sys.path.insert(0, sim_str)
    return sim_root


@lru_cache(maxsize=2)
def _official_generator(sim_root: str):
    _ensure_sim_root(Path(sim_root))
    from policy.omnicontact.CFgen_meta2_carrybox import CfGenCarryBox

    return CfGenCarryBox(pad=30)


def _normalize_pose_batch(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-5):
        raise ValueError("Reference contains a near-zero quaternion")
    return np.concatenate((pos, quat / norm), axis=-1)


def generate_official_cfgen_raw(
    sim_root: Path,
    root_pose7: np.ndarray,
    box_pose7: np.ndarray,
    half_dims: np.ndarray,
    goal_pos: np.ndarray,
) -> tuple[dict, float]:
    """Run unmodified official CfGenCarryBox with official rules/IK parameters."""
    gen = _official_generator(str(Path(sim_root).resolve()))
    traj, target_yaw = gen.generate(
        np.asarray(root_pose7[:3], dtype=np.float32),
        np.asarray(root_pose7[3:7], dtype=np.float32),
        np.asarray(box_pose7[:3], dtype=np.float32),
        np.asarray(box_pose7[3:7], dtype=np.float32),
        box_half_dims=np.asarray(half_dims, dtype=np.float32),
        target_obj_pos=np.asarray(goal_pos, dtype=np.float32),
    )
    return traj, float(target_yaw)


def pack_official_cfgen_direct(traj: dict) -> np.ndarray:
    """Pack official CF-Gen output (links, contact, IK+FK q/base, finite-diff dq) into V7.2 (T, 111) clip."""
    pairs = (
        ("ref_left_wrist_pos", "ref_left_wrist_quat"),
        ("ref_right_wrist_pos", "ref_right_wrist_quat"),
        ("ref_torso_future_pos", "ref_torso_future_quat"),
        ("ref_left_ankle_future_pos", "ref_left_ankle_future_quat"),
        ("ref_right_ankle_future_pos", "ref_right_ankle_future_quat"),
    )
    links = np.stack([_normalize_pose_batch(traj[p], traj[r]) for p, r in pairs], axis=1)
    q = np.asarray(traj["dof_pos"], dtype=np.float32)
    dq = np.gradient(q, DT, axis=0).astype(np.float32)
    base_p = _normalize_pose_batch(traj["ref_base_pos"], traj["ref_base_quat"])
    obj_p = _normalize_pose_batch(traj["ref_object_pos"], traj["ref_object_quat"])
    contact = np.asarray(traj["ref_contact"], dtype=np.float32).reshape(-1, 4)
    frames = np.concatenate((links.reshape(-1, 35), contact, q, dq, base_p, obj_p), axis=-1).astype(np.float32)
    if frames.shape[1] != RAW_DIM or not np.isfinite(frames).all():
        raise ValueError(f"Invalid packed official CF-Gen clip shape={frames.shape}")
    return frames


def verify_official_fk_consistency(sim_root: Path, traj: dict) -> dict[str, float]:
    """Check max/mean Euclidean mismatch between FK(ref_base, dof_pos) and CF-Gen's 5 link trajectories."""
    _ensure_sim_root(Path(sim_root))
    from policy.omnicontact.CFgen_meta1_loco import KINEMATICS

    body_map = (
        ("left_palm_link", "ref_left_wrist_pos"),
        ("right_palm_link", "ref_right_wrist_pos"),
        ("torso_link", "ref_torso_future_pos"),
        ("left_ankle_pitch_link", "ref_left_ankle_future_pos"),
        ("right_ankle_pitch_link", "ref_right_ankle_future_pos"),
    )
    base_p = np.asarray(traj["ref_base_pos"], dtype=np.float32)
    base_q = np.asarray(traj["ref_base_quat"], dtype=np.float32)
    dof = np.asarray(traj["dof_pos"], dtype=np.float32)
    errs = []
    # Sample evenly up to 64 frames for fast validation.
    step = max(1, len(dof) // 64)
    for idx in range(0, len(dof), step):
        fk = KINEMATICS.forward(dof[idx], base_p[idx], base_q[idx])
        for body_name, traj_key in body_map:
            errs.append(float(np.linalg.norm(fk[body_name]["pos"] - traj[traj_key][idx])))
    return {"mean_pos_err_m": float(np.mean(errs)), "max_pos_err_m": float(np.max(errs))}


@lru_cache(maxsize=2)
def _ik_workspace(sim_root: str):
    root = _ensure_sim_root(Path(sim_root))
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ

    model = mj.MjModel.from_xml_path(str(root / "g1_description" / "g1_29dof.xml"))
    data = mj.MjData(model)
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"Expected G1 nq=36 nv=35, got {model.nq}/{model.nv}")
    body_names = (
        "left_palm_link",
        "right_palm_link",
        "torso_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
    )
    body_ids = [model.body(name).id for name in body_names]
    limits = model.jnt_range[1:30].copy()  # hinge joints 0..28 follow free root joint 0
    return model, data, body_ids, limits, np.asarray(DEFAULT_JOINT_POS_MJ, dtype=np.float64).copy()


def _rot_vec_err(target_mat: np.ndarray, cur_mat: np.ndarray) -> np.ndarray:
    """Small rotation error vector in world frame mapping cur_mat -> target_mat."""
    return 0.5 * (
        np.cross(cur_mat[:, 0], target_mat[:, 0])
        + np.cross(cur_mat[:, 1], target_mat[:, 1])
        + np.cross(cur_mat[:, 2], target_mat[:, 2])
    )


def _quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    m = np.empty(9, dtype=np.float64)
    mj.mju_quat2Mat(m, np.asarray(q_wxyz, dtype=np.float64))
    return m.reshape(3, 3)


def reconstruct_clip_from_5links_ik(
    sim_root: Path,
    traj: dict,
    *,
    init_base_pos: np.ndarray | None = None,
    init_base_quat: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Reconstruct (base_p, base_q, dof_pos, dq) purely via whole-body MuJoCo Jacobian IK from the 5 key links.

    Explicitly ignores traj['dof_pos'], traj['ref_base_pos'], and traj['ref_base_quat'] so that
    only official CF-Gen's 5 key-link poses, contact signals, and object trajectory are used.
    """
    model, data, body_ids, limits, default_q = _ik_workspace(str(Path(sim_root).resolve()))
    lw_p = np.asarray(traj["ref_left_wrist_pos"], dtype=np.float64)
    lw_q = np.asarray(traj["ref_left_wrist_quat"], dtype=np.float64)
    rw_p = np.asarray(traj["ref_right_wrist_pos"], dtype=np.float64)
    rw_q = np.asarray(traj["ref_right_wrist_quat"], dtype=np.float64)
    to_p = np.asarray(traj["ref_torso_future_pos"], dtype=np.float64)
    to_q = np.asarray(traj["ref_torso_future_quat"], dtype=np.float64)
    la_p = np.asarray(traj["ref_left_ankle_future_pos"], dtype=np.float64)
    la_q = np.asarray(traj["ref_left_ankle_future_quat"], dtype=np.float64)
    ra_p = np.asarray(traj["ref_right_ankle_future_pos"], dtype=np.float64)
    ra_q = np.asarray(traj["ref_right_ankle_future_quat"], dtype=np.float64)
    obj_p = np.asarray(traj["ref_object_pos"], dtype=np.float32)
    obj_q = np.asarray(traj["ref_object_quat"], dtype=np.float32)
    contact = np.asarray(traj["ref_contact"], dtype=np.float32)

    n = len(to_p)
    target_pos = np.stack((lw_p, rw_p, to_p, la_p, ra_p), axis=1)  # (N, 5, 3)
    target_rot = np.empty((n, 5, 3, 3), dtype=np.float64)
    target_quats = (lw_q, rw_q, to_q, la_q, ra_q)
    for k, q_arr in enumerate(target_quats):
        for i in range(n):
            target_rot[i, k] = _quat_to_mat(q_arr[i])

    base_p_out = np.empty((n, 3), dtype=np.float32)
    base_q_out = np.empty((n, 4), dtype=np.float32)
    q_out = np.empty((n, 29), dtype=np.float32)

    # Warm start at frame 0 using torso XY/yaw and default upright joints.
    qpos = np.zeros(36, dtype=np.float64)
    if init_base_pos is not None and init_base_quat is not None:
        qpos[:3] = np.asarray(init_base_pos, dtype=np.float64)
        qpos[3:7] = np.asarray(init_base_quat, dtype=np.float64)
    else:
        qpos[:3] = [to_p[0, 0], to_p[0, 1], 0.77]
        # Extract yaw from torso target rotation.
        yaw0 = float(np.arctan2(target_rot[0, 2, 1, 0], target_rot[0, 2, 0, 0]))
        qpos[3:7] = [np.cos(0.5 * yaw0), 0.0, 0.0, np.sin(0.5 * yaw0)]
    qpos[7:36] = default_q

    jp = np.empty((3, 35), dtype=np.float64)
    jr = np.empty((3, 35), dtype=np.float64)
    eye35 = np.eye(35, dtype=np.float64)
    # Weights: [LW, RW, Torso, LA, RA]
    pos_w = np.array([2.0, 2.0, 2.5, 2.0, 2.0], dtype=np.float64)
    rot_w = np.array([0.8, 0.8, 0.6, 0.8, 0.8], dtype=np.float64)
    # Smooth posture regularization weighted lightly so exact 5-link targets win.
    reg_diag = np.concatenate((np.full(6, 1e-4), np.full(29, 4e-3)))
    reg_mat = np.diag(reg_diag)

    pos_errs = []
    for i in range(n):
        prev_qpos = qpos.copy()
        max_iters = 20 if i == 0 else 6
        for _ in range(max_iters):
            data.qpos[:] = qpos
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)

            jac_rows = []
            rhs_rows = []
            max_e = 0.0
            for k, bid in enumerate(body_ids):
                pe = target_pos[i, k] - data.xpos[bid]
                re = _rot_vec_err(target_rot[i, k], data.xmat[bid].reshape(3, 3))
                max_e = max(max_e, float(np.linalg.norm(pe)))
                mj.mj_jacBody(model, data, jp, jr, bid)
                jac_rows.append(pos_w[k] * jp)
                rhs_rows.append(pos_w[k] * pe)
                jac_rows.append(rot_w[k] * jr)
                rhs_rows.append(rot_w[k] * re)
            if max_e < 5e-4:
                break

            J = np.vstack(jac_rows)
            r = np.concatenate(rhs_rows)
            # Soft pull toward previous joints to resolve wrist/waist redundancy smoothly.
            posture_rhs = np.zeros(35, dtype=np.float64)
            anchor_q = default_q if i == 0 else prev_qpos[7:36]
            posture_rhs[6:] = reg_diag[6:] * (anchor_q - qpos[7:36])

            A = J.T @ J + reg_mat + 1e-4 * eye35
            b = J.T @ r + posture_rhs
            dq_step = np.linalg.solve(A, b)
            step_scale = min(1.0, 0.25 / max(1e-8, float(np.abs(dq_step).max())))
            dq_step *= step_scale
            mj.mj_integratePos(model, qpos, dq_step, 1.0)
            qpos[7:36] = np.clip(qpos[7:36], limits[:, 0], limits[:, 1])

        data.qpos[:] = qpos
        mj.mj_kinematics(model, data)
        frame_err = max(float(np.linalg.norm(target_pos[i, k] - data.xpos[bid])) for k, bid in enumerate(body_ids))
        pos_errs.append(frame_err)

        # Keep quaternion hemisphere continuous across frames.
        if i > 0 and float(np.dot(base_q_out[i - 1], qpos[3:7])) < 0.0:
            qpos[3:7] *= -1.0
        base_p_out[i] = qpos[:3].astype(np.float32)
        base_q_out[i] = (qpos[3:7] / np.linalg.norm(qpos[3:7])).astype(np.float32)
        q_out[i] = qpos[7:36].astype(np.float32)

    dq_out = np.gradient(q_out, DT, axis=0).astype(np.float32)
    links = np.stack(
        [
            _normalize_pose_batch(traj[p], traj[r])
            for p, r in (
                ("ref_left_wrist_pos", "ref_left_wrist_quat"),
                ("ref_right_wrist_pos", "ref_right_wrist_quat"),
                ("ref_torso_future_pos", "ref_torso_future_quat"),
                ("ref_left_ankle_future_pos", "ref_left_ankle_future_quat"),
                ("ref_right_ankle_future_pos", "ref_right_ankle_future_quat"),
            )
        ],
        axis=1,
    )
    frames = np.concatenate(
        (
            links.reshape(-1, 35),
            contact.reshape(-1, 4),
            q_out,
            dq_out,
            _normalize_pose_batch(base_p_out, base_q_out),
            _normalize_pose_batch(obj_p, obj_q),
        ),
        axis=-1,
    ).astype(np.float32)
    stats = {
        "ik_mean_link_err_m": float(np.mean(pos_errs)),
        "ik_max_link_err_m": float(np.max(pos_errs)),
    }
    return frames, stats
