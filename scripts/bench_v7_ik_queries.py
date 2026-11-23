"""Time the host-to-host raw FK queries used inside the V7 recovery residual."""
import argparse
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np
import torch

from carrywm.data import load_demo, split_paths
from carrywm.recovery_gpu import BatchedRecoveryFK


LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_palm_link",
         "right_palm_link", "torso_link", "left_ankle_pitch_link", "right_ankle_pitch_link")
ACTIVE = [j for j in range(29) if j not in (12, 13)]


def perturbations(pose, joint_bounds):
    """Perturb one local degree of freedom per row, as in a two-point Jacobian."""
    out = np.repeat(pose[None], 29, axis=0)
    eps = np.sqrt(np.finfo(np.float64).eps)
    out[0, 31] += max(abs(pose[31]) * 1e-4, eps)
    pitch = np.array([np.cos(eps / 2), 0., np.sin(eps / 2), 0.])
    mj.mju_mulQuat(out[1, 32:], pose[32:], pitch)
    for row, j in enumerate(ACTIVE, 2):
        h = 1e-4 * pose[j] if pose[j] else eps
        lo, hi = joint_bounds[j]
        if pose[j] + h < lo or pose[j] + h > hi:
            h = -h
        out[row, j] += h
    return out


def cpu_query(model, data, bodies, inputs):
    out = np.empty((len(inputs), len(bodies), 7), dtype=np.float64)
    for i, pose in enumerate(inputs):
        data.qpos[:7], data.qpos[7:] = pose[29:], pose[:29]
        mj.mj_kinematics(model, data)
        out[i, :, :3], out[i, :, 3:] = data.xpos[bodies], data.xquat[bodies]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 1:
        parser.error("reps and warmup must be positive")
    dev = torch.device(args.device)
    if dev.type != "cuda":
        parser.error("device must be a CUDA device for the host transfer comparison")
    torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    model = mj.MjModel.from_xml_path(str(args.sim_root / "g1_description/g1_29dof.xml"))
    data = mj.MjData(model)
    bodies = np.asarray([model.body(name).id for name in LINKS])
    gpu_bodies = torch.as_tensor(bodies, device=dev)
    fk = BatchedRecoveryFK(args.sim_root, dev)
    joint_bounds = model.jnt_range[np.argsort(model.jnt_qposadr)[1:]]
    rng = np.random.default_rng(args.seed)
    paths = split_paths(args.motion_dir, "train")
    chosen = rng.choice(len(paths), min(8, len(paths)), replace=False)
    clips = [load_demo(paths[i]).frames for i in chosen]
    contact = [np.flatnonzero((c[:, 37:39] > .5).all(axis=1)) for c in clips]

    def sample_pose():
        i = int(rng.integers(len(clips)))
        frame = rng.choice(contact[i]) if len(contact[i]) else rng.integers(len(clips[i]))
        raw = clips[i][frame]
        pose = np.concatenate((raw[39:68], raw[97:104])).astype(np.float64)
        pose[32:] /= np.linalg.norm(pose[32:])
        return pose

    @torch.inference_mode()
    def gpu_query(inputs):
        gpu_inputs = torch.from_numpy(inputs).to(dev)
        p, q = fk._body_poses(gpu_inputs, gpu_bodies)
        return torch.cat((p, q), dim=-1).cpu().numpy()

    report = dict(hardware=torch.cuda.get_device_name(dev), torch=str(torch.__version__),
                  mujoco=mj.__version__, seed=args.seed, reps=args.reps, warmup=args.warmup,
                  clips=[paths[i].name for i in chosen], links=LINKS, results=[],
                  scope="One CPU MuJoCo worker or one eager GPU batch, inputs and outputs on host. "
                        "29 finite-difference candidates per real contact pose, 232 represents 8 aligned "
                        "poses. Timings exclude residual formulas, clipping/trust-region decisions, "
                        "Jacobian assembly, process scheduling and parallel CPU IK workers.")
    for n in (1, 29, 232):
        samples = {"cpu": [], "gpu": []}
        max_pos = max_quat = max_raw_quat = 0.
        for rep in range(args.warmup + args.reps):
            if n == 1:
                batch = sample_pose()[None]
            else:
                batch = np.concatenate([perturbations(sample_pose(), joint_bounds) for _ in range(n // 29)])
            order = ("cpu", "gpu") if rep % 2 == 0 else ("gpu", "cpu")
            values = {}
            for route in order:
                torch.cuda.synchronize(dev)
                start = time.perf_counter_ns()
                values[route] = cpu_query(model, data, bodies, batch) if route == "cpu" else gpu_query(batch)
                # GPU .cpu() above synchronously returns the requested poses to SciPy's side.
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if rep >= args.warmup:
                    samples[route].append(elapsed)
            c, g = values["cpu"], values["gpu"]
            cq, gq = c[:, :, 3:], g[:, :, 3:]
            sign = np.where(np.sum(cq * gq, axis=-1, keepdims=True) < 0., -1., 1.)
            np.testing.assert_allclose(g[:, :, :3], c[:, :, :3], atol=2e-10, rtol=0.)
            np.testing.assert_allclose(gq * sign, cq, atol=2e-10, rtol=0.)
            max_pos = max(max_pos, float(np.max(abs(g[:, :, :3] - c[:, :, :3]))))
            max_quat = max(max_quat, float(np.max(abs(gq * sign - cq))))
            max_raw_quat = max(max_raw_quat, float(np.max(abs(gq - cq))))
        result = dict(candidates=n, max_position_error_m=max_pos, max_quat_error=max_quat,
                      max_quat_raw_component_error=max_raw_quat,
                      p50_ms={k: float(np.median(v)) for k, v in samples.items()},
                      p95_ms={k: float(np.percentile(v, 95)) for k, v in samples.items()})
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
