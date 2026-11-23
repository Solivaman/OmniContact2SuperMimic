#!/usr/bin/env python3
"""Compare the production CPU online recovery with CPU IK + batched GPU FK on captured drops."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path, help="Captured drops JSON with a samples array")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=3e-6)
    return parser.parse_args()


def plan_task(root, entry, method):
    from carrywm.recovery_gpu import prepare_recovery
    from carrywm.recovery_fk_v7 import generate_recovery_clip
    from carrywm.sim import _valid_recovery_grasp

    pose, box, half = (np.asarray(entry[key], dtype=np.float64) for key in ("root", "box", "half"))
    goal = np.asarray(entry["goal"], dtype=np.float32)
    start = time.perf_counter()
    if method == "hybrid":
        result = prepare_recovery(root, pose, box, half, goal)
        return result, {"prepare_sec": time.perf_counter() - start}
    generation_finished = None
    try:
        frames, phase = generate_recovery_clip(root, pose, box, half, goal)
        generation_finished = time.perf_counter()
        result = (frames, phase) if _valid_recovery_grasp(root, frames, half, phase) else None
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        # Same numerical/planning rejection as _v7_recovery_plan.
        result = None
    end = time.perf_counter()
    if generation_finished is None:
        generation_finished = end
    return result, {"generate_sec": generation_finished-start, "validate_sec": end-generation_finished}


def worker_init(root, warmup, ready):
    from carrywm.warp_sim import _init_cpu_worker
    _init_cpu_worker()
    # Warm both kinds of planning even when a worker receives only one timed method.
    for method in ("cpu", "hybrid"):
        plan_task(root, warmup, method)
    ready.put(os.getpid())


def ready_pid():
    return os.getpid()


def finish(fk, prepared):
    from carrywm.recovery_gpu import RecoverySkeleton
    skeletons = [p for p in prepared if isinstance(p, RecoverySkeleton)]
    fk.complete(skeletons)
    outputs = [(p.frames, p.phase) if isinstance(p, RecoverySkeleton) else p for p in prepared]
    return outputs, len(skeletons)


def run_group(pool, root, group, method, fk, device):
    import torch
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    jobs = [pool.submit(plan_task, root, sample, method) for sample in group]
    planned = [job.result() for job in jobs]  # Includes worker queue and return IPC for all frames.
    planning_done = time.perf_counter()
    plans = [p for p, _ in planned]
    user_times = [t for _, t in planned]
    skeletons = 0
    if method == "hybrid":
        plans, skeletons = finish(fk, plans)
        torch.cuda.synchronize(device)
    ended = time.perf_counter()
    stats = {
        "wall_sec": ended-started, "pool_and_ipc_wall_sec": planning_done-started,
        "worker_seconds_sum": {k:sum(t[k] for t in user_times) for k in user_times[0]},
        "finish_including_transfers_wall_sec": ended-planning_done if method == "hybrid" else 0.,
        "accepted": sum(p is not None for p in plans),
    }
    if method == "hybrid":
        stats.update(fk_completed=skeletons, cpu_fallback_accepted=stats["accepted"]-skeletons)
    return plans, stats


QUAT_START = (3, 10, 17, 24, 31, 100, 107)
QUAT_NAME = ("left_palm","right_palm","torso","left_ankle","right_ankle","root","box")


def compare_plans(ref, cand, atol, rtol):
    row = {"cpu_accepted": ref is not None, "hybrid_accepted": cand is not None}
    if ref is None or cand is None:
        row["ok"] = ref is None and cand is None
        return row, None, None
    x, phase = ref
    y, new_phase = cand
    row.update(cpu_shape=list(x.shape), hybrid_shape=list(y.shape), phase_equal=bool(np.array_equal(phase,new_phase)))
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 111 or not len(x):
        row["ok"] = False
        return row, None, None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        row.update(ok=False, finite=False)
        return row, None, None
    err = np.abs(x.astype(np.float64) - y)
    by_column = err.max(axis=0)
    bad = np.sum(~np.isclose(y,x,atol=atol,rtol=rtol),axis=0)
    signs = {}
    for name, k in zip(QUAT_NAME, QUAT_START):
        # Preserve exact hemisphere behavior consumed by the encoder, in addition
        # to comparing the signed wxyz components in the full 111-column arrays.
        dot = np.einsum("ij,ij->i", x[:,k:k+4], y[:,k:k+4])
        signs[name] = int(np.count_nonzero(dot < 0))
    row.update(frames=len(x), max_abs=float(np.max(by_column)), worst_column=int(np.argmax(by_column)),
               violating_columns=np.flatnonzero(bad).tolist(), opposite_quaternion_sign_frames=signs,
               ok=bool(row["phase_equal"] and not np.any(bad) and not any(signs.values())))
    return row, by_column, bad


def run(args):
    import torch
    from carrywm.recovery_gpu import BatchedRecoveryFK

    if min(args.workers,args.group_size)<1 or (args.limit is not None and args.limit<1):
        raise ValueError("Workers, group-size and limit must be positive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[k] = "1"
    torch.set_num_threads(1)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    root = args.sim_root.resolve()
    raw = args.inputs.read_bytes()
    input_data = json.loads(raw)
    samples = input_data["samples"][:args.limit]
    if not samples:
        raise ValueError("No captured drops")
    fk = BatchedRecoveryFK(root,dev)
    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    groups = []
    details = []
    err_max = np.zeros(111,np.float64)
    wrong_values = np.zeros(111,np.int64)
    all_compared = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=worker_init,
                                 initargs=(root,samples[0],ready)) as pool:
            # Workers are spawned on submit. Wait for every initializer before timing,
            # including when the first group has fewer examples than worker processes.
            launches = [pool.submit(ready_pid) for _ in range(args.workers)]
            for fut in launches:
                fut.result()
            pids = {ready.get(timeout=300) for _ in range(args.workers)}
            if len(pids)!=args.workers:
                raise RuntimeError("Incomplete recovery pool warmup")
            warmed = False
            for sample in samples:
                plan, _ = pool.submit(plan_task,root,sample,"hybrid").result()
                if plan is not None:
                    _, used = finish(fk,[plan])
                    if used:
                        warmed = True
                        break
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            for start in range(0,len(samples),args.group_size):
                group = samples[start:start+args.group_size]
                order = ("cpu","hybrid") if len(groups)%2==0 else ("hybrid","cpu")
                plans, timings = {}, {}
                for method in order:
                    plans[method],timings[method] = run_group(pool,root,group,method,fk,dev)
                for local,(x,y) in enumerate(zip(plans["cpu"],plans["hybrid"])):
                    meta,errs,violations = compare_plans(x,y,args.atol,args.rtol)
                    if errs is not None:
                        err_max = np.maximum(err_max,errs)
                        wrong_values += violations
                        all_compared += meta["frames"]
                    sample = group[local]
                    details.append(dict(index=start+local,tick=sample.get("t"),world=sample.get("w"),**meta))
                groups.append(dict(first_input=start,count=len(group),order=list(order),**timings))
                print(f"{start}-{start+len(group)-1}  CPU {timings['cpu']['wall_sec']:.3f}s  "
                      f"hybrid {timings['hybrid']['wall_sec']:.3f}s",flush=True)
    finally:
        ready.close()
        ready.join_thread()

    def count(cpu,hybrid):
        return sum(s["cpu_accepted"]==cpu and s["hybrid_accepted"]==hybrid for s in details)
    time_summary = {}
    for method in ("cpu","hybrid"):
        elapsed = np.asarray([g[method]["wall_sec"] for g in groups])
        time_summary[method] = {
            "total_wall_sec": float(elapsed.sum()), "median_group_wall_sec": float(np.median(elapsed)),
            "p95_group_wall_sec": float(np.percentile(elapsed,95)), "input_plans_per_sec": len(samples)/float(elapsed.sum()),
            "pool_and_ipc_wall_sec": sum(g[method]["pool_and_ipc_wall_sec"] for g in groups),
            "completion_and_transfers_wall_sec":sum(g[method]["finish_including_transfers_wall_sec"] for g in groups),
        }
    both, neither = count(True,True),count(False,False)
    report = dict(input=str(args.inputs),input_sha256=hashlib.sha256(raw).hexdigest(),captured_iteration=input_data.get("iteration"),
                  examples=len(samples),group_size=args.group_size,workers=args.workers,device=torch.cuda.get_device_name(dev),
                  torch_version=torch.__version__, numpy_version=np.__version__, atol=args.atol,rtol=args.rtol,
                  warmed_cuda_fk=warmed, all_checks_pass=all(s["ok"] for s in details),
                  decisions={"both_accepted":both,"both_rejected":neither,"only_cpu_accepted":count(True,False),
                             "only_hybrid_accepted":count(False,True)},
                  compared_frames=all_compared, all_111_max_abs_by_column=err_max.tolist(),
                  out_of_tolerance_values_by_column=wrong_values.tolist(),
                  opposite_quaternion_sign_frames={k:sum(s.get("opposite_quaternion_sign_frames",{}).get(k,0)
                                                         for s in details) for k in QUAT_NAME},
                  time=time_summary, wall_speedup=time_summary["cpu"]["total_wall_sec"]/time_summary["hybrid"]["total_wall_sec"],
                  peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(dev), groups=groups,samples=details,
                  timing_scope="Warmed shared CPU process pool. Each captured group is run through both methods, with alternating order. "
                  "Wall time includes submit, CPU generation or skeleton preparation and admission, full result IPC and "
                  "the hybrid owner's batched FK, H2D, D2H and host assembly. Grouped captures are not a live training throughput or overlap measurement. "
                  "CPU worker seconds are sums across concurrently active processes. Admission is reported only for the supplied input population.")
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("examples","decisions","compared_frames","all_checks_pass","wall_speedup")}),flush=True)
    return report


if __name__ == "__main__":
    report = run(arguments())
    raise SystemExit(0 if report["all_checks_pass"] else 1)
