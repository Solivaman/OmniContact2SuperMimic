#!/usr/bin/env python3
"""Evaluate official OmniContact ONNX policies (MLP & Transformer) using V7 CF-Gen trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_official_carrybox as eoc
from official_cfgen_v72_adapter import verify_official_fk_consistency
from carrywm.geometry_v7 import _probe, _v7_type
from carrywm.recovery_fk_v7 import _recovery_type
from carrywm.sim import _matrix
from carrywm.train import _init_eval_worker


def _generate_v7_cfgen_raw(sim_root: Path, root7: np.ndarray, box7: np.ndarray, half: np.ndarray, goal: np.ndarray):
    """Run V7CarryBox generator and return its raw OmniContact-schema trajectory dict."""
    root = Path(sim_root).resolve()
    gen = _v7_type(root)()
    gen.cfg["ik_joint_indices"] = np.array([j for j in range(29) if j not in (12, 13)])
    gen.probe = _probe(root, tuple(float(x) for x in half))
    gen._left_palm_center, gen._right_palm_center = gen.probe.pads.astype(np.float32)
    gen.initial_box_rot = _matrix(box7[3:7])
    return gen.generate(
        np.asarray(root7[:3], dtype=np.float32),
        np.asarray(root7[3:7], dtype=np.float32),
        np.asarray(box7[:3], dtype=np.float32),
        np.asarray(box7[3:7], dtype=np.float32),
        box_half_dims=np.asarray(half, dtype=np.float32),
        target_obj_pos=np.asarray(goal, dtype=np.float32),
    )


def _generate_v7_replan_raw(sim_root: Path, root7: np.ndarray, box7: np.ndarray, half: np.ndarray, goal: np.ndarray):
    """Fast position-only FK V7 generator for online replanning, returning raw traj dict."""
    root = Path(sim_root).resolve()
    source = _recovery_type(root)
    if source is not None:
        gen = source()
        gen.cfg["ik_joint_indices"] = np.array([j for j in range(29) if j not in (12, 13)])
        gen.probe = _probe(root, tuple(float(x) for x in half))
        gen._left_palm_center, gen._right_palm_center = gen.probe.pads.astype(np.float32)
        gen.initial_box_rot = _matrix(box7[3:7])
        try:
            return gen.generate(
                np.asarray(root7[:3], dtype=np.float32),
                np.asarray(root7[3:7], dtype=np.float32),
                np.asarray(box7[:3], dtype=np.float32),
                np.asarray(box7[3:7], dtype=np.float32),
                box_half_dims=np.asarray(half, dtype=np.float32),
                target_obj_pos=np.asarray(goal, dtype=np.float32),
            )
        except KeyError:
            pass
    return _generate_v7_cfgen_raw(root, root7, box7, half, goal)


def _worker_eval_chunk(spec):
    sim_root_str, methods, episodes = spec
    sim_root = Path(sim_root_str)
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))

    # Bind replanning inside this isolated eval worker to return V7 raw traj dicts.
    eoc.generate_recovery_clip = _generate_v7_replan_raw

    consts = eoc._make_mj_env_consts(sim_root)
    runners = {}
    if "official_transformer_v7_cfgen" in methods:
        runners["official_transformer_v7_cfgen"] = eoc._OfficialOnnxAdapter(
            sim_root, sim_root / "policy/omnicontact/model/omnicontact_transformer.onnx", consts
        )
    if "official_mlp_v7_cfgen" in methods:
        runners["official_mlp_v7_cfgen"] = eoc._OfficialOnnxAdapter(
            sim_root, sim_root / "policy/omnicontact/model/omnicontact_mlp.onnx", consts
        )

    results = {m: [] for m in methods}
    fk_stats = []
    for ep in episodes:
        root7 = np.array(ep["root_pos"] + ep["root_quat"], dtype=np.float32)
        box7 = np.array(ep["init_pos"] + ep["init_quat"], dtype=np.float32)
        half = np.asarray(ep["half"], dtype=np.float32)
        goal = np.asarray(ep["goal_pos"], dtype=np.float32)

        v7_traj = None
        try:
            v7_traj, _ = _generate_v7_cfgen_raw(sim_root, root7, box7, half, goal)
            if len(fk_stats) < 8:
                fk_stats.append(verify_official_fk_consistency(sim_root, v7_traj))
        except Exception:
            v7_traj = None

        for m in methods:
            results[m].append(
                eoc._run_episode_with_runner(
                    sim_root, ep, consts, runners[m], "v7_cfgen", v7_traj, allow_replan=True
                )
            )
    return results, fk_stats


def _run_suite(pool, sim_root: Path, workers: int, episodes: list[dict], methods: tuple[str, ...]):
    n_workers = min(workers, len(episodes))
    chunks = [episodes[i::n_workers] for i in range(n_workers)]
    specs = [(str(sim_root), methods, chunk) for chunk in chunks]
    all_res = {m: [] for m in methods}
    all_fk = []
    for res, fk_s in pool.map(_worker_eval_chunk, specs):
        all_fk.extend(fk_s)
        for m, lst in res.items():
            all_res[m].extend(lst)
    for m in all_res:
        all_res[m].sort(key=lambda r: r["ep_id"])
    summary = {m: eoc._summarize_records(all_res[m]) for m in methods if all_res[m]}
    fk_summary = {
        "samples": len(all_fk),
        "mean_pos_err_m": float(np.mean([x["mean_pos_err_m"] for x in all_fk])) if all_fk else None,
        "max_pos_err_m": float(np.max([x["max_pos_err_m"] for x in all_fk])) if all_fk else None,
    }
    return {"summary": summary, "fk_consistency": fk_summary, "episodes": all_res}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-root", required=True, type=Path)
    p.add_argument("--motion-dir", type=Path, default=None)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bench-a-episodes", type=int, default=200)
    p.add_argument("--bench-b-episodes", type=int, default=100)
    p.add_argument("--methods", type=str, default="official_mlp_v7_cfgen,official_transformer_v7_cfgen")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.bench_a_episodes = 6
        args.bench_b_episodes = 0
        args.workers = min(args.workers, 4)

    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    eps_a = eoc.sample_official_episodes(args.bench_a_episodes, args.seed, randomize_box_size=True)
    eps_b = eoc.sample_official_episodes(args.bench_b_episodes, args.seed + 1000, randomize_box_size=False) if args.bench_b_episodes > 0 else []
    eps_c = eoc.load_official_demo_episodes(args.sim_root)
    eps_d = eoc.load_case1_test_episodes(args.motion_dir) if args.motion_dir is not None else []

    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    out = {
        "eval_type": "official_onnx_on_v7_cfgen",
        "methods": list(methods),
        "seed": args.seed,
    }
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_eval_worker) as pool:
        t_a = time.monotonic()
        print(f"=== Running Benchmark A (Official ONNX + V7 CF-Gen, {len(eps_a)} eps) ===", flush=True)
        res_a = _run_suite(pool, args.sim_root, args.workers, eps_a, methods)
        out["benchmark_a_official_random"] = res_a
        print(f"--- Benchmark A done in {time.monotonic() - t_a:.1f}s ---", flush=True)
        eoc._print_suite_summary("A", res_a["summary"], show_box_split=True)
        print(f"  V7 FK check: {res_a['fk_consistency']}", flush=True)

        if eps_b:
            t_b = time.monotonic()
            print(f"=== Running Benchmark B (Official ONNX + V7 CF-Gen, {len(eps_b)} eps) ===", flush=True)
            res_b = _run_suite(pool, args.sim_root, args.workers, eps_b, methods)
            out["benchmark_b_30cm_cube"] = res_b
            print(f"--- Benchmark B done in {time.monotonic() - t_b:.1f}s ---", flush=True)
            eoc._print_suite_summary("B", res_b["summary"])

        if eps_c and not args.smoke:
            t_c = time.monotonic()
            print(f"=== Running Benchmark C (Official ONNX + V7 CF-Gen, {len(eps_c)} eps) ===", flush=True)
            res_c = _run_suite(pool, args.sim_root, args.workers, eps_c, methods)
            out["benchmark_c_sim2sim_demos"] = res_c
            print(f"--- Benchmark C done in {time.monotonic() - t_c:.1f}s ---", flush=True)
            eoc._print_suite_summary("C", res_c["summary"])

        if eps_d:
            eps_d_run = eps_d[:6] if args.smoke else eps_d
            t_d = time.monotonic()
            print(f"=== Running Benchmark D (Official ONNX + V7 CF-Gen, {len(eps_d_run)} eps) ===", flush=True)
            res_d = _run_suite(pool, args.sim_root, args.workers, eps_d_run, methods)
            out["benchmark_d_fairness_subset"] = res_d
            print(f"--- Benchmark D done in {time.monotonic() - t_d:.1f}s ---", flush=True)
            eoc._print_suite_summary("D", res_d["summary"])

    out["wall_seconds"] = round(time.monotonic() - t0, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(eoc._clean_json(out), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(f"=== ALL OFFICIAL+V7-CFGEN BENCHMARKS SAVED ({out['wall_seconds']}s): {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
