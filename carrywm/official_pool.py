"""Offline sampling, planning, serialization and stats for official Carry Box CF-Gen packs."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import warnings

import numpy as np

from .data import RAW_DIM
from .geometry_v7 import generate_v7_clip, valid_start_mask
from .stages_v7 import STAGES, _clip_box_metadata, _stage_labels

DEFAULT_PELVIS_INIT_Z = 0.778
MIN_ROBOT_CLEARANCE = 0.65
DEFAULT_VAL_SEED_A = 20261
DEFAULT_VAL_SEED_B = 20262
DEFAULT_TRAIN_SEED_A = 73001
DEFAULT_TRAIN_SEED_B = 73002
RESERVED_SEEDS = frozenset({42, 1042, 5302})
TABLE_HALF_SIZES = np.array([[0.2, 0.2, 0.01], [0.2, 0.2, 0.01]], dtype=np.float64)


@dataclass
class OfficialPackClip:
    ep_id: int
    label: str
    bench_type: str
    frames: np.ndarray
    phase: np.ndarray
    half: np.ndarray
    tables: np.ndarray
    goal: np.ndarray
    valid_mask: np.ndarray | None = None


def geometry_v7_sha256() -> str:
    return hashlib.sha256(Path(__file__).with_name("geometry_v7.py").read_bytes()).hexdigest()


def sample_official_spec(rng, *, randomize_box_size: bool) -> dict:
    """Sample one official Carry Box episode spec matching Appendix sec:append_eval_protocol."""
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
    table1_pos = (init_pos - np.array([0.0, 0.0, half[2] + 0.01], dtype=np.float32)).astype(np.float64)
    table2_pos = (goal_pos - np.array([0.0, 0.0, half[2] + 0.01], dtype=np.float32)).astype(np.float64)
    return {
        "bench_type": "bench_a" if randomize_box_size else "bench_b",
        "half": half.astype(np.float64),
        "init_pos": init_pos,
        "init_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "goal_pos": goal_pos,
        "root_pos": np.array([0.0, 0.0, DEFAULT_PELVIS_INIT_Z], dtype=np.float32),
        "root_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "tables": np.stack((table1_pos, table2_pos), axis=0),
    }


def interleave_specs(specs_a, specs_b):
    """Deterministically interleave two sequences proportionally (exact 2:1 -> A, A, B)."""
    n_a, n_b = len(specs_a), len(specs_b)
    if not n_a:
        return list(specs_b)
    if not n_b:
        return list(specs_a)
    out = []
    ia = ib = 0
    while ia < n_a or ib < n_b:
        if ib >= n_b or (ia < n_a and (ia + 1) * n_b <= (ib + 1) * n_a):
            out.append(specs_a[ia])
            ia += 1
        else:
            out.append(specs_b[ib])
            ib += 1
    return out


def _validate_seeds(seed_a: int, seed_b: int) -> None:
    if seed_a == seed_b:
        raise ValueError(f"seed_a and seed_b must be distinct, got {seed_a}")
    for name, seed in (("seed_a", seed_a), ("seed_b", seed_b)):
        if seed in RESERVED_SEEDS:
            raise ValueError(f"{name}={seed} conflicts with reserved benchmark/split seeds {sorted(RESERVED_SEEDS)}")


def sample_mixed_official_specs(
    n_a: int,
    n_b: int,
    seed_a: int,
    seed_b: int,
    *,
    prefix: str = "official",
) -> list[dict]:
    """Sample Bench A and Bench B specs with independent RNG streams and interleave them."""
    if n_a < 0 or n_b < 0 or (n_a == 0 and n_b == 0):
        raise ValueError("n_a and n_b must be nonnegative with n_a + n_b > 0")
    _validate_seeds(int(seed_a), int(seed_b))
    rng_a = np.random.default_rng(int(seed_a))
    rng_b = np.random.default_rng(int(seed_b))
    specs_a = []
    for idx in range(n_a):
        spec = sample_official_spec(rng_a, randomize_box_size=True)
        spec["label"] = f"{prefix}_a_{idx:04d}"
        specs_a.append(spec)
    specs_b = []
    for idx in range(n_b):
        spec = sample_official_spec(rng_b, randomize_box_size=False)
        spec["label"] = f"{prefix}_b_{idx:04d}"
        specs_b.append(spec)
    mixed = interleave_specs(specs_a, specs_b)
    for ep_id, spec in enumerate(mixed):
        spec["ep_id"] = ep_id
    return mixed


def _init_cpu_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _spec_geometry_stats(spec: dict, num_frames: int) -> dict:
    root_pos = np.asarray(spec["root_pos"], dtype=np.float64)
    init_pos = np.asarray(spec["init_pos"], dtype=np.float64)
    goal_pos = np.asarray(spec["goal_pos"], dtype=np.float64)
    half = np.asarray(spec["half"], dtype=np.float64)
    rq = np.asarray(spec["root_quat"], dtype=np.float64)
    ryaw = math.atan2(2.0 * (rq[0] * rq[3] + rq[1] * rq[2]), 1.0 - 2.0 * (rq[2] * rq[2] + rq[3] * rq[3]))
    byaw = math.atan2(init_pos[1] - root_pos[1], init_pos[0] - root_pos[0])
    heading_deg = abs(math.degrees(math.atan2(math.sin(byaw - ryaw), math.cos(byaw - ryaw))))
    return {
        "max_box_cm": float(200.0 * np.max(half)),
        "init_dist_m": float(math.hypot(init_pos[0] - root_pos[0], init_pos[1] - root_pos[1])),
        "init_heading_deg": float(heading_deg),
        "travel_dist_m": float(math.hypot(goal_pos[0] - init_pos[0], goal_pos[1] - init_pos[1])),
        "num_frames": int(num_frames),
    }


def _plan_one_spec_worker(task: tuple[str, dict, bool]) -> dict:
    sim_root_str, spec, compute_valid_mask = task
    sim_root = Path(sim_root_str)
    root7 = np.concatenate((spec["root_pos"], spec["root_quat"])).astype(np.float32)
    box7 = np.concatenate((spec["init_pos"], spec["init_quat"])).astype(np.float32)
    half = np.asarray(spec["half"], dtype=np.float64)
    goal = np.asarray(spec["goal_pos"], dtype=np.float32)
    tables = np.asarray(spec["tables"], dtype=np.float64)

    try:
        frames, phase = generate_v7_clip(sim_root, root7, box7, half, goal)
        frames = np.ascontiguousarray(frames, dtype=np.float32)
        phase = np.ascontiguousarray(phase, dtype=np.int32)
        if frames.ndim != 2 or frames.shape[1] != RAW_DIM or len(frames) < 2 or not np.isfinite(frames).all():
            return {"ok": False, "error": f"Invalid generated clip shape {frames.shape}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    valid_mask = None
    has_any_valid = True
    stage_counts = None
    if compute_valid_mask:
        valid_mask = np.ascontiguousarray(
            valid_start_mask(sim_root, frames, half, "cfgen", phase),
            dtype=bool,
        )
        support_valid, _, _ = _clip_box_metadata(frames, half, tables, TABLE_HALF_SIZES)
        labels, final_grasp = _stage_labels(frames)
        eligible = valid_mask & support_valid
        eligible[min(final_grasp + 1, max(0, len(frames) - 8)):] = False
        has_any_valid = bool(eligible.any())
        stage_counts = {
            name: int((eligible & (labels == idx)).sum())
            for idx, name in enumerate(STAGES)
        }

    return {
        "ok": True,
        "spec": spec,
        "frames": frames,
        "phase": phase,
        "valid_mask": valid_mask,
        "has_any_valid": has_any_valid,
        "stage_counts": stage_counts,
        "stats": _spec_geometry_stats(spec, len(frames)),
    }


def _quantile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 4),
        "p25": round(float(np.quantile(arr, 0.25)), 4),
        "p50": round(float(np.quantile(arr, 0.50)), 4),
        "p75": round(float(np.quantile(arr, 0.75)), 4),
        "p95": round(float(np.quantile(arr, 0.95)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def _feasibility_summary(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"total": 0}
    stages = {}
    for name in STAGES:
        counts = [item["stage_counts"][name] for item in items]
        stages[name] = {
            "clips_with_stage": int(sum(1 for c in counts if c > 0)),
            "clips_with_stage_rate": round(float(sum(1 for c in counts if c > 0) / n), 4),
            "mean_eligible_frames": round(float(np.mean(counts)), 2),
        }
    return {
        "total": n,
        "has_any_valid": int(sum(1 for item in items if item["has_any_valid"])),
        "has_any_valid_rate": round(float(sum(1 for item in items if item["has_any_valid"]) / n), 4),
        "stages": stages,
    }


def save_official_pack(path: Path, clips: list[OfficialPackClip], metadata: dict) -> Path:
    """Atomically write an official V7 CF-Gen pack as an .npz archive and companion .json manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta["count"] = len(clips)
    payload = {
        "meta_json": np.array(json.dumps(meta, ensure_ascii=False), dtype=np.str_),
        "count": np.array([len(clips)], dtype=np.int64),
    }
    for idx, clip in enumerate(clips):
        payload[f"f_{idx}"] = np.ascontiguousarray(clip.frames, dtype=np.float32)
        payload[f"p_{idx}"] = np.ascontiguousarray(clip.phase, dtype=np.int32)
        payload[f"half_{idx}"] = np.ascontiguousarray(clip.half, dtype=np.float64)
        payload[f"tables_{idx}"] = np.ascontiguousarray(clip.tables, dtype=np.float64)
        payload[f"goal_{idx}"] = np.ascontiguousarray(clip.goal, dtype=np.float32)
        payload[f"label_{idx}"] = np.array(clip.label, dtype=np.str_)
        payload[f"bench_{idx}"] = np.array(clip.bench_type, dtype=np.str_)
        if clip.valid_mask is not None:
            payload[f"valid_{idx}"] = np.ascontiguousarray(clip.valid_mask, dtype=bool)

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    with tmp.open("wb") as f:
        np.savez(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

    manifest_path = path.with_suffix(".json")
    tmp_json = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp.json")
    tmp_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_json, manifest_path)
    return path


def load_official_pack(path: Path) -> tuple[tuple[OfficialPackClip, ...], dict]:
    """Load and validate an official V7 CF-Gen pack generated by save_official_pack."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"].item()))
        count = int(data["count"][0])
        has_valid = any(f"valid_{idx}" in data for idx in range(count))
        stale_valid = False
        if has_valid and meta.get("geometry_v7_sha256") != geometry_v7_sha256():
            stale_valid = True
            warnings.warn(
                f"Ignoring precomputed valid_mask in {path} because geometry_v7.py changed; "
                "recomputing admissible starts dynamically.",
                RuntimeWarning,
            )
        clips = []
        for idx in range(count):
            frames = np.ascontiguousarray(data[f"f_{idx}"], dtype=np.float32)
            phase = np.ascontiguousarray(data[f"p_{idx}"], dtype=np.int32)
            half = np.ascontiguousarray(data[f"half_{idx}"], dtype=np.float64)
            tables = np.ascontiguousarray(data[f"tables_{idx}"], dtype=np.float64)
            goal = np.ascontiguousarray(data[f"goal_{idx}"], dtype=np.float32)
            label = str(data[f"label_{idx}"].item())
            bench_type = str(data[f"bench_{idx}"].item())
            if frames.ndim != 2 or frames.shape[1] != RAW_DIM or len(frames) < 2 or not np.isfinite(frames).all():
                raise ValueError(f"Invalid frames in pack {path} index {idx}: {frames.shape}")
            if phase.shape != (len(frames),):
                raise ValueError(f"Invalid phase shape in pack {path} index {idx}: {phase.shape}")
            if half.shape != (3,) or np.any(half <= 0) or not np.isfinite(half).all():
                raise ValueError(f"Invalid half dimensions in pack {path} index {idx}: {half}")
            if tables.shape != (2, 3) or not np.isfinite(tables).all():
                raise ValueError(f"Invalid tables in pack {path} index {idx}: {tables.shape}")
            if goal.shape != (3,) or not np.isfinite(goal).all():
                raise ValueError(f"Invalid goal in pack {path} index {idx}: {goal}")
            valid_mask = None
            if not stale_valid and f"valid_{idx}" in data:
                valid_mask = np.ascontiguousarray(data[f"valid_{idx}"], dtype=bool)
                if valid_mask.shape != (len(frames),):
                    raise ValueError(f"Invalid valid_mask shape in pack {path} index {idx}: {valid_mask.shape}")
            clips.append(
                OfficialPackClip(
                    ep_id=idx,
                    label=label,
                    bench_type=bench_type,
                    frames=frames,
                    phase=phase,
                    half=half,
                    tables=tables,
                    goal=goal,
                    valid_mask=valid_mask,
                )
            )
    return tuple(clips), meta


def build_official_pack(
    sim_root: Path,
    output_path: Path,
    *,
    n_a: int,
    n_b: int,
    seed_a: int,
    seed_b: int,
    workers: int = 12,
    compute_valid_mask: bool = False,
    require_valid_starts: bool = False,
    prefix: str = "official",
) -> dict:
    """Sample, plan, verify and save a mixed Bench A + Bench B V7 CF-Gen pack."""
    if n_a < 0 or n_b < 0 or (n_a == 0 and n_b == 0):
        raise ValueError("n_a and n_b must be nonnegative with n_a + n_b > 0")
    if workers < 1:
        raise ValueError("workers must be positive")
    _validate_seeds(int(seed_a), int(seed_b))
    sim_root = Path(sim_root).resolve()
    output_path = Path(output_path)

    rng_a = np.random.default_rng(int(seed_a))
    rng_b = np.random.default_rng(int(seed_b))

    def next_spec(bench_type: str, counter: int) -> dict:
        if bench_type == "bench_a":
            spec = sample_official_spec(rng_a, randomize_box_size=True)
            spec["label"] = f"{prefix}_a_{counter:04d}"
        else:
            spec = sample_official_spec(rng_b, randomize_box_size=False)
            spec["label"] = f"{prefix}_b_{counter:04d}"
        return spec

    ok_a: list[dict] = []
    ok_b: list[dict] = []
    replaced_a = 0
    replaced_b = 0
    plan_errors_a = 0
    plan_errors_b = 0
    drawn_a = 0
    drawn_b = 0
    max_attempts = max(5 * (n_a + n_b) + 50, 100)
    total_attempts = 0
    need_mask = bool(compute_valid_mask or require_valid_starts)

    pool = None
    if workers > 1 and (n_a + n_b) > 1:
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=min(workers, n_a + n_b),
            mp_context=ctx,
            initializer=_init_cpu_worker,
        )

    def run_batch(pending_specs: list[dict]) -> list[dict]:
        tasks = [(str(sim_root), spec, need_mask) for spec in pending_specs]
        if pool is not None and len(tasks) > 1:
            return list(pool.map(_plan_one_spec_worker, tasks))
        return [_plan_one_spec_worker(t) for t in tasks]

    try:
        while len(ok_a) < n_a or len(ok_b) < n_b:
            rem_a = n_a - len(ok_a)
            rem_b = n_b - len(ok_b)
            if total_attempts + rem_a + rem_b > max_attempts:
                raise RuntimeError(
                    f"Exceeded replacement budget ({max_attempts}) while generating pack "
                    f"(ok_a={len(ok_a)}/{n_a}, ok_b={len(ok_b)}/{n_b}, "
                    f"replaced_a={replaced_a}, replaced_b={replaced_b})"
                )
            batch = []
            for _ in range(rem_a):
                batch.append(next_spec("bench_a", drawn_a))
                drawn_a += 1
            for _ in range(rem_b):
                batch.append(next_spec("bench_b", drawn_b))
                drawn_b += 1
            total_attempts += len(batch)

            for spec, res in zip(batch, run_batch(batch)):
                bench = spec["bench_type"]
                accepted = res["ok"] and (not require_valid_starts or res["has_any_valid"])
                if accepted:
                    if bench == "bench_a":
                        ok_a.append(res)
                    else:
                        ok_b.append(res)
                else:
                    if bench == "bench_a":
                        replaced_a += 1
                        if not res["ok"]:
                            plan_errors_a += 1
                    else:
                        replaced_b += 1
                        if not res["ok"]:
                            plan_errors_b += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    ordered = interleave_specs(ok_a, ok_b)
    clips: list[OfficialPackClip] = []
    for ep_id, item in enumerate(ordered):
        spec = item["spec"]
        clips.append(
            OfficialPackClip(
                ep_id=ep_id,
                label=spec["label"],
                bench_type=spec["bench_type"],
                frames=item["frames"],
                phase=item["phase"],
                half=np.asarray(spec["half"], dtype=np.float64),
                tables=np.asarray(spec["tables"], dtype=np.float64),
                goal=np.asarray(spec["goal_pos"], dtype=np.float32),
                valid_mask=item["valid_mask"] if compute_valid_mask else None,
            )
        )

    all_stats = [item["stats"] for item in ordered]
    metadata = {
        "prefix": prefix,
        "seed_a": int(seed_a),
        "seed_b": int(seed_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "count": len(clips),
        "replaced_a": int(replaced_a),
        "replaced_b": int(replaced_b),
        "plan_errors_a": int(plan_errors_a),
        "plan_errors_b": int(plan_errors_b),
        "compute_valid_mask": bool(compute_valid_mask),
        "require_valid_starts": bool(require_valid_starts),
        "geometry_v7_sha256": geometry_v7_sha256(),
        "stats": {
            key: _quantile_summary([s[key] for s in all_stats])
            for key in ("max_box_cm", "init_dist_m", "init_heading_deg", "travel_dist_m", "num_frames")
        },
    }
    if need_mask:
        le_35 = [it for it in ordered if it["stats"]["max_box_cm"] <= 35.0]
        gt_35 = [it for it in ordered if it["stats"]["max_box_cm"] > 35.0]
        metadata["feasibility"] = {
            "overall": _feasibility_summary(ordered),
            "bench_a": _feasibility_summary([it for it in ordered if it["spec"]["bench_type"] == "bench_a"]),
            "bench_b": _feasibility_summary([it for it in ordered if it["spec"]["bench_type"] == "bench_b"]),
            "le_35cm": _feasibility_summary(le_35),
            "gt_35cm": _feasibility_summary(gt_35),
        }

    save_official_pack(output_path, clips, metadata)
    return metadata
