#!/usr/bin/env python3
"""Compare paged, hybrid and contiguous V7 reference gather on actual replay-ring rows."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from carrywm.data import OFFSETS
from carrywm.replay_v4 import HybridClipBank


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument("--page-frames", type=int, default=65536)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--paged-budget-gib", type=float, default=11.,
                         help="Retained reference pages and indexes budget; V7 default is 11 GiB")
    budgets.add_argument("--unlimited-paged-budget", dest="paged_budget_gib", action="store_const", const=None,
                         help="Remove the reference budget for an uncapped comparison")
    parser.add_argument("--warp-headroom-mib", type=int, default=1536,
                        help="Free driver memory to retain when adding a new GPU page")
    parser.add_argument("--controlled-dynamic", type=float, help="Also sample a fixed fraction of recovery rows, e.g. 0.1")
    parser.add_argument("--backends", default="hybrid,paged,contiguous")
    return parser.parse_args()


def load_cpu_replay(path):
    checkpoint = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    return checkpoint["replay"], checkpoint.get("iteration", checkpoint.get("iter"))


def checksum(tensor):
    # Raw bytes distinguish quaternion sign, signed zero and every float32 bit.
    raw = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def prepare_queries(replay, host, args):
    count = min(replay.rows, replay.ticks)
    if not count:
        raise ValueError("checkpoint contains no populated replay rows")
    slots = replay.clip_ids[:count].detach().numpy().reshape(-1)
    frames = replay.frames[:count].detach().numpy().reshape(-1)
    lengths = np.asarray(host._slot_lengths)
    if slots.min() < 0 or slots.max() >= len(lengths) or np.any(lengths[slots] <= 0):
        raise ValueError("a populated replay row refers to a missing clip")
    dyn = ~np.asarray(host._slot_static, dtype=bool)[slots]
    rng = np.random.default_rng(args.seed)
    modes = ["active_ring"]
    if args.controlled_dynamic is not None:
        modes.append("controlled_dynamic")
        dynamic_rows, static_rows = np.flatnonzero(dyn), np.flatnonzero(~dyn)
    queries = []
    offsets = (torch.tensor([1]), torch.as_tensor(OFFSETS), torch.arange(83))
    total = args.reps + args.warmup
    for mode in modes:
        for repeat in range(total):
            if mode == "active_ring":
                ix = rng.integers(len(slots), size=args.batch)
            else:
                n_dyn = round(args.batch * args.controlled_dynamic)
                if (n_dyn and not len(dynamic_rows)) or (args.batch>n_dyn and not len(static_rows)):
                    raise ValueError("Requested dynamic mix is absent from the populated replay ring")
                ix = np.concatenate([rng.choice(rows,size=n,replace=True) for rows,n in
                                     ((dynamic_rows,n_dyn),(static_rows,args.batch-n_dyn)) if n])
                rng.shuffle(ix)
            # Each backend and all three horizons use the identical real slot/frame pairs.
            ids = torch.from_numpy(slots[ix].copy())
            times = torch.from_numpy(frames[ix].copy())
            expected = [checksum(host.gather(ids,times,off)) for off in offsets]
            queries.append((mode, repeat, ids, times, int(dyn[ix].sum()), expected, checksum(host.remaining(ids,times))))
    return queries, offsets, {"populated_transitions": len(slots), "unique_active_slots": int(len(np.unique(slots))),
                              "ring_dynamic_fraction": float(dyn.mean())}


def run_backend(name, device, args, queries, offsets):
    from carrywm.device_clips import PagedClipBank
    replay, _ = load_cpu_replay(args.checkpoint)
    cpu = replay.clip_bank
    if isinstance(cpu, HybridClipBank):
        cpu = cpu.host
    del replay
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    if name == "hybrid":
        bank = HybridClipBank(cpu,device)
    elif name == "contiguous":
        # This independent, mmap-backed deserialization has no other clients.
        bank = cpu.to(device)
    else:
        bank = PagedClipBank.from_host(cpu,device,page_frames=args.page_frames,
                                      max_device_bytes=args.paged_budget_bytes,
                                      reserve_device_bytes=args.headroom_bytes)
    torch.cuda.synchronize(device)
    build_sec = time.perf_counter()-start
    info = {
        "build_sec":build_sec, "build_peak_allocated_bytes":torch.cuda.max_memory_allocated(device)-base_alloc,
        "build_peak_reserved_bytes":torch.cuda.max_memory_reserved(device)-base_reserved,
        "resident_allocated_bytes":torch.cuda.memory_allocated(device)-base_alloc,
        "resident_reserved_bytes":torch.cuda.memory_reserved(device)-base_reserved, "storage":bank.storage_metrics(),
        "cases":[], "exact":True,
    }
    spill_mask = None
    if name == "paged":
        metrics = info["storage"]
        info["paged_placement"] = {
            "budget_bytes":args.paged_budget_bytes, "warp_headroom_bytes":args.headroom_bytes,
            "accounted_device_bytes":metrics["replay/clip_accounted_device_bytes"],
            "spilled_slots":metrics["replay/clip_spill_slots"], "spilled_frames":metrics["replay/clip_spill_frames"],
            "spilled_bytes":metrics["replay/clip_spill_bytes"], "resident_pages":metrics["replay/clip_pages"],
        }
        spill_mask = np.zeros(len(cpu._slot_lengths), dtype=bool)
        if bank._spill:
            spill_mask[list(bank._spill)] = True
    d_offsets = [o.to(device) for o in offsets]
    for mode in dict.fromkeys(q[0] for q in queries):
        for j, off in enumerate(d_offsets):
            elapsed, remaining_elapsed, dynamic = [], [], []
            spilled_queries = 0
            exact = True
            ref_base = torch.cuda.memory_allocated(device)
            ref_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            for which,rep,ids,frames,dynamic_count,expected,remaining in queries:
                if which != mode:
                    continue
                gpu_ids, gpu_frames = ids.to(device), frames.to(device)
                # Production already holds the query on the device. Hybrid's internal
                # host round trips and its recovery-row uploads stay inside this timer.
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                raw = bank.gather(gpu_ids,gpu_frames,off)
                torch.cuda.synchronize(device)
                gather_ms = (time.perf_counter_ns()-start)/1e6
                if checksum(raw) != expected[j]:
                    exact = False
                del raw
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                rest = bank.remaining(gpu_ids,gpu_frames)
                torch.cuda.synchronize(device)
                rest_ms = (time.perf_counter_ns()-start)/1e6
                if checksum(rest) != remaining:
                    exact = False
                del rest, gpu_ids, gpu_frames
                if rep >= args.warmup:
                    elapsed.append(gather_ms)
                    remaining_elapsed.append(rest_ms)
                    dynamic.append(dynamic_count)
                    if spill_mask is not None:
                        spilled_queries += int(spill_mask[ids.numpy()].sum())
            info["cases"].append({"sampling":mode,"offset_count":len(off),"all_signed_bits_equal":exact,
                                  "batch_dynamic_fraction":sum(dynamic)/(len(dynamic)*args.batch),
                                  "gather_ms":elapsed,"gather_p50_ms":float(np.median(elapsed)),
                                  "gather_p95_ms":float(np.percentile(elapsed,95)),
                                  "remaining_p50_ms":float(np.median(remaining_elapsed)),
                                  "gather_peak_extra_allocated_bytes":torch.cuda.max_memory_allocated(device)-ref_base,
                                  "gather_peak_extra_reserved_bytes":torch.cuda.max_memory_reserved(device)-ref_reserved})
            if spill_mask is not None:
                info["cases"][-1]["host_spill_query_fraction"] = spilled_queries / (len(dynamic) * args.batch)
            info["exact"] &= exact
    return info


def main(args):
    if args.batch<1 or args.reps<1 or args.warmup<0 or args.page_frames<1:
        raise ValueError("invalid batch, page size or repetition count")
    if args.warp_headroom_mib < 0 or (args.paged_budget_gib is not None and
                                     (not np.isfinite(args.paged_budget_gib) or args.paged_budget_gib < 0)):
        raise ValueError("reference budget and Warp headroom must be nonnegative")
    args.paged_budget_bytes = None if args.paged_budget_gib is None else round(args.paged_budget_gib * 1024**3)
    args.headroom_bytes = args.warp_headroom_mib * 1024**2
    if args.controlled_dynamic is not None and not 0<=args.controlled_dynamic<=1:
        raise ValueError("controlled dynamic fraction must be between 0 and 1")
    backends = args.backends.split(",")
    if not backends or len(set(backends))!=len(backends) or set(backends)-{"paged","hybrid","contiguous"}:
        raise ValueError("choose unique backends from paged,hybrid,contiguous")
    device = torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    replay, iteration = load_cpu_replay(args.checkpoint)
    host = replay.clip_bank
    if isinstance(host, HybridClipBank):
        host = host.host
    queries, offsets, ring = prepare_queries(replay, host, args)
    report = {"checkpoint":str(args.checkpoint),"iteration":iteration,"seed":args.seed,"batch":args.batch,
              "warmup":args.warmup,"reps":args.reps,"page_frames":args.page_frames,"ring":ring,
              "host_used_frames":host._used_frames,"host_static_slots":sum(host._slot_static),
              "checkpoint_size_bytes":args.checkpoint.stat().st_size, "controlled_dynamic":args.controlled_dynamic,
              "paged_budget_gib":args.paged_budget_gib,"warp_headroom_mib":args.warp_headroom_mib,
              "device":torch.cuda.get_device_name(device),"device_total_bytes":torch.cuda.get_device_properties(device).total_memory,
              "torch_version":torch.__version__,"backends":{},
              "scope":"Independent checkpoint mmap per backend. Only one reference bank and one output batch live on GPU at once. "
              "Queries sampled with replacement from populated real ring entries. Time includes benchmarked gather and all its "
              "host/device transfers, with device synchronization at both boundaries. Correctness compared outside timed sections "
              "against SHA-256 of all CPU-reference bytes. Only the paged backend uses the configured budget and driver "
              "headroom. The independent benchmark measures clip storage without allocating the training simulator or models. "
              "CUDA allocator statistics count PyTorch allocations in this process."}
    del replay, host
    for name in backends:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            row = run_backend(name, device, args, queries, offsets)
        except torch.cuda.OutOfMemoryError as exc:
            row = {"exact":False,"oom":str(exc),"at_failure_peak_allocated_bytes":torch.cuda.max_memory_allocated(device)}
        report["backends"][name] = row
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(name, [(r["sampling"],r["offset_count"],round(r["gather_p50_ms"],3)) for r in row.get("cases",[])],
              "exact",row["exact"],flush=True)
    return all(r["exact"] for r in report["backends"].values())


if __name__ == "__main__":
    raise SystemExit(0 if main(parse_args()) else 1)
