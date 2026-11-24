#!/usr/bin/env python3
"""Generate offline Bench A + Bench B V7 CF-Gen training and validation packs."""
import argparse
import json
from pathlib import Path
import time

from carrywm.official_pool import build_official_pack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-a", required=True, type=int)
    parser.add_argument("--n-b", required=True, type=int)
    parser.add_argument("--seed-a", required=True, type=int)
    parser.add_argument("--seed-b", required=True, type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--compute-valid-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompute V7 admissible start masks and 5-stage feasibility stats",
    )
    parser.add_argument(
        "--require-valid-starts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resample clips that have zero admissible V7 training starts",
    )
    parser.add_argument("--prefix", default="official", help="Clip label prefix")
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    summary = build_official_pack(
        args.sim_root,
        args.output,
        n_a=args.n_a,
        n_b=args.n_b,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        workers=args.workers,
        compute_valid_mask=args.compute_valid_mask,
        require_valid_starts=args.require_valid_starts,
        prefix=args.prefix,
    )
    summary["wall_seconds"] = round(time.monotonic() - t0, 2)
    summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
