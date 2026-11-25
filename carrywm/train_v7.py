"""Learn carry-box control from simulated transitions and motion references."""
import argparse
import math
import os
import sys

from . import train as base
from . import train_v4, train_v6
from .stages_v7 import ReturnSamplerCfg
from .v7 import LearnerV7


ULTRA_PRESET = {
    "optimizer": "muon",
    "actor_beta2": 0.95,
    "actor_grad_clip_value": 0.1,
    "reset_sampling": "return",
    "box_mass_scale_range": (0.75, 1.25),
    "box_friction_range": (0.5, 1.0),
}


def configure(parser):
    train_v6.configure(parser)
    parser.description = __doc__
    cpus = os.cpu_count() or 2
    parser.add_argument("--ultra", action="store_true",
                        help="Enable V7.3-Ultra SuperMimic/OmniContact preset (Muon hidden + Adam ends, "
                             "Actor beta2=0.95 + grad value clip 0.1, ReturnSampler, light box mass/friction DR)")
    parser.add_argument("--reset-sampling", choices=("failure", "return"), default="failure",
                        help="V7 reset curriculum: binary failure-rate bins or SuperMimic discounted-return bins")
    parser.add_argument("--return-sampler-horizon", type=int, default=20,
                        help="Buffer ticks between ReturnSampler V updates (SuperMimic: 20)")
    parser.add_argument("--return-sampler-memory", type=int, default=256,
                        help="Window cap for per-bin return averaging (SuperMimic: 256)")
    parser.add_argument("--return-sampler-steepness", type=float, default=25.0,
                        help="Softmax gain on -steepness * V / v_max (SuperMimic: 25.0)")
    parser.add_argument("--return-sampler-explore", type=float, default=0.5,
                        help="Uniform bin exploration probability share (SuperMimic: 0.5)")
    parser.add_argument("--return-sampler-keep", type=float, default=0.1,
                        help="Historical focus squared-effort retention share (SuperMimic: 0.1)")
    parser.add_argument("--clip-storage", choices=("hybrid", "device"), default="device",
                        help="Store all references in growable GPU pages (device), or store recovery frames in host RAM (hybrid)")
    parser.add_argument("--clip-budget-gib", type=float, default=11.0,
                        help="GPU reference page and index budget in GiB; allocated as references arrive")
    parser.add_argument("--warp-headroom-mib", type=int, default=1536,
                        help="Release unused PyTorch cache before collection when driver free memory falls below this size")
    parser.add_argument("--gpu-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Batch online recovery reference kinematics on the simulation GPU")
    parser.add_argument("--defer-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Defer unready online recovery plans across ticks without blocking simulation resets")
    parser.add_argument("--recovery-workers", type=int, default=None,
                        help="CPU worker processes for online carry recovery IK")
    parser.add_argument("--fused-rotations", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse small rotation products in compiled V7 tracking geometry")
    parser.set_defaults(num_envs=2048, batch_size=2048, actor_microbatch=2048,
                        teacher_init_fraction=0., teacher_decay_iters=0, teacher_bc_weight=0.,
                        capacity=2_048_000, warmup_steps=16_384, collect_steps=3,
                        precision="bf16", iterations=2000,
                        eval_workers=min(12, max(1, cpus // 2)),
                        save_every=50,
                        full_save_every=200)


def _explicit_flags(argv):
    tokens = sys.argv[1:] if argv is None else list(argv)
    return {tok.split("=", 1)[0] for tok in tokens if tok.startswith("--")}


def args_from_cli(argv=None):
    args = train_v4.args_from_cli(argv, configure_parser=configure,
                                  world_count=2048)
    if args.ultra:
        explicit = _explicit_flags(argv)
        flag_map = {
            "optimizer": "--optimizer",
            "actor_beta2": "--actor-beta2",
            "actor_grad_clip_value": "--actor-grad-clip-value",
            "reset_sampling": "--reset-sampling",
            "box_mass_scale_range": "--box-mass-scale-range",
            "box_friction_range": "--box-friction-range",
        }
        for key, value in ULTRA_PRESET.items():
            if flag_map[key] not in explicit:
                setattr(args, key, value)
    from .sim import validate_box_dr
    args.box_mass_scale_range, args.box_friction_range = validate_box_dr(
        args.box_mass_scale_range, args.box_friction_range
    )
    ReturnSamplerCfg(
        gamma=base.GAMMA,
        horizon=args.return_sampler_horizon,
        memory=args.return_sampler_memory,
        steepness=args.return_sampler_steepness,
        explore=args.return_sampler_explore,
        keep=args.return_sampler_keep,
    )
    if args.teacher_init_fraction or args.teacher_bc_weight or args.teacher_decay_iters:
        raise ValueError("V7 uses its own actor from initialization and throughout collection and learning")
    if args.goal_mode != "full":
        raise ValueError("V7 requires full references with future box poses")
    if not math.isfinite(args.clip_budget_gib) or args.clip_budget_gib <= 0:
        raise ValueError("clip-budget-gib must be positive and finite")
    if args.warp_headroom_mib < 0:
        raise ValueError("warp-headroom-mib must be nonnegative")
    if args.recovery_workers is not None and args.recovery_workers < 1:
        raise ValueError("recovery-workers must be positive")
    return train_v6.configure_objective(args)


def train(args):
    train_v4.train(args, version=7, learner_type=LearnerV7)


if __name__ == "__main__":
    train(args_from_cli())
