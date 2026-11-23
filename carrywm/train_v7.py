"""Learn carry-box control from simulated transitions and motion references."""
import argparse
import math

from . import train_v4, train_v6
from .v7 import LearnerV7


def configure(parser):
    train_v6.configure(parser)
    parser.description = __doc__
    parser.add_argument("--clip-storage", choices=("hybrid", "device"), default="device",
                        help="Store all references in growable GPU pages (device), or store recovery frames in host RAM (hybrid)")
    parser.add_argument("--clip-budget-gib", type=float, default=11.0,
                        help="GPU reference page and index budget in GiB; allocated as references arrive")
    parser.add_argument("--warp-headroom-mib", type=int, default=1536,
                        help="Release unused PyTorch cache before collection when driver free memory falls below this size")
    parser.add_argument("--gpu-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Batch online recovery reference kinematics on the simulation GPU")
    parser.add_argument("--fused-rotations", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse small rotation products in compiled V7 tracking geometry")
    parser.set_defaults(num_envs=2048, batch_size=2048, actor_microbatch=2048,
                        teacher_init_fraction=0., teacher_decay_iters=0, teacher_bc_weight=0.,
                        capacity=2_048_000, warmup_steps=16_384, collect_steps=3,
                        precision="bf16", iterations=2000)


def args_from_cli(argv=None):
    args = train_v4.args_from_cli(argv, configure_parser=configure,
                                  world_count=2048)
    if args.teacher_init_fraction or args.teacher_bc_weight or args.teacher_decay_iters:
        raise ValueError("V7 uses its own actor from initialization and throughout collection and learning")
    if args.goal_mode != "full":
        raise ValueError("V7 requires full references with future box poses")
    if not math.isfinite(args.clip_budget_gib) or args.clip_budget_gib <= 0:
        raise ValueError("clip-budget-gib must be positive and finite")
    if args.warp_headroom_mib < 0:
        raise ValueError("warp-headroom-mib must be nonnegative")
    return train_v6.configure_objective(args)


def train(args):
    train_v4.train(args, version=7, learner_type=LearnerV7)


if __name__ == "__main__":
    train(args_from_cli())
