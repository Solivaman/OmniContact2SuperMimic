"""Train the V6 carry-box policy with relative fall termination, balanced rewards, and early teacher guidance."""
import argparse
import math

from . import train_v4
from . import train_v5
from .v6 import (
    ACTION_L2_WEIGHT,
    ACTION_RATE_WEIGHT,
    BALANCE_REWARD_WEIGHT,
    GRAVITY_Z_THRESHOLD,
    HEIGHT_THRESHOLD_M,
    OBS_NOISE_SCALE,
    TEACHER_BC_BATCH,
    TEACHER_BC_WEIGHT,
    TEACHER_DECAY_ITERS,
    TEACHER_INIT_FRACTION,
    TERMINATION_SOFTNESS,
    LearnerV6,
    v6_tracking_weights,
    v6_value_max,
)


def configure(parser):
    train_v5.configure(parser)
    parser.description = __doc__
    parser.add_argument("--balance-reward-weight", type=float, default=BALANCE_REWARD_WEIGHT)
    parser.add_argument("--action-rate-weight", type=float, default=ACTION_RATE_WEIGHT)
    parser.add_argument("--action-l2-weight", type=float, default=ACTION_L2_WEIGHT)
    parser.add_argument("--obs-noise-scale", type=float, default=OBS_NOISE_SCALE)
    parser.add_argument("--critic-norm", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize Critic inputs online with RunningNorm")
    parser.add_argument("--teacher-init-fraction", type=float, default=TEACHER_INIT_FRACTION)
    parser.add_argument("--teacher-decay-iters", type=int, default=TEACHER_DECAY_ITERS)
    parser.add_argument("--teacher-bc-weight", type=float, default=TEACHER_BC_WEIGHT)
    parser.add_argument("--teacher-bc-batch", type=int, default=TEACHER_BC_BATCH)
    parser.set_defaults(
        termination_height_m=HEIGHT_THRESHOLD_M,
        termination_gravity_z=GRAVITY_Z_THRESHOLD,
        termination_softness=TERMINATION_SOFTNESS,
        eval_episodes=6,
    )


def args_from_cli(argv=None):
    args = train_v4.args_from_cli(argv, configure_parser=configure)
    return configure_objective(args)


def configure_objective(args):
    if not math.isfinite(args.drop_penalty) or args.drop_penalty < 0:
        raise ValueError("V6 requires a finite nonnegative drop penalty")
    if not math.isfinite(args.balance_reward_weight) or args.balance_reward_weight < 0:
        raise ValueError("V6 requires a finite nonnegative balance reward weight")
    if not math.isfinite(args.obs_noise_scale) or args.obs_noise_scale < 0:
        raise ValueError("V6 requires a finite nonnegative obs-noise-scale")
    if not 0.0 <= args.teacher_init_fraction <= 1.0:
        raise ValueError("teacher-init-fraction must be in [0, 1]")
    if args.teacher_decay_iters < 0 or args.teacher_bc_weight < 0 or args.teacher_bc_batch < 1:
        raise ValueError("teacher decay, BC weight, and BC batch size must be valid nonnegative/positive values")
    args.tracking_weights = v6_tracking_weights(args.balance_reward_weight)
    args.critic_value_max = v6_value_max(args.balance_reward_weight)
    return args


def train(args):
    train_v4.train(args, version=6, learner_type=LearnerV6)


if __name__ == "__main__":
    train(args_from_cli())
