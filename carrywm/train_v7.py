"""Learn carry-box control from simulated transitions and motion references."""
from . import train_v4, train_v6
from .v7 import LearnerV7


def configure(parser):
    train_v6.configure(parser)
    parser.description = __doc__
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
    return train_v6.configure_objective(args)


def train(args):
    train_v4.train(args, version=7, learner_type=LearnerV7)


if __name__ == "__main__":
    train(args_from_cli())
