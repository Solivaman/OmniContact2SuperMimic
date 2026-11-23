import pytest
import torch

from carrywm.train_v4 import UpdateClock
from carrywm.train_v7 import args_from_cli
from carrywm.validation import _phase_masks


def parse(*options):
    return args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", ".", *options])


def test_teacher_free_defaults_and_update_ratio():
    args = parse()
    assert args.num_envs == args.batch_size == 2048
    assert args.capacity == 2_048_000 and args.warmup_steps == 16_384
    assert args.cfgen_ratio == .1 and args.cfgen_goal_noise_std_m == 0
    assert not (args.teacher_init_fraction or args.teacher_bc_weight or args.warmup_teacher)
    clock = UpdateClock(args.num_envs)
    assert [clock.tick() for _ in range(10)] == [0, 0, 1, 0, 1] * 2
    assert 2 * 10 * args.wm_batch_size * 8 == 16 * 5 * args.num_envs


@pytest.mark.parametrize("options", [
    ("--teacher-init-fraction", ".1"),
    ("--teacher-bc-weight", ".1"), ("--warmup-teacher",), ("--goal-mode", "flow"),
])
def test_v7_rejects_incompatible_training_modes(options):
    with pytest.raises(ValueError):
        parse(*options)


def test_physical_phase_diagnostics_use_measured_lift_and_support():
    c = torch.zeros(5, 6)
    c[1:4, 2:4] = 1
    e = [dict(lifted=i>=2, supported=i<2 or i==4) for i in range(5)]
    masks = _phase_masks(torch.zeros(6, 84), c, torch.zeros(6, 111), e, grasp_phases=True)
    assert masks["bilateral_onset"].nonzero().flatten().tolist() == [1]
    assert masks["lift_onset"].nonzero().flatten().tolist() == [2]
    assert masks["carrying"].nonzero().flatten().tolist() == [2, 3]
