"""Tests for SuperMimic-style shared cosine LR decay in V7.3-cos."""
import math
import os
from pathlib import Path

import pytest
import torch

from carrywm.train import (
    DEFAULT_LR_COSINE_FRAC,
    DEFAULT_LR_FINAL_FRAC,
    lr_schedule_factor,
    resolve_lr_cosine_iters,
)
from carrywm.train_v4 import check_resume
from carrywm.train_v7 import args_from_cli as v7_args_from_cli
from carrywm.v7 import LearnerV7


def _find_sim_root():
    paths = (
        os.environ.get("CARRY_SIM_ROOT"),
        os.environ.get("SIM2SIM_ROOT"),
        os.environ.get("OMNICONTACT_SIM2SIM"),
        str(Path(__file__).resolve().parents[2] / "papers/OmniContact/code-sim2sim"),
        "../OmniContact_sim2sim",
        "../papers/OmniContact/code-sim2sim",
    )
    for entry in paths:
        if entry and (Path(entry) / "g1_description/g1_29dof.xml").is_file():
            return Path(entry).resolve()
    return None


@pytest.fixture
def sim_root():
    root = _find_sim_root()
    if root is None:
        pytest.skip("Set CARRY_SIM_ROOT to the released MuJoCo checkout")
    return root


def test_cosine_lr_formula_matches_supermimic_table_a5():
    assert math.isclose(DEFAULT_LR_FINAL_FRAC, 0.03)
    assert math.isclose(DEFAULT_LR_COSINE_FRAC, 250_000 / 320_000)
    assert resolve_lr_cosine_iters(320_000) == 250_000
    assert resolve_lr_cosine_iters(15_000) == 11_719
    assert resolve_lr_cosine_iters(15_000, lr_cosine_iters=8_000) == 8_000

    # Constant mode leaves LR unchanged at every iteration.
    for it in (1, 100, 11_719, 15_000):
        assert lr_schedule_factor(it, iterations=15_000, schedule="constant") == 1.0

    # Cosine decay 1.0 -> 0.03 over T=1000 steps (0-based 0..1000 -> 1-based 1..1001), then constant.
    f1 = lr_schedule_factor(1, iterations=1500, schedule="cosine", lr_cosine_iters=1000)
    f_mid = lr_schedule_factor(501, iterations=1500, schedule="cosine", lr_cosine_iters=1000)
    f_end = lr_schedule_factor(1001, iterations=1500, schedule="cosine", lr_cosine_iters=1000)
    f_tail = lr_schedule_factor(1500, iterations=1500, schedule="cosine", lr_cosine_iters=1000)

    assert math.isclose(f1, 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(f_mid, 0.515, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(f_end, 0.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(f_tail, 0.03, rel_tol=0.0, abs_tol=1e-12)


def test_cli_defaults_production_flags_and_validation():
    default_args = v7_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."])
    assert default_args.lr_schedule == "constant"
    assert math.isclose(default_args.lr_final_frac, 0.03)
    assert math.isclose(default_args.lr_cosine_frac, 0.78125)
    assert default_args.lr_cosine_iters is None

    cos_args = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", ".",
        "--iterations", "15000", "--lr", "2e-4",
        "--lr-schedule", "cosine", "--lr-final-frac", "0.03", "--lr-cosine-frac", "0.78125",
    ])
    assert cos_args.lr_schedule == "cosine"
    assert cos_args.iterations == 15_000
    assert resolve_lr_cosine_iters(cos_args.iterations, cos_args.lr_cosine_frac, cos_args.lr_cosine_iters) == 11_719

    for bad_flag in (
        ["--lr", "0"],
        ["--lr-final-frac", "0"],
        ["--lr-final-frac", "1.1"],
        ["--lr-cosine-frac", "0"],
        ["--lr-cosine-frac", "1.5"],
        ["--lr-cosine-iters", "0"],
    ):
        with pytest.raises(SystemExit):
            v7_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", ".", *bad_flag])


def test_learner_v7_set_lr_updates_all_optimizers_and_survives_resume(sim_root, tmp_path):
    args = v7_args_from_cli([
        "--sim-root", str(sim_root),
        "--motion-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--precision", "fp32",
        "--encoder-width", "32",
        "--decoder-width", "32",
        "--wm-width", "32",
        "--iterations", "15000",
        "--lr", "2e-4",
        "--lr-schedule", "cosine",
        "--lr-final-frac", "0.03",
        "--lr-cosine-frac", "0.78125",
    ])
    args.device = "cpu"
    net = LearnerV7(args)
    assert len(net.optimizers) == 3

    f1 = net.set_lr(1)
    assert math.isclose(f1, 1.0)
    for opt in net.optimizers:
        assert all(math.isclose(g["lr"], 2e-4, rel_tol=1e-7) for g in opt.param_groups)

    f_end = net.set_lr(11_720)
    assert math.isclose(f_end, 0.03, rel_tol=0.0, abs_tol=1e-12)
    for opt in net.optimizers:
        assert all(math.isclose(g["lr"], 6e-6, rel_tol=1e-7) for g in opt.param_groups)

    # Simulate checkpoint load_state_dict restoring an earlier optimizer LR, then next set_lr call.
    saved_states = [opt.state_dict() for opt in net.optimizers]
    net2 = LearnerV7(args)
    for opt, state in zip(net2.optimizers, saved_states):
        opt.load_state_dict(state)
    f_mid = net2.set_lr(5861)
    expected_lr = 2e-4 * f_mid
    for opt in net2.optimizers:
        assert all(math.isclose(g["lr"], expected_lr, rel_tol=1e-7) for g in opt.param_groups)


def test_check_resume_lr_compatibility():
    base_args = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", ".",
        "--iterations", "10000",
    ])
    legacy_ck = {
        "version": 7,
        "iteration": 1000,
        "config": {
            "num_envs": base_args.num_envs,
            "capacity": base_args.capacity,
            "cfgen_ratio": base_args.cfgen_ratio,
            "action_noise": base_args.action_noise,
            "goal_mode": base_args.goal_mode,
            "seed": base_args.seed,
            "termination_height_m": base_args.termination_height_m,
            "termination_gravity_z": base_args.termination_gravity_z,
            "termination_softness": base_args.termination_softness,
            "critic_value_max": base_args.critic_value_max,
            "drop_penalty": base_args.drop_penalty,
            "balance_reward_weight": base_args.balance_reward_weight,
            "action_rate_weight": base_args.action_rate_weight,
            "action_l2_weight": base_args.action_l2_weight,
            "obs_noise_scale": base_args.obs_noise_scale,
            "critic_norm": base_args.critic_norm,
            "teacher_init_fraction": base_args.teacher_init_fraction,
            "teacher_decay_iters": base_args.teacher_decay_iters,
            "teacher_bc_weight": base_args.teacher_bc_weight,
        },
    }
    check_resume(legacy_ck, base_args, 7)

    cos_args = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", ".",
        "--iterations", "15000", "--lr-schedule", "cosine",
        "--lr-final-frac", "0.03", "--lr-cosine-frac", "0.78125",
    ])
    with pytest.raises(ValueError, match="lr_schedule"):
        check_resume(legacy_ck, cos_args, 7)

    cos_ck = {
        **legacy_ck,
        "config": {
            **legacy_ck["config"],
            "lr_schedule": "cosine",
            "lr_final_frac": 0.03,
            "lr_cosine_frac": 0.78125,
            "lr_cosine_iters": None,
        },
    }
    check_resume(cos_ck, cos_args, 7)

    bad_frac = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", ".",
        "--iterations", "15000", "--lr-schedule", "cosine",
        "--lr-final-frac", "0.05", "--lr-cosine-frac", "0.78125",
    ])
    with pytest.raises(ValueError, match="lr_final_frac"):
        check_resume(cos_ck, bad_frac, 7)
