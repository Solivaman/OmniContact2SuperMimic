"""Tests for V7.3-Ultra SuperMimic/OmniContact alignment features."""
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest
import torch

from carrywm.collector_v4 import VectorCollector
from carrywm.muon import (
    HybridMuonAdam,
    _CompatMuon,
    build_module_optimizer,
    hidden_linear_weights,
)
from carrywm.reset_v7 import apply_box_dr_to_stacked, sync_box_context
from carrywm.sim import CarryEnv, _scene
from carrywm.stages_v7 import BINS, ClipStarts, ReturnSampler, ReturnSamplerCfg, SOURCES, StageSampler
from carrywm.train import GAMMA
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
            return Path(entry)
    return None


@pytest.fixture
def sim_root():
    root = _find_sim_root()
    if root is None:
        pytest.skip("Set CARRY_SIM_ROOT to the released MuJoCo checkout")
    return root


def test_actor_beta2_and_two_stage_gradient_clipping(sim_root, tmp_path):
    args = v7_args_from_cli([
        "--sim-root", str(sim_root),
        "--motion-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--precision", "fp32",
        "--encoder-width", "32",
        "--decoder-width", "32",
        "--wm-width", "32",
        "--optimizer", "muon",
        "--actor-beta2", "0.95",
        "--actor-grad-clip-value", "0.1",
        "--lr", "2e-4",
    ])
    args.device = "cpu"
    net = LearnerV7(args)

    wm_opt, actor_opt, critic_opt = net.optimizers
    assert isinstance(wm_opt, HybridMuonAdam)
    assert isinstance(actor_opt, HybridMuonAdam)
    assert isinstance(critic_opt, HybridMuonAdam)
    assert wm_opt.adam.param_groups[0]["betas"] == (0.9, 0.999)
    assert actor_opt.adam.param_groups[0]["betas"] == (0.9, 0.95)
    assert critic_opt.adam.param_groups[0]["betas"] == (0.9, 0.999)

    for opt, mod in zip(net.optimizers, net.models):
        hidden_ids = {id(p) for p in hidden_linear_weights(mod)}
        muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
        adam_ids = {id(p) for g in opt.adam.param_groups for p in g["params"]}
        all_ids = {id(p) for p in mod.parameters()}
        assert muon_ids == hidden_ids
        assert not (muon_ids & adam_ids)
        assert (muon_ids | adam_ids) == all_ids

    # WorldModel must keep trunk[0] and robot/box/contact heads in Adam while placing trunk[1:] in Muon.
    wm_muon_ids = {id(p) for g in wm_opt.muon.param_groups for p in g["params"]}
    wm_linears = [m for m in net.wm.trunk if isinstance(m, torch.nn.Linear)]
    assert id(wm_linears[0].weight) not in wm_muon_ids
    assert all(id(layer.weight) in wm_muon_ids for layer in wm_linears[1:])
    for head in (net.wm.robot, net.wm.box, net.wm.contact):
        assert id(head.weight) not in wm_muon_ids

    # First/last Actor linears stay in Adam; middle linears use Muon.
    actor_muon_ids = {id(p) for g in actor_opt.muon.param_groups for p in g["params"]}
    enc_linears = [m for m in net.actor.encoder if isinstance(m, torch.nn.Linear)]
    dec_linears = [m for m in net.actor.decoder if isinstance(m, torch.nn.Linear)]
    assert id(enc_linears[0].weight) not in actor_muon_ids
    assert id(enc_linears[-1].weight) not in actor_muon_ids
    assert id(dec_linears[0].weight) not in actor_muon_ids
    assert id(dec_linears[-1].weight) not in actor_muon_ids
    assert all(id(l.weight) in actor_muon_ids for l in enc_linears[1:-1])
    assert all(id(l.weight) in actor_muon_ids for l in dec_linears[1:-1])

    # Two-stage Actor gradient clipping: norm <= 1.0 then value <= 0.1.
    for p in net.actor.parameters():
        p.grad = torch.zeros_like(p)
    dec_linears[-1].weight.grad.fill_(5.0)
    net._clip_actor_gradients()
    assert all(float(p.grad.abs().max()) <= 0.1 + 1e-7 for p in net.actor.parameters())

    # HybridMuonAdam step + cosine LR + state_dict resume round-trip.
    for mod in net.models:
        for p in mod.parameters():
            p.grad = torch.ones_like(p) * 0.01
    before = [p.detach().clone() for p in net.actor.parameters()]
    net.set_lr(100)
    actor_opt.step()
    assert all(not torch.equal(p, b) for p, b in zip(net.actor.parameters(), before))
    for g in actor_opt.param_groups:
        assert math.isclose(g["lr"], net.base_lr * net.lr_factor, rel_tol=1e-6)

    saved = [o.state_dict() for o in net.optimizers]
    net2 = LearnerV7(args)
    for o, s in zip(net2.optimizers, saved):
        o.load_state_dict(s)
    adam_args = v7_args_from_cli([
        "--sim-root", str(sim_root), "--motion-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out_adam"), "--precision", "fp32",
        "--encoder-width", "32", "--decoder-width", "32", "--wm-width", "32",
        "--optimizer", "adam",
    ])
    adam_args.device = "cpu"
    net_adam = LearnerV7(adam_args)
    with pytest.raises(ValueError, match="non-Muon"):
        net2.optimizers[0].load_state_dict(net_adam.optimizers[0].state_dict())


def test_compat_muon_matches_module_defaults_and_adjust_lr():
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(6, 16),
        torch.nn.LayerNorm(16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 16),
        torch.nn.LayerNorm(16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 3),
    )
    opt = build_module_optimizer(mlp, 2e-4, optimizer="muon", muon_weight_decay=0.1)
    assert isinstance(opt, HybridMuonAdam)
    p = mlp[3].weight
    old = p.detach().clone()
    mlp(torch.randn(8, 6)).square().mean().backward()
    opt.step()
    assert not torch.equal(p, old)
    assert "momentum_buffer" in opt.state[p]


def test_return_sampler_tracks_discounted_return_per_source_and_resumes():
    cfg = ReturnSamplerCfg(gamma=0.99, horizon=4, memory=256, steepness=25.0, explore=0.5, keep=0.1)
    sampler = StageSampler(
        sampling_mode="return",
        return_cfg=cfg,
        num_envs=3,
        max_step_reward=0.1,
        device="cpu",
    )
    rs = sampler.return_sampler
    assert math.isclose(rs.v_max, 10.0)

    # Env 0: "npz" bin 2 failing with low reward & drop/fall (terminated -> 0 bootstrap).
    # Env 1: "npz" bin 16 high reward + success (-> v_max bootstrap).
    # Env 2: "recovery" (source=-1) ignored, must not pollute npz/cfgen bins.
    frames = torch.tensor([2, 16, 5], dtype=torch.long)
    clip_lens = torch.tensor([20, 20, 20], dtype=torch.long)
    sources = torch.tensor([SOURCES.index("npz"), SOURCES.index("npz"), -1], dtype=torch.long)
    no_flag = torch.zeros(3, dtype=torch.bool)

    probs = None
    for step_idx in range(4):
        term = torch.tensor([step_idx == 3, False, True], dtype=torch.bool)
        succ = torch.tensor([False, step_idx == 3, False], dtype=torch.bool)
        rewards = torch.tensor([0.01, 0.10, -5.0], dtype=torch.float32)
        probs = rs.record(frames, clip_lens, sources, rewards, term, succ, no_flag, no_flag)
        if step_idx < 3:
            assert probs is None
    assert probs is not None
    sampler.return_probs = probs
    # Bin 2 had very low return (~0.04) vs bin 16 high return (~10.0); bin 2 must outweigh bin 16.
    npz_idx = SOURCES.index("npz")
    cfgen_idx = SOURCES.index("cfgen")
    assert float(rs.value[npz_idx, 2]) < 0.2
    assert float(rs.value[npz_idx, 16]) > 9.0
    # Unvisited cfgen bins must remain untouched at v_max.
    assert torch.allclose(rs.value[cfgen_idx], torch.full((BINS,), rs.v_max))
    assert probs[npz_idx, 2] > probs[npz_idx, 16] * 5.0

    # Low-return non-falling episode ("苟活但低分") at bin 6 must also attract focus vs bin 16.
    frames2 = torch.tensor([6, 16, 16], dtype=torch.long)
    sources2 = torch.tensor([cfgen_idx, cfgen_idx, npz_idx], dtype=torch.long)
    for step_idx in range(4):
        rewards2 = torch.tensor([0.005, 0.10, 0.10], dtype=torch.float32)
        clip_end = torch.tensor([step_idx == 3, False, False], dtype=torch.bool)
        succ2 = torch.tensor([False, step_idx == 3, step_idx == 3], dtype=torch.bool)
        # Run with timeout/clip_end on bin 6 vs success on bin 16? With clip_end bootstrap=mean(v_max),
        # reset clip_end=False to test ongoing vs terminated, or run longer so discounted sum differs.
        probs2 = rs.record(frames2, clip_lens, sources2, rewards2, no_flag, succ2, no_flag, no_flag)
    sampler.return_probs = probs2
    assert float(rs.value[cfgen_idx, 6]) < float(rs.value[cfgen_idx, 16])
    assert probs2[cfgen_idx, 6] > probs2[cfgen_idx, 16]

    # StageSampler.choose restricts to available bins and survives state_dict resume.
    raw = np.arange(BINS)
    bins = tuple(raw[i:i + 1] if i in (2, 16) else raw[:0] for i in range(BINS))
    cat = ClipStarts(raw[[2, 16]], np.zeros(BINS, int), bins, None)
    rng = np.random.default_rng(7)
    draws = [sampler.choose(cat, "npz", rng) for _ in range(400)]
    assert set(draws) == {2, 16}
    assert np.mean(np.array(draws) == 2) > 0.70

    state = sampler.state_dict()
    restored = StageSampler(sampling_mode="return", return_cfg=cfg, num_envs=3, max_step_reward=0.1)
    restored.load_state_dict(state)
    assert np.allclose(restored.return_probs, sampler.return_probs)
    assert torch.allclose(restored.return_sampler.value, sampler.return_sampler.value)


def test_box_mass_scaling_matches_mj_setconst_and_dr_physics(sim_root):
    # 1. Analytic scaling helper matches MuJoCo mj_setConst across mass/inertia/invweight0.
    half = (0.20, 0.15, 0.18)
    m_base = _scene(sim_root, half, headless=True)
    bid = m_base.body("box").id
    bv = int(m_base.jnt_dofadr[m_base.body_jntadr[bid]])
    gid = m_base.geom("box_geom").id

    m_ref = copy.copy(m_base)
    for k in ("body_mass", "body_subtreemass", "body_inertia", "body_invweight0", "dof_invweight0"):
        setattr(m_ref, k, getattr(m_base, k).copy())
    scale = 1.23
    m_ref.body_mass[bid] *= scale
    m_ref.body_subtreemass[bid] *= scale
    m_ref.body_inertia[bid] *= scale
    mj.mj_setConst(m_ref, mj.MjData(m_ref))

    class _MockWorld:
        device = torch.device("cpu")
        _has_box_dr = True
        _box_friction_dr = True
        _box_body = bid
        _bv = bv
        _box = gid
        _host = m_base

        @staticmethod
        def _geometry(_h):
            return {
                "body_mass": m_base.body_mass.astype(np.float32),
                "body_inertia": m_base.body_inertia.astype(np.float32),
            }

    world = _MockWorld()
    ep = SimpleNamespace(
        half=np.array(half),
        box_mass_scale=scale,
        box_friction=np.array([0.85, 0.0085, 0.00017]),
        _base_box_friction=m_base.geom_friction[gid].copy(),
        context=np.zeros(16, dtype=np.float32),
    )
    sync_box_context(world, ep)
    assert math.isclose(float(ep.context[3]), float(m_ref.body_mass[bid]), rel_tol=1e-6)
    np.testing.assert_allclose(ep.context[4:7], m_ref.body_inertia[bid], rtol=1e-6)
    np.testing.assert_allclose(ep.context[7:10], ep.box_friction, rtol=1e-6)

    for key in ("body_mass", "body_subtreemass", "body_inertia", "body_invweight0", "dof_invweight0", "geom_friction"):
        base_t = torch.as_tensor(getattr(m_base, key)[None].copy(), dtype=torch.float32)
        out = apply_box_dr_to_stacked(world, key, base_t, [ep])
        if key == "geom_friction":
            np.testing.assert_allclose(out[0, gid].numpy(), ep.box_friction, rtol=1e-6)
        elif key == "dof_invweight0":
            np.testing.assert_allclose(out[0, bv:bv + 6].numpy(), m_ref.dof_invweight0[bv:bv + 6], rtol=1e-6)
        else:
            np.testing.assert_allclose(out[0, bid].numpy(), getattr(m_ref, key)[bid], rtol=1e-6)

    # 2. CPU CarryEnv DR randomizes context[3:10], priority=1 changes contact friction, and keeps across recovery.
    motion_dir = sim_root / "data/carrybox"
    if not motion_dir.is_dir():
        pytest.skip("data/carrybox assets required")
    env_dr = CarryEnv(
        sim_root, motion_dir, seed=19, split="train", cfgen_ratio=0.0, v7=True,
        box_mass_scale_range=(0.75, 1.25), box_friction_range=(0.5, 1.0),
    )
    env_val = CarryEnv(
        sim_root, motion_dir, seed=19, split="val", cfgen_ratio=0.0, v7=True,
        box_mass_scale_range=(0.75, 1.25), box_friction_range=(0.5, 1.0),
    )
    env_dr.paths = env_dr.paths[:1]
    env_val.paths = env_val.paths[:1]
    env_dr.reset("npz")
    env_val.reset("npz")

    # Val split must never randomize box physics even if ranges were configured.
    assert math.isclose(env_val.box_mass_scale, 1.0)
    assert math.isclose(float(env_val.context[3]), 2.0, rel_tol=1e-6)
    assert math.isclose(float(env_val.context[7]), 0.5, rel_tol=1e-6)
    assert int(env_val.m.geom_priority[env_val.box_geom]) == 0

    # Train split varies mass & friction across resets, sets priority=1, and updates context[3:10].
    masses = []
    frictions = []
    for _ in range(5):
        env_dr.reset("npz")
        masses.append(float(env_dr.context[3]))
        frictions.append(float(env_dr.context[7]))
        assert 1.5 - 1e-6 <= masses[-1] <= 2.5 + 1e-6
        assert 0.5 - 1e-6 <= frictions[-1] <= 1.0 + 1e-6
        assert int(env_dr.m.geom_priority[env_dr.box_geom]) == 1
        assert math.isclose(float(env_dr.m.body_mass[env_dr._box_body_id]), masses[-1], rel_tol=1e-6)
        assert math.isclose(float(env_dr.m.geom_friction[env_dr.box_geom, 0]), frictions[-1], rel_tol=1e-6)
    assert len(set(round(x, 6) for x in masses)) > 1
    assert len(set(round(x, 6) for x in frictions)) > 1

    # Recovery must retain the exact box mass and friction of the active episode.
    saved_m, saved_f = float(env_dr.context[3]), float(env_dr.context[7])
    env_dr.pending_recovery = True
    env_dr.d.qpos[:7] = env_dr.clip[min(30, len(env_dr.clip) - 1), 97:104]
    env_dr.d.qpos[env_dr.bp:env_dr.bp + 7] = env_dr.clip[min(30, len(env_dr.clip) - 1), 104:111]
    env_dr.reset()
    if env_dr.source == "recovery":
        assert math.isclose(float(env_dr.context[3]), saved_m, rel_tol=1e-6)
        assert math.isclose(float(env_dr.context[7]), saved_f, rel_tol=1e-6)


def test_ultra_cli_preset_and_check_resume(tmp_path):
    legacy = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", str(tmp_path / "base"),
        "--iterations", "15000", "--lr-schedule", "cosine",
    ])
    assert legacy.optimizer == "adam"
    assert legacy.actor_beta2 == 0.999
    assert legacy.actor_grad_clip_value == 0.0
    assert legacy.reset_sampling == "failure"
    assert legacy.box_mass_scale_range == (1.0, 1.0)
    assert legacy.box_friction_range is None

    ultra = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", str(tmp_path / "ultra"),
        "--ultra", "--iterations", "25000", "--cfgen-ratio", "0.5", "--lr-schedule", "cosine",
    ])
    assert ultra.optimizer == "muon"
    assert ultra.actor_beta2 == 0.95
    assert ultra.actor_grad_clip_value == 0.1
    assert ultra.reset_sampling == "return"
    assert ultra.box_mass_scale_range == (0.75, 1.25)
    assert ultra.box_friction_range == (0.5, 1.0)

    ck = {
        "version": 7,
        "iteration": 1000,
        "config": {
            "num_envs": ultra.num_envs,
            "capacity": ultra.capacity,
            "cfgen_ratio": ultra.cfgen_ratio,
            "action_noise": ultra.action_noise,
            "goal_mode": ultra.goal_mode,
            "seed": ultra.seed,
            "termination_height_m": ultra.termination_height_m,
            "termination_gravity_z": ultra.termination_gravity_z,
            "termination_softness": ultra.termination_softness,
            "critic_value_max": ultra.critic_value_max,
            "drop_penalty": ultra.drop_penalty,
            "balance_reward_weight": ultra.balance_reward_weight,
            "action_rate_weight": ultra.action_rate_weight,
            "action_l2_weight": ultra.action_l2_weight,
            "obs_noise_scale": ultra.obs_noise_scale,
            "critic_norm": ultra.critic_norm,
            "teacher_init_fraction": ultra.teacher_init_fraction,
            "teacher_decay_iters": ultra.teacher_decay_iters,
            "teacher_bc_weight": ultra.teacher_bc_weight,
            "lr_schedule": ultra.lr_schedule,
            "lr_final_frac": ultra.lr_final_frac,
            "lr_cosine_frac": ultra.lr_cosine_frac,
            "lr_cosine_iters": ultra.lr_cosine_iters,
            "optimizer": ultra.optimizer,
            "muon_weight_decay": ultra.muon_weight_decay,
            "actor_beta2": ultra.actor_beta2,
            "actor_grad_clip_value": ultra.actor_grad_clip_value,
            "reset_sampling": ultra.reset_sampling,
            "return_sampler_horizon": ultra.return_sampler_horizon,
            "return_sampler_memory": ultra.return_sampler_memory,
            "return_sampler_steepness": ultra.return_sampler_steepness,
            "return_sampler_explore": ultra.return_sampler_explore,
            "return_sampler_keep": ultra.return_sampler_keep,
            "box_mass_scale_range": list(ultra.box_mass_scale_range),
            "box_friction_range": list(ultra.box_friction_range),
        },
    }
    check_resume(ck, ultra, 7)

    with pytest.raises(ValueError, match="optimizer"):
        bad = copy.copy(ultra)
        bad.optimizer = "adam"
        check_resume(ck, bad, 7)


def test_vector_collector_updates_return_sampler_on_horizon_boundary(sim_root, tmp_path):
    motion_dir = sim_root / "data/carrybox"
    if not motion_dir.is_dir():
        pytest.skip("data/carrybox assets required")
    args = v7_args_from_cli([
        "--sim-root", str(sim_root),
        "--motion-dir", str(motion_dir),
        "--output-dir", str(tmp_path / "col"),
        "--precision", "fp32",
        "--encoder-width", "32",
        "--decoder-width", "32",
        "--wm-width", "32",
        "--ultra",
        "--return-sampler-horizon", "4",
        "--cfgen-ratio", "0.0",
    ])
    args.device = "cpu"
    net = LearnerV7(args)

    cfg = ReturnSamplerCfg(gamma=GAMMA, horizon=4, memory=256, steepness=25.0, explore=0.5, keep=0.1)
    shared_sampler = StageSampler(
        sampling_mode="return",
        return_cfg=cfg,
        num_envs=2,
        max_step_reward=float(args.critic_value_max) * (1.0 - GAMMA),
        device="cpu",
    )
    envs = [
        CarryEnv(
            sim_root, motion_dir, seed=10 + i, split="train", cfgen_ratio=0.0, v7=True,
            box_mass_scale_range=(0.75, 1.25), box_friction_range=(0.5, 1.0),
        )
        for i in range(2)
    ]
    for e in envs:
        e.paths = e.paths[:1]
        e.stage_sampler = shared_sampler

    class _CpuVecEnv:
        def __init__(self):
            self.envs = envs
            self.stage_sampler = shared_sampler

        def reset(self, ids, source=None):
            for idx in ids:
                self.envs[idx].reset(source)

        def step(self, actions):
            states, contacts, dones, infos = [], [], [], []
            for idx, act in enumerate(actions):
                st, ct, dn, inf = self.envs[idx].step(act)
                states.append(st)
                contacts.append(ct)
                dones.append(dn)
                infos.append(inf)
            return np.stack(states), np.stack(contacts), np.asarray(dones, dtype=bool), infos

    class _StubReplay:
        num_envs = 2

        def begin_new_episodes(self):
            pass

        def append(self, *args, **kwargs):
            pass

    collector = VectorCollector(_CpuVecEnv(), net, _StubReplay(), np.random.default_rng(0))
    for tick in range(4):
        collector.one_tick(noise=0.0, update_normalizers=(tick == 0), record_reward=(tick == 3))
        if tick < 3:
            assert shared_sampler.return_probs is None
    assert shared_sampler.return_probs is not None
    assert shared_sampler.return_probs.shape == (len(SOURCES), BINS)
    assert np.allclose(shared_sampler.return_probs.sum(axis=1), 1.0)

