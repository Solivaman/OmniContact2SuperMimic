"""Tests for V7.2 speed optimizations and numerical/checkpoint equivalence."""
from collections import deque
from concurrent.futures import Future
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import carrywm.train as train_mod
from carrywm.device_clips import PagedClipBank
from carrywm.model import CarryGeometry
from carrywm.recovery_gpu import BatchedRecoveryFK, RecoverySkeleton
from carrywm.sim import CarryEnv, _matrix
from carrywm.stages_v7 import _clip_box_metadata
from carrywm.train import HORIZON, OFFSETS, WM_STEPS, frozen
from carrywm.train_v3 import wm_iteration
from carrywm.train_v4 import args_from_cli as v4_args_from_cli
from carrywm.train_v7 import args_from_cli as v7_args_from_cli
from carrywm.v7 import LearnerV7
from carrywm.warp_sim import WarpCarryWorlds, _Episode, _UNSET


def _find_sim_root():
    paths = (
        os.environ.get("SIM2SIM_ROOT"),
        os.environ.get("OMNICONTACT_SIM2SIM"),
        "/Users/huweihang/Documents/ChatGPT/人形/OmniContact2SuperMimic-v7.2/../papers/OmniContact/code-sim2sim",
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
        pytest.skip("Set SIM2SIM_ROOT to the released MuJoCo model")
    return root


def _sample_state(batch=4, seed=11):
    gen = torch.Generator().manual_seed(seed)
    state = torch.zeros(batch, 84)
    state[:, :3] = torch.randn(batch, 3, generator=gen) * 0.2
    state[:, 2] += 0.8
    state[:, 3:7] = torch.randn(batch, 4, generator=gen)
    state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
    state[:, 7:36] = torch.randn(batch, 29, generator=gen) * 0.4
    state[:, 71:74] = state[:, :3] + torch.tensor([0.35, 0.0, 0.1])
    state[:, 74:78] = torch.randn(batch, 4, generator=gen)
    state[:, 74:78] = state[:, 74:78] / state[:, 74:78].norm(dim=-1, keepdim=True)
    return state


def test_palm_fk_matches_full_fk_and_preserves_state_dict(sim_root):
    base_geo = CarryGeometry(sim_root)
    v7_geo = CarryGeometry(sim_root, box_goal=True, fused_rotations=True)
    assert set(v7_geo.state_dict().keys()) == set(base_geo.state_dict().keys())

    state = _sample_state(batch=8, seed=7)
    ctx = torch.ones(8, 16) * 0.15
    fk = v7_geo.forward_kinematics(state)
    palm_rot, palm_pos = v7_geo.palm_fk(state[..., :3], state[..., 3:7], state[..., 7:36])

    full_palms = fk.index_select(-3, v7_geo.flow_ids[:2])
    torch.testing.assert_close(palm_rot, full_palms[..., :3, :3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(palm_pos, full_palms[..., :3, 3], atol=1e-6, rtol=1e-5)

    via_palm_fk = v7_geo.palms_in_box(state, ctx)
    via_full_fk = v7_geo.palms_in_box(state, ctx, fk=fk)
    torch.testing.assert_close(via_palm_fk, via_full_fk, atol=1e-5, rtol=1e-5)


def test_wm_iteration_deferred_sync_and_single_stats_update(sim_root):
    torch.manual_seed(13)
    args = SimpleNamespace(
        device="cpu",
        sim_root=sim_root,
        goal_mode="full",
        lr=1e-3,
        precision="fp32",
        encoder_width=16,
        decoder_width=16,
        wm_width=16,
        drop_penalty=1.0,
        critic_norm=True,
        obs_noise_scale=0.0,
        compile=False,
        fused_rotations=False,
    )
    net = LearnerV7(args)
    calls = []

    class Windows:
        def wm(self, batch, steps):
            calls.append((batch, steps))
            state = torch.zeros(batch, steps + 1, 84)
            state[:, :, 2] = 0.8
            state[:, :, [3, 74]] = 1.0
            state[:, :, 7] = 0.05
            return (
                state,
                torch.zeros(batch, steps, 29),
                torch.zeros(batch, steps, 6),
                torch.full((batch, 16), 0.15),
            )

    metrics = wm_iteration(net, Windows(), batch_size=32)
    assert calls == [(32, WM_STEPS)] * 10
    assert metrics["wm/optimizer_steps"] == 10
    assert metrics["wm/windows_per_update"] == 32
    assert isinstance(metrics["wm/loss"], float) and np.isfinite(metrics["wm/loss"])
    assert isinstance(metrics["wm/box_8step_m"], float) and np.isfinite(metrics["wm/box_8step_m"])
    assert float(net.wm.inputs.count.item()) == 32 * WM_STEPS
    weight = next(net.wm.parameters())
    assert net.optimizers[0].state[weight]["step"].item() == 10


@pytest.mark.parametrize("checkpoint_encoder", [False, True])
def test_chunked_rollout_matches_single_step_and_backward_gradients(sim_root, checkpoint_encoder):
    torch.manual_seed(29)
    args = SimpleNamespace(
        device="cpu",
        sim_root=sim_root,
        goal_mode="full",
        lr=1e-3,
        precision="fp32",
        encoder_width=16,
        decoder_width=16,
        wm_width=16,
        drop_penalty=1.0,
        critic_norm=True,
        obs_noise_scale=0.0,
        compile=False,
        fused_rotations=False,
    )
    net = LearnerV7(args)
    state = _sample_state(batch=3, seed=3)
    prev = torch.zeros(3, 29)
    ctx = torch.full((3, 16), 0.15)
    hist = 0.05 * torch.randn(3, 10, 141)
    raw = 0.02 * torch.randn(3, HORIZON + 51, 111)
    raw[..., 99] = 0.8
    raw[..., 106] = 0.9
    for i in range(5):
        raw[..., 7 * i + 3] = 1.0
    raw[..., [100, 107]] = 1.0
    remaining = torch.full((3,), float(HORIZON + 10))
    inputs = (state, prev, ctx, hist, raw, remaining)

    net.actor.zero_grad(set_to_none=True)
    with frozen(net.wm, net.critic):
        ret_c, mask_c, st_c, gl_c, rw_c, ct_c, vr_c = net._rollout(
            inputs, checkpoint_encoder=checkpoint_encoder
        )
        (-ret_c.mean()).backward()
        grads_c = [p.grad.detach().clone() for p in net.actor.parameters()]

    net.actor.zero_grad(set_to_none=True)
    with patch.object(train_mod, "ROLLOUT_CHUNK", HORIZON + 1):
        with frozen(net.wm, net.critic):
            ret_s, mask_s, st_s, gl_s, rw_s, ct_s, vr_s = net._rollout(
                inputs, checkpoint_encoder=checkpoint_encoder
            )
            (-ret_s.mean()).backward()
            grads_s = [p.grad.detach().clone() for p in net.actor.parameters()]

    torch.testing.assert_close(ret_c, ret_s, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(mask_c, mask_s, atol=1e-6, rtol=1e-5)
    assert len(st_c) == HORIZON + 1 and len(rw_c) == HORIZON
    for a, b in zip(st_c, st_s):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(gl_c, gl_s):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(rw_c, rw_s):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(ct_c, ct_s):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(vr_c, vr_s):
        assert torch.equal(a, b)
    for ga, gs in zip(grads_c, grads_s):
        torch.testing.assert_close(ga, gs, atol=1e-5, rtol=1e-4)


def test_stage_sampler_cached_starts_holding_and_extent_z():
    T = 12
    frames = np.zeros((T, 111), dtype=np.float32)
    frames[:, 107] = 1.0  # identity box quat except rotated frame
    frames[-1, 107:111] = np.array([np.cos(0.3), np.sin(0.3), 0.0, 0.0], dtype=np.float32)
    half = np.array([0.12, 0.15, 0.20], dtype=np.float64)
    table_positions = np.array([[0.0, 0.0, 0.40], [1.0, 0.0, 0.40]], dtype=np.float64)
    table_half_sizes = np.array([[0.30, 0.30, 0.05], [0.30, 0.30, 0.05]], dtype=np.float64)

    # Set per-frame cases:
    # 0: held at frame 0 (must be False)
    frames[0, 37:39] = 1.0
    frames[0, 104:107] = [3.0, 0.0, 0.8]
    # 1: held free in mid-air
    frames[1, 37:39] = 1.0
    frames[1, 104:107] = [3.0, 0.0, 0.8]
    # 2: resting on floor
    frames[2, 37:39] = 1.0
    frames[2, 104:107] = [3.0, 0.0, 0.202]
    # 3: resting on table 0 (top at 0.45, box bottom <= 0.455)
    frames[3, 37:39] = 1.0
    frames[3, 104:107] = [0.0, 0.0, 0.654]
    # 4: lifted clearly above table 0 (bottom > 0.455)
    frames[4, 37:39] = 1.0
    frames[4, 104:107] = [0.0, 0.0, 0.72]
    # 5: single hand only in mid-air
    frames[5, 37] = 1.0
    frames[5, 104:107] = [3.0, 0.0, 0.8]

    _, starts_holding, extent_z = _clip_box_metadata(frames, half, table_positions, table_half_sizes)
    expected_ext_z = float(np.abs(_matrix(frames[-1, 107:111])[2]) @ half)
    assert np.isclose(extent_z, expected_ext_z)

    # Check scalar CarryEnv._starts_holding equivalence frame-by-frame.
    dummy_m = SimpleNamespace(geom_size=table_half_sizes)
    dummy_d = SimpleNamespace(mocap_pos=table_positions)
    for t in range(T):
        env_stub = CarryEnv.__new__(CarryEnv)
        env_stub._clip_catalogue = None
        env_stub.clip = frames
        env_stub.index = t
        env_stub.half = half
        env_stub.table_ids = [0, 1]
        env_stub.tables_geom = [0, 1]
        env_stub.d = dummy_d
        env_stub.m = dummy_m
        assert bool(starts_holding[t]) == env_stub._starts_holding(), f"mismatch at frame {t}"


def _make_deferred_world_harness(max_inflight=4):
    w = WarpCarryWorlds.__new__(WarpCarryWorlds)
    w.num_envs = 3
    w.v7 = True
    w._gpu_recovery = False
    w._batch_reset = True
    w._keep_recovery_on_device = False
    w._recovery_fk = None
    w._recovery_futures = {}
    w._defer_recovery = True
    w._deferred_recoveries = deque()
    w._max_inflight_recoveries = max_inflight
    w._recovery_workers = 4
    w.last_recovery_wait_sec = 0.0
    w.last_recovery_deferred_queued = 0
    w.last_recovery_deferred_consumed = 0
    w.last_recovery_deferred_dropped = 0
    w.teacher_history = False
    w.stage_sampler = None
    w._rec_pool = Mock()

    def submit_future(*_):
        f = Mock(spec=Future)
        f.done.return_value = False
        f.result.side_effect = AssertionError("deferred mode must not block on unfinished future")
        return f

    w._rec_pool.submit.side_effect = submit_future
    w._qpos = torch.arange(3 * 43, dtype=torch.float32).reshape(3, 43)
    w._qvel = torch.zeros(3, 41)
    w._env_index = torch.tensor([5, 6, 7], dtype=torch.int32)
    w._overflow_torch = torch.zeros(3, dtype=torch.int32)
    w._packed_flags = torch.empty(4, 5, dtype=torch.uint8)
    w._step_physics = lambda action: action
    w._sync_worlds_to_host = Mock(wraps=w._sync_worlds_to_host)
    w.envs = []
    for i in range(3):
        e = _Episode.__new__(_Episode)
        e.source, e.name, e.root, e.bp, e.v7 = "npz", f"scene_{i}", Path("/scene"), 36, True
        e._precomputed_recovery, e.pending_recovery = _UNSET, False
        e._recovery_snapshot = None
        e._deferred_snapshot = None
        e._clip_gpu = None
        e.half = np.full(3, i + 0.15)
        e.goal = np.full(3, i + 0.2)
        e.d = SimpleNamespace(
            qpos=np.zeros(43, dtype=np.float32),
            qvel=np.zeros(41, dtype=np.float32),
            mocap_pos=np.full((2, 3), i + 0.5, dtype=np.float32),
        )
        e.rng = np.random.default_rng(i)
        w.envs.append(e)
    return w


def test_deferred_recovery_queues_caps_consumes_and_cancels():
    w = _make_deferred_world_harness(max_inflight=1)
    drop = torch.tensor([True, True, False])
    zero = torch.zeros_like(drop)
    w._eval_gpu_step = Mock(
        return_value=(torch.zeros(3, 84), torch.zeros(3, 6), None, drop, zero, drop, zero, zero, None)
    )
    w.step_gpu(torch.zeros(3, 29))

    # Max inflight=1: world 0 accepted, world 1 dropped before host sync.
    assert list(w._recovery_futures.keys()) == [0]
    assert w.last_recovery_deferred_dropped == 1
    assert not w.envs[1].pending_recovery
    w._sync_worlds_to_host.assert_called_once_with([0])
    snap0 = w.envs[0]._recovery_snapshot
    assert snap0["name"] == "scene_0"
    np.testing.assert_array_equal(snap0["qpos"], w._qpos[0].numpy())

    # Unfinished future is deferred onto queue without calling result()!
    w.last_recovery_deferred_queued = 0
    w.last_recovery_deferred_consumed = 0
    w._prepare_recoveries([0], source=None)
    assert w.last_recovery_deferred_queued == 1
    assert w.last_recovery_deferred_consumed == 0
    assert not w.envs[0].pending_recovery
    assert w.envs[0]._recovery_snapshot is None
    assert len(w._deferred_recoveries) == 1

    # Once ready, another resetting world consumes the deferred plan + snapshot cleanly.
    plan = (np.ones((12, 111), dtype=np.float32), np.ones(12, dtype=np.int32))
    fut0, _ = w._deferred_recoveries[0]
    fut0.done.return_value = True
    fut0.result.side_effect = lambda: plan
    w._prepare_recoveries([2], source=None)
    assert w.last_recovery_deferred_consumed == 1
    assert len(w._deferred_recoveries) == 0
    assert w.envs[2].pending_recovery
    assert w.envs[2]._precomputed_recovery is plan
    assert w.envs[2]._deferred_snapshot is snap0

    # Verify close() cancels remaining deferred futures too.
    w._eval_gpu_step.return_value = (
        torch.zeros(3, 84), torch.zeros(3, 6), None,
        torch.tensor([False, False, True]), zero, torch.tensor([False, False, True]), zero, zero, None
    )
    w.step_gpu(torch.zeros(3, 29))
    w._prepare_recoveries([2], source=None)
    assert len(w._deferred_recoveries) == 1
    leftover_fut, _ = w._deferred_recoveries[0]
    pool = w._rec_pool
    w.close()
    leftover_fut.cancel.assert_called_once()
    pool.shutdown.assert_called_once_with(wait=True)
    assert len(w._deferred_recoveries) == 0


def test_batched_recovery_fk_keep_on_device_and_paged_register(sim_root):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fk = BatchedRecoveryFK(sim_root, device)

    def make_plan(T, seed):
        rng = np.random.default_rng(seed)
        frames = np.zeros((T, 111), dtype=np.float32)
        frames[:, 39:68] = rng.normal(0.0, 0.25, size=(T, 29)).astype(np.float32)
        frames[:, 97:100] = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        frames[:, 100:104] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        frames[0, :35] = np.tile(np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], np.float32), 5)
        frames[:, 107] = 1.0
        return RecoverySkeleton(frames.copy(), np.zeros(T, dtype=np.int32))

    p_cpu = [make_plan(1, 1), make_plan(17, 2)]
    p_dev = [RecoverySkeleton(p.frames.copy(), p.phase.copy()) for p in p_cpu]

    fk.complete(p_cpu, keep_on_device=False)
    fk.complete(p_dev, keep_on_device=True)

    bank = PagedClipBank(device, raw_dim=111, page_frames=64)
    slots_cpu = bank.register_many([p.frames for p in p_cpu], static=False)
    slots_dev = bank.register_many([p.gpu_frames for p in p_dev], static=False)
    for a, b, sa, sb in zip(p_cpu, p_dev, slots_cpu, slots_dev):
        assert b.gpu_frames is not None and b.gpu_frames.device == torch.device(device)
        torch.testing.assert_close(b.gpu_frames.cpu(), torch.from_numpy(a.frames), atol=1e-6, rtol=1e-5)
        idx = torch.arange(len(a.frames), dtype=torch.int64, device=device)
        z0 = torch.tensor([0], dtype=torch.int64, device=device)
        g_cpu = bank.gather(torch.tensor([sa], dtype=torch.int64, device=device), z0, idx)[0]
        g_dev = bank.gather(torch.tensor([sb], dtype=torch.int64, device=device), z0, idx)[0]
        assert torch.equal(g_cpu, g_dev)


def test_v7_cli_defaults_and_checkpoint_cadence():
    args = v7_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."])
    cpus = os.cpu_count() or 2
    assert args.defer_recovery is True
    assert args.recovery_workers is None
    assert args.full_save_every == 200
    assert args.save_every == 50
    assert args.eval_workers == min(12, max(1, cpus // 2))

    args_off = v7_args_from_cli(
        ["--sim-root", ".", "--motion-dir", ".", "--output-dir", ".", "--no-defer-recovery", "--recovery-workers", "10"]
    )
    assert args_off.defer_recovery is False
    assert args_off.recovery_workers == 10

    with pytest.raises(ValueError):
        v7_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", ".", "--recovery-workers", "0"])
    with pytest.raises(ValueError):
        v7_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", ".", "--full-save-every", "0"])

    args_v4 = v4_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."])
    assert args_v4.full_save_every is None
