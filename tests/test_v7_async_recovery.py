"""Test V7's GPU-step recovery scheduling with deterministic terminal states."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from carrywm.warp_sim import _Episode, _UNSET, _recovery_worker, WarpCarryWorlds


def world_harness(count, *, prefetch=True):
    world = WarpCarryWorlds.__new__(WarpCarryWorlds)
    world.num_envs = count
    world.v7 = prefetch
    world._gpu_recovery = False
    world._recovery_fk = None
    if prefetch:
        world.last_recovery_wait_sec = 0.
    world._recovery_futures = {} if prefetch else None
    world._recovery_workers = 8
    world._rec_pool = Mock()
    # Calling result in step_gpu would block all GPU work until the plan returns.
    def queued(*_):
        future = Mock(spec=Future)
        future.done.return_value = False
        future.result.side_effect = AssertionError("plan was read before reset")
        return future
    world._rec_pool.submit.side_effect = queued
    world.teacher_history = False
    world.stage_sampler = None
    world._qpos = torch.arange(count*43, dtype=torch.float32).reshape(count, 43)
    world._qvel = torch.zeros(count, 41)
    world._env_index = torch.arange(10, 10 + count)
    world._overflow_torch = torch.zeros(count, dtype=torch.int32)
    world._packed_flags = torch.empty(count+1, 5, dtype=torch.uint8)
    world._step_physics = lambda action: action
    world._sync_worlds_to_host = Mock(wraps=world._sync_worlds_to_host)
    world.envs = []
    for i in range(count):
        e = _Episode.__new__(_Episode)
        e.source, e.root, e.bp, e.v7 = "npz", Path("/scene"), 36, prefetch
        e._precomputed_recovery, e.pending_recovery = _UNSET, False
        e.half, e.goal = np.full(3, i+.15), np.full(3, i+.2)
        e.d = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41))
        e.rng = np.random.default_rng(i)
        world.envs.append(e)
    return world


def drop_tick(world, drops, fallen=()):
    n = world.num_envs
    drop, fall = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    drop[list(drops)] = True
    fall[list(fallen)] = True
    zero = torch.zeros_like(drop)
    world._eval_gpu_step = Mock(return_value=(
        torch.zeros(n, 84), torch.zeros(n, 6), None, drop | fall, zero, drop, zero, fall, None,
    ))
    return world.step_gpu(torch.zeros(n, 29))


def test_submit_at_terminal_use_unchanged_snapshot_and_consume_in_reset_order():
    w = world_harness(5)
    w.envs[4].source = "recovery"
    _, _, done, _, _ = drop_tick(w, [0, 1, 2, 4], fallen=[1])
    assert done.tolist() == [True, True, True, False, True]
    assert list(w._recovery_futures) == [0, 2]
    w._sync_worlds_to_host.assert_called_once_with([0, 2])
    for i, call in zip([0, 2], w._rec_pool.submit.call_args_list):
        fn, root, pelvis, box, half, goal, v7 = call.args
        assert fn is _recovery_worker and root == "/scene" and v7
        np.testing.assert_array_equal(pelvis, w._qpos[i, :7].numpy())
        np.testing.assert_array_equal(box, w._qpos[i, 36:].numpy())
        np.testing.assert_array_equal(half, w.envs[i].half)
        np.testing.assert_array_equal(goal, w.envs[i].goal)
        for submitted, stored in ((pelvis, w.envs[i].d.qpos), (box, w.envs[i].d.qpos),
                                  (half, w.envs[i].half), (goal, w.envs[i].goal)):
            assert not np.shares_memory(submitted, stored)
    order = []
    plan = (np.ones((10, 111)), np.ones(10))
    def result(i):
        order.append(i)
        return plan if i == 2 else None
    for i, fut in w._recovery_futures.items():
        fut.result.side_effect = lambda i=i: result(i)
    w._prepare_recoveries([2, 0], source=None)
    assert order == [2, 0] and not w._recovery_futures
    assert w.envs[2]._try_recovery() and w.envs[2].clip is plan[0]
    assert not w.envs[0]._try_recovery()
    assert w.envs[0]._precomputed_recovery is _UNSET
    w._sync_worlds_to_host.assert_called_once()
    assert all(w.envs[i].rng.random() == np.random.default_rng(i).random() for i in (0, 2))


def test_single_drop_prefetched_and_explicit_or_partial_reset_discards_only_its_job():
    w = world_harness(2)
    drop_tick(w, [0])
    first = w._recovery_futures[0]
    w._prepare_recoveries([1], source="npz")
    assert w._recovery_futures[0] is first
    w._prepare_recoveries([0], source="cfgen")
    first.cancel.assert_called_once()
    first.result.assert_not_called()
    assert not w._recovery_futures and w.envs[0]._precomputed_recovery is _UNSET
    # Simulate a later independent episode in the very same world.
    drop_tick(w, [0])
    second = w._recovery_futures[0]
    assert second is not first
    second.result.side_effect = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        w._prepare_recoveries([0], source=None)
    pool = w._rec_pool
    w.close()
    w.close()
    second.cancel.assert_called_once()
    pool.shutdown.assert_called_once_with(wait=True)
    assert not w._recovery_futures


def test_legacy_gpu_does_not_start_worker_on_single_drop():
    w = world_harness(2, prefetch=False)
    drop_tick(w, [0])
    w._prepare_recoveries([0], source=None)
    w._rec_pool.submit.assert_not_called()
    assert w.envs[0]._precomputed_recovery is _UNSET


def test_reports_only_consumer_wait_and_resets_timer_per_reset(monkeypatch):
    w = world_harness(3)
    drop_tick(w, [0, 1])
    slow, ready = w._recovery_futures.values()
    slow.result.side_effect = ready.result.side_effect = None
    ready.done.return_value = True
    clock = iter([8., 8.4])
    monkeypatch.setattr("carrywm.warp_sim.perf_counter", lambda: next(clock))
    w._prepare_recoveries([1, 0], source=None)
    assert w.last_recovery_wait_sec == pytest.approx(.4)
    w.reset([])
    assert w.last_recovery_wait_sec == 0.


def test_explicit_serial_fallback_times_only_the_recovery_solver(monkeypatch):
    w = world_harness(1)
    w._recovery_workers = 0
    w._recovery_futures = None
    drop_tick(w, [0])
    e = w.envs[0]
    e._try_recovery = Mock(return_value=False)
    clock = iter([4., 5.25])
    monkeypatch.setattr("carrywm.warp_sim.perf_counter", lambda: next(clock))
    w._prepare_recoveries([0], source=None)
    assert w.last_recovery_wait_sec == 1.25 and e._precomputed_recovery is None
    w._rec_pool.submit.assert_not_called()


def test_gpu_recovery_completes_selected_skeletons_together_and_preserves_other_results():
    from carrywm.recovery_gpu import prepare_recovery, RecoverySkeleton
    w = world_harness(5)
    w._gpu_recovery = True
    w._recovery_fk = Mock()
    drop_tick(w, range(5))
    for call in w._rec_pool.submit.call_args_list:
        assert call.args[0] is prepare_recovery
    def plan(value):
        return RecoverySkeleton(np.full((3,111), value, np.float32), np.arange(3))
    one, two = plan(1), plan(2)
    fallback = (np.ones((6,111),np.float32),np.arange(6))
    for i, value in enumerate((one, None, two, fallback)):
        job = w._recovery_futures[i]
        job.result.side_effect = None
        job.result.return_value = value
    def complete(batch):
        assert len(batch) == 2 and batch[0] is two and batch[1] is one
        for skeleton in batch:
            skeleton.frames[:, :35] = 7
    w._recovery_fk.complete.side_effect = complete
    w._prepare_recoveries([2,3,1,0], None)
    assert w._recovery_fk.complete.call_count == 1
    assert w.envs[0]._try_recovery() and w.envs[0].clip is one.frames
    assert w.envs[2]._try_recovery() and w.envs[2].phase is two.phase
    assert (w.envs[2].clip[:, :35] == 7).all()
    assert w.envs[3]._precomputed_recovery is fallback
    assert w.envs[1]._precomputed_recovery is None
    assert set(w._recovery_futures) == {4}
    unused = w._recovery_futures[4]
    w._prepare_recoveries([4], "npz")
    unused.cancel.assert_called_once()
    unused.result.assert_not_called()
    assert w._recovery_fk.complete.call_count == 1


def test_gpu_recovery_without_pool_runs_serial_ik_and_retries_failed_batch(monkeypatch):
    from carrywm import recovery_gpu
    w = world_harness(2)
    w._gpu_recovery, w._recovery_workers, w._recovery_futures = True, 0, None
    w._recovery_fk = Mock()
    drop_tick(w,[0,1])
    skel = recovery_gpu.RecoverySkeleton(np.zeros((2,111),np.float32),np.arange(2))
    prepare = Mock(side_effect=[skel,None])
    monkeypatch.setattr(recovery_gpu,"prepare_recovery",prepare)
    w._recovery_fk.complete.side_effect = RuntimeError("device failed")
    with pytest.raises(RuntimeError,match="device failed"):
        w._prepare_recoveries([0,1],None)
    assert prepare.call_count == 2 and w.envs[0]._precomputed_recovery is skel
    assert w.envs[1]._precomputed_recovery is None
    w._rec_pool.submit.assert_not_called()
    w._recovery_fk.complete.side_effect = None
    w._prepare_recoveries([0,1],None)
    assert prepare.call_count == 2 and w._recovery_fk.complete.call_count == 2
    assert w.envs[0]._precomputed_recovery[0] is skel.frames
    # Explicit ordinary reset discards an old CPU or GPU result even with no pool.
    w._prepare_recoveries([0],"npz")
    assert w.envs[0]._precomputed_recovery is _UNSET


@pytest.mark.parametrize("missing_at", ["specialize","generate"])
def test_recovery_gpu_unsupported_upstream_field_returns_original_cpu_plan(monkeypatch,missing_at):
    from carrywm import recovery_gpu
    root = Path("/scene")
    quat = np.array([0.,0.,0.,1.,0.,0.,0.])
    half,goal = np.ones(3)*.1,np.zeros(3)
    cpu_plan = (np.ones((2,111),np.float32),np.arange(2))
    fallback = Mock(return_value=cpu_plan)
    monkeypatch.setattr(recovery_gpu,"_v7_recovery_plan",fallback)
    if missing_at == "specialize":
        monkeypatch.setattr(recovery_gpu,"_skeleton_type",Mock(side_effect=KeyError("alternate_builder")))
    else:
        monkeypatch.setattr(recovery_gpu,"_skeleton_type",lambda _:lambda:SimpleNamespace(cfg={}))
        monkeypatch.setattr(recovery_gpu,"_probe",lambda *_:SimpleNamespace(pads=np.zeros((2,3))))
        monkeypatch.setattr(recovery_gpu,"generate_clip",Mock(side_effect=KeyError("extra_link")))
    out = recovery_gpu.prepare_recovery(root,quat,quat,half,goal)
    assert out is cpu_plan
    fallback.assert_called_once()
