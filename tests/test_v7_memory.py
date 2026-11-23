import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch

from carrywm.replay_v4 import ClipBank, HybridClipBank, VectorReplay
from carrywm.train_v4 import _reserve_collection_headroom


def test_hybrid_gather_survives_growth_compaction_and_checkpoint():
    clips = [np.arange(21, dtype=np.float32).reshape(7, 3) + i * 100 for i in range(7)]
    bank = HybridClipBank(ClipBank(raw_dim=3, init_frames=21, init_clips=2), "cpu")
    s0, = bank.register_many(clips[:1], static=True)
    old, = bank.register_many(clips[1:2])
    # Compaction is first needed partway through this same registration call.
    # Both newly returned slots and previously selected episode slots must survive.
    u, v = bank.register_many(clips[2:4], active_ids_fn=lambda: {s0})
    assert old == v and u != v
    s1, = bank.register_many(clips[4:5], static=True)
    d1, d2 = bank.register_many(clips[5:], active_ids_fn=lambda: {u, v})
    ids = torch.tensor([s0, s1, u, d2, s0, d1, v])
    times = torch.tensor([0, 2, 5, 6, 4, 1, 2], dtype=torch.int32)
    offsets = torch.tensor([0, 1, 11])
    source = [clips[i] for i in [0, 4, 2, 6, 0, 5, 3]]
    expected = torch.from_numpy(np.stack([clip[np.minimum(t.item() + offsets.numpy(), 6)]
                                          for t, clip in zip(times, source)]))
    for actual in [bank, HybridClipBank(pickle.loads(pickle.dumps(bank.snapshot_cpu())), "cpu")]:
        assert len(actual._static_frames) == 14
        assert torch.equal(actual.gather(ids, times, offsets), expected)
        assert torch.equal(actual.gather(ids[:2], times[:2], offsets), expected[:2])
        assert torch.equal(actual.gather(ids[2:4], times[2:4], offsets), expected[2:4])
        assert torch.equal(actual.remaining(ids, times), 6 - times)


def test_device_checkpoint_migrates_without_changing_replay_or_sampled_references():
    rep = VectorReplay(4, 24, 1)
    clip = np.arange(6 * 111, dtype=np.float32).reshape(6, 111)
    static_slot = rep.register_clips([clip], static=True)
    recover_slot = rep.register_clips([clip + 1000])
    for i, slot in enumerate((static_slot, recover_slot)):
        state = torch.zeros((1, 84))
        act = torch.zeros((1, 29))
        rep.append_gpu(state, state, act, act, torch.zeros(1, 6), torch.zeros(1, 16),
                       torch.zeros(1, 141), torch.tensor([i]), slot, torch.tensor([True]))
    restored = VectorReplay.from_legacy(pickle.loads(pickle.dumps(rep)), "cpu", hybrid_clips=True)
    assert restored.ticks == rep.ticks
    for a, b in zip(restored.actor(32, 32), rep.actor(32, 32)):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_hybrid_device_residency_and_mixed_gpu_windows():
    bank = HybridClipBank(ClipBank(init_frames=320), "cuda:0")
    clips = [np.arange(106 * 111, dtype=np.float32).reshape(106, 111) + 50000*i
             for i in range(6)]
    bank.register_many(clips[:2], static=True)
    bank.register_many(clips[2:], active_ids_fn=lambda: {0, 1})
    ids = torch.tensor([4, 0, 5, 2, 1, 3], device="cuda:0")
    times = torch.tensor([0, 1, 55, 88, 104, 105], dtype=torch.int32, device="cuda:0")
    for offsets in (torch.arange(83), torch.tensor([0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50])):
        got = bank.gather(ids, times, offsets.cuda())
        expected = bank.host.gather(ids.cpu(), times.cpu(), offsets)
        assert got.is_cuda and torch.equal(got.cpu(), expected)
    assert bank.host.frames_buf.device.type == "cpu"
    assert bank.storage_metrics()["replay/clip_device_bytes"] == 2 * clips[0].nbytes


def test_headroom_releases_only_unused_cache_under_pressure():
    mib = 1024**2
    with patch.object(torch.cuda, "mem_get_info", return_value=(1200*mib, 24*1024*mib)), \
         patch.object(torch.cuda, "memory_reserved", side_effect=[4096*mib, 2000*mib, 2000*mib]), \
         patch.object(torch.cuda, "memory_allocated", return_value=2000*mib), \
         patch.object(torch.cuda, "empty_cache") as empty:
        metrics = _reserve_collection_headroom(torch.device("cuda"), 1536*mib)
        assert metrics["gpu/cache_released_before_collect_bytes"] == 2096*mib
        _reserve_collection_headroom(torch.device("cuda"), 512*mib)
        empty.assert_called_once()
