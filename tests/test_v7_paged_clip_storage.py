"""Storage regressions for fixed-address device pages, live slots and checkpoint migration."""
import numpy as np
import pytest
import torch

from carrywm.data import OFFSETS
from carrywm.device_clips import PagedClipBank
from carrywm.replay_v4 import ClipBank, VectorReplay


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def clip(frames, seed):
    x = np.random.default_rng(seed).normal(size=(frames,111)).astype(np.float32)
    # Preserve FP32 signed zero and all signed quaternion components through copies.
    x[::2,0] = np.float32(-0.0)
    x[1::2,0] = np.float32(0.0)
    return x


def check_windows(bank, ids, starts, source, device):
    at = torch.tensor(starts,device=device,dtype=torch.int32)
    slots = torch.tensor(ids,device=device,dtype=torch.int64)
    for off in ([1],OFFSETS.tolist(),list(range(83))):
        offsets = torch.tensor(off,device=device)
        actual = bank.gather(slots,at,offsets)
        expected = np.stack([a[np.minimum(int(t)+np.asarray(off),len(a)-1)] for a,t in zip(source,starts)])
        assert actual.device.type == torch.device(device).type
        assert torch.equal(actual.detach().cpu().contiguous().view(torch.int32),torch.from_numpy(expected).view(torch.int32))
    remaining = torch.tensor([max(0,len(a)-1-t) for a,t in zip(source,starts)],dtype=torch.float32)
    assert torch.equal(bank.remaining(slots,at).cpu(),remaining)


@pytest.mark.parametrize("device", DEVICES)
def test_growth_preserves_page_storage_and_reuses_reclaimed_slots(device):
    bank = PagedClipBank(device=device,raw_dim=111,page_frames=64)
    static, kept, stale, long_clip, replacement = [clip(n,i) for i,n in enumerate((46,74,39,228,39))]
    fixed, = bank.register_many([static],static=True)
    keep, dead = bank.register_many([kept,stale])
    old_pages = [p.data_ptr() for p in bank.pages]
    large, = bank.register_many([long_clip])
    grown = [p.data_ptr() for p in bank.pages]
    assert len(grown)>len(old_pages)
    assert grown[:len(old_pages)]==old_pages
    check_windows(bank,[fixed,large,keep,dead],[45,63,31,0],[static,long_clip,kept,stale],device)

    assert bank.reclaim({keep,large})==1
    reuse, = bank.register_many([replacement])
    assert reuse==dead
    assert len(bank.pages)<=len(grown)
    assert bank.register_many([static.copy()],static=True)==[fixed]
    check_windows(bank,[keep,reuse,large,large,fixed],[72,0,127,202,40],
                  [kept,replacement,long_clip,long_clip,static],device)
    metrics = bank.storage_metrics()
    assert metrics["replay/clip_frames_used"]==sum(len(c) for c in (static,kept,long_clip,replacement))
    assert metrics["replay/clip_pages"]==len(bank.pages)
    if torch.device(device).type=="cuda":
        assert metrics["replay/clip_device_bytes"]==len(bank.pages)*64*111*4
        assert all(p.is_cuda for p in bank.pages)


@pytest.mark.parametrize("device", DEVICES)
def test_replay_checkpoint_roundtrip_keeps_sparse_slot_ids_and_actual_frames(tmp_path,device):
    static, stale, active = clip(101,1),clip(65,2),clip(200,3)
    host = ClipBank(device="cpu",raw_dim=111,init_frames=512)
    fixed, = host.register_many([static],static=True)
    discard, live = host.register_many([stale,active])
    bank = PagedClipBank.from_host(host,device,page_frames=64)
    assert bank.reclaim({live})==1
    replay = VectorReplay(12,19,2)
    states = torch.zeros((2,84))
    acts = torch.zeros((2,29))
    for t in range(5):
        replay.append_gpu(states,states,acts,acts,torch.zeros((2,6)),torch.zeros((2,16)),
                          torch.zeros((2,141)),torch.tensor([t,61+t]),torch.tensor([fixed,live]),
                          torch.tensor([t==0,t==0]))
    # Save a CPU-normalized bank as in V7's checkpoint writer; leave the hole
    # at 'discard' to detect any slot renumbering during serialization.
    replay.clip_bank = bank.snapshot_cpu()
    path = tmp_path/"actual_replay.pt"
    torch.save({"replay":replay},path)
    loaded = torch.load(path,mmap=True,weights_only=False,map_location="cpu")["replay"]
    assert loaded.ticks==5 and loaded.clip_bank._slot_lengths[discard]==0
    restored = PagedClipBank.from_host(loaded.clip_bank,device,page_frames=64)
    populated = loaded.clip_ids[:loaded.ticks].flatten().tolist()
    starts = loaded.frames[:loaded.ticks].flatten().tolist()
    assert populated==[fixed,live]*5
    check_windows(restored,populated,starts,[static,active]*5,device)
    assert restored.register_many([static.copy()],static=True)==[fixed]
    fresh, = restored.register_many([clip(50,4)])
    assert fresh==discard
    check_windows(restored,[live,fixed],[63,96],[active,static],device)


def two_page_budget():
    # Pages and the per-slot pointer tables are charged at CUDA's 512-byte
    # allocation granularity. The initial 64-slot table costs 1,024 bytes.
    return 2 * ((64 * 111 * 4 + 511) // 512) * 512 + 1024 + 3 * 512


@pytest.mark.parametrize("device", DEVICES)
def test_budget_pressure_reclaims_an_inactive_slot_before_spilling(device):
    import gc
    import weakref

    budget = two_page_budget()
    bank = PagedClipBank(device, page_frames=64, max_device_bytes=budget)
    old, retained, new = clip(64,101),clip(64,102),clip(64,103)
    obsolete, live = bank.register_many([old,retained])
    weak_old = weakref.ref(old)
    del old
    gc.collect()
    assert weak_old() is None
    scans = []
    def active():
        scans.append(True)
        return {live}
    replacement, = bank.register_many([new], active_ids_fn=active)
    assert scans and replacement==obsolete
    metrics = bank.storage_metrics()
    assert metrics["replay/clip_spill_slots"]==0
    assert metrics["replay/clip_accounted_device_bytes"]<=metrics["replay/clip_device_budget_bytes"]==budget
    assert metrics["replay/clip_pages"]==2
    assert metrics["replay/clip_frames_used"]==128
    check_windows(bank,[live,replacement,live],[63,3,1],[retained,new,retained],device)


@pytest.mark.parametrize("device", DEVICES)
def test_budget_mixed_new_spills_and_cross_page_gathers_remain_exact_after_mmap(tmp_path,device):
    budget = two_page_budget()
    bank = PagedClipBank(device,page_frames=64,max_device_bytes=budget)
    direct_a,direct_b,host_a,host_b = [clip(n,i+101) for i,n in enumerate((80,32,109,71))]
    gpu_a,gpu_b = bank.register_many([direct_a,direct_b])
    pointers = [page.data_ptr() for page in bank.pages]
    # The second insertion occurs under pressure too: host_a must remain live
    # even though none of this registration call is in the replay ring yet.
    spill_a,spill_b = bank.register_many([host_a,host_b],active_ids_fn=lambda:{gpu_a,gpu_b})
    assert len({gpu_a,gpu_b,spill_a,spill_b})==4
    assert [page.data_ptr() for page in bank.pages]==pointers
    metrics = bank.storage_metrics()
    assert metrics["replay/clip_spill_slots"]==2
    assert metrics["replay/clip_spill_frames"]==len(host_a)+len(host_b)
    assert metrics["replay/clip_spill_bytes"]==host_a.nbytes+host_b.nbytes
    assert metrics["replay/clip_device_budget_bytes"]==budget
    assert metrics["replay/clip_accounted_device_bytes"]<=budget
    if torch.device(device).type=="cuda":
        assert metrics["replay/clip_host_bytes"]==metrics["replay/clip_spill_bytes"]
        assert (metrics["replay/clip_device_bytes"]+metrics["replay/clip_device_index_bytes"]
                <=metrics["replay/clip_accounted_device_bytes"])

    selectors = [
        ([gpu_a,gpu_b,gpu_a],[63,31,28],[direct_a,direct_b,direct_a]),
        ([spill_b,spill_a,spill_a],[67,106,0],[host_b,host_a,host_a]),
        ([gpu_a,spill_b,gpu_b,spill_a],[63,70,0,97],[direct_a,host_b,direct_b,host_a]),
    ]
    for ids,starts,source in selectors:
        check_windows(bank,ids,starts,source,device)
    path = tmp_path/"mixed.pt"
    torch.save(bank.snapshot_cpu(),path)
    snapshot = torch.load(path,mmap=True,weights_only=False,map_location="cpu")
    loaded = PagedClipBank.from_host(snapshot,device,page_frames=64,max_device_bytes=budget)
    assert loaded.storage_metrics()["replay/clip_spill_slots"]==2
    for ids,starts,source in selectors:
        check_windows(loaded,ids,starts,source,device)

    assert bank.reclaim({gpu_a,gpu_b,spill_b})==1
    metrics = bank.storage_metrics()
    assert metrics["replay/clip_spill_slots"]==1
    assert metrics["replay/clip_spill_bytes"]==host_b.nbytes
    check_windows(bank,[spill_b,gpu_a],[67,63],[host_b,direct_a],device)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_budget_spills_payloads_and_indexes_without_device_allocation(device):
    bank = PagedClipBank(device,page_frames=64,max_device_bytes=0)
    x,y = clip(15,205),clip(99,206)
    a, = bank.register_many([x],static=True)
    b, = bank.register_many([y],active_ids_fn=lambda:{a})
    metrics = bank.storage_metrics()
    assert (metrics["replay/clip_device_budget_bytes"]==0 and metrics["replay/clip_accounted_device_bytes"]==0)
    assert metrics["replay/clip_pages"]==0 and metrics["replay/clip_spill_slots"]==2
    assert metrics["replay/clip_spill_bytes"]==x.nbytes+y.nbytes
    check_windows(bank,[a,b,b],[14,3,98],[x,y,y],device)


@pytest.mark.parametrize("device", DEVICES)
def test_vector_replay_uses_budget_for_init_snapshot_and_from_legacy(tmp_path,device):
    # One full production page fits with its tables, but there is no room for a
    # second 512-byte per-resident-clip table. Pin the first clip with a real
    # actor-visible replay row before registering the new, overflowing clip.
    budget = 65536*111*4+1024+512
    rep = VectorReplay(12,7,2,device=device,clip_storage="device",clip_max_device_bytes=budget)
    assert isinstance(rep.clip_bank,PagedClipBank)
    x,y = clip(81,301),clip(95,302)
    a = int(rep.register_clips([x],static=True)[0].item())
    def append(slots,starts,tick):
        state = torch.zeros((2,84),device=device)
        state[:,0] = torch.arange(2,device=device)+2*tick
        act = torch.zeros((2,29),device=device)
        rep.append_gpu(state,state,act,act,torch.zeros((2,6),device=device),torch.zeros((2,16),device=device),
                       torch.zeros((2,141),device=device),torch.tensor(starts,device=device),
                       torch.tensor(slots,device=device),torch.tensor([tick==0]*2,device=device))
    append([a,a],[3,77],0)
    b = int(rep.register_clips([y])[0].item())
    append([a,b],[4,48],1)
    append([a,b],[5,49],2)
    metrics=rep.clip_bank.storage_metrics()
    assert metrics["replay/clip_spill_slots"]==1
    assert metrics["replay/clip_device_budget_bytes"]==budget
    assert metrics["replay/clip_accounted_device_bytes"]<=budget
    snap=rep.snapshot_cpu()
    assert isinstance(snap.clip_bank,ClipBank) and snap.device.type=="cpu"
    path=tmp_path/"replay_with_budget.pt"
    torch.save({"replay":snap},path)
    loaded=torch.load(path,mmap=True,weights_only=False,map_location="cpu")["replay"]
    restored=VectorReplay.from_legacy(loaded,device,clip_storage="device",clip_max_device_bytes=budget)
    assert isinstance(restored.clip_bank,PagedClipBank)
    assert restored.clip_bank.storage_metrics()["replay/clip_spill_slots"]==1
    assert restored.clip_bank.storage_metrics()["replay/clip_device_budget_bytes"]==budget
    assert restored.ticks==3
    ids=restored.clip_ids[:3].cpu().flatten().tolist()
    starts=restored.frames[:3].cpu().flatten().tolist()
    assert ids==[a,a,a,b,a,b] and starts==[3,77,4,48,5,49]
    source=[x,x,x,y,x,y]
    check_windows(restored.clip_bank,ids,starts,source,device)
    state, _, _, _, raw, remaining = restored.actor(16,32)
    chosen=state[:,0].long().cpu().tolist()
    expected=np.stack([source[i][np.minimum(starts[i]+np.arange(83),len(source[i])-1)] for i in chosen])
    assert torch.equal(raw.cpu().contiguous().view(torch.int32),torch.from_numpy(expected).view(torch.int32))
    assert remaining.cpu().tolist()==[len(source[i])-1-starts[i] for i in chosen]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_gather_with_no_host_spills_avoids_cpu_reads_of_query_tensors(monkeypatch):
    bank = PagedClipBank("cuda:0",page_frames=64,max_device_bytes=two_page_budget())
    x = clip(81,7)
    direct, = bank.register_many([x])
    ids = torch.tensor([direct,direct],device=bank.device)
    frames = torch.tensor([60,79],device=bank.device)
    offsets = torch.arange(83,device=bank.device)
    expected = bank.gather(ids,frames,offsets)
    expected_remaining = bank.remaining(ids,frames)
    # Touch the overflow route, then remove the stale host-only clip. A bank that
    # returns to zero spills must go back to issuing its original device gather.
    overflow, = bank.register_many([clip(200,9)],active_ids_fn=lambda:{direct})
    assert overflow!=direct and bank.storage_metrics()["replay/clip_spill_slots"]==1
    assert bank.reclaim({direct})==1
    assert bank.storage_metrics()["replay/clip_spill_slots"]==0

    def avoid_host_read(method):
        def guarded(tensor,*args,**kwargs):
            if tensor.is_cuda:
                raise AssertionError(f"zero-spill fast path called Tensor.{method.__name__} on CUDA data")
            return method(tensor,*args,**kwargs)
        return guarded
    original_to = torch.Tensor.to
    def avoid_host_transfer(tensor,*args,**kwargs):
        target = kwargs.get("device",args[0] if args else None)
        if torch.is_tensor(target):
            target=target.device
        if tensor.is_cuda and isinstance(target,(torch.device,str)) and torch.device(target).type=="cpu":
            raise AssertionError("zero-spill fast path transferred a query or result to CPU")
        return original_to(tensor,*args,**kwargs)
    with monkeypatch.context() as probe:
        for name in ("cpu","tolist","item","numpy"):
            probe.setattr(torch.Tensor,name,avoid_host_read(getattr(torch.Tensor,name)))
        probe.setattr(torch.Tensor,"to",avoid_host_transfer)
        actual=bank.gather(ids,frames,offsets)
        remaining=bank.remaining(ids,frames)
    assert torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    assert torch.equal(remaining,expected_remaining)
