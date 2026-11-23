#!/usr/bin/env python3
"""Compare normal and mixed V7 resets plus the next real MuJoCo Warp tick on CUDA."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from carrywm.warp_sim import WarpCarryWorlds


def check(sim_root, motion_dir, device, cache_dir, worlds=6):
    args = dict(sim_root=sim_root, motion_dir=motion_dir, num_envs=worlds, seed=8263,
                device=device, cfgen_ratio=.1, cfgen_cache_dir=cache_dir, plan_workers=1,
                recovery_workers=0, v7=True, use_cuda_graph=False)
    baseline = WarpCarryWorlds(**args)
    try:
        candidate = WarpCarryWorlds(**args)
        try:
            baseline._batch_reset = False
            assert candidate._batch_reset
            result = compare(baseline, candidate)
        finally:
            candidate.close()
    finally:
        baseline.close()
    return result


def compare(a, b):
    fields = ("_qpos", "_qvel", "_mocap", "_state_gpu", "_context_gpu", "_initial_box",
              "_goal_gpu", "_goal_on_floor", "_clip_len", "_env_index", "_lifted",
              "_carried_steps", "_lost_steps", "_stable_steps")
    def snapshot(w):
        return {**{key:getattr(w, key).clone() for key in fields},
                **{key:x.clone() for key, x in w._model_views.items()}}
    n = a.num_envs
    all_ids = list(range(n))
    partial = list(range(0,n,2))
    actions = torch.linspace(-.06, .06, n*29, device=a.device).reshape(n,29)
    report = []
    for name, ids, source in (("all_npz",all_ids,"npz"), ("sparse_cfgen",partial,"cfgen"),
                              ("mixed_recovery_and_ordinary",[0,1],None)):
        if source is None:
            # Inject only a completed worker result here to isolate selection/writes.
            # The pose and velocity come from the preceding real physics step.
            for w in (a,b):
                w._sync_worlds_to_host([0])
                e = w.envs[0]
                e.pending_recovery = True
                e._precomputed_recovery = (e.clip, e.phase)
                w.envs[1].pending_recovery = False
        before = snapshot(b)
        a.reset(ids, source=source)
        b.reset(ids, source=source)
        expected, actual = snapshot(a),snapshot(b)
        rest = sorted(set(all_ids)-set(ids))
        for key, value in actual.items():
            torch.testing.assert_close(value[rest],before[key][rest],rtol=0,atol=0)
            torch.testing.assert_close(value[ids],expected[key][ids],rtol=3e-6,atol=2e-5)
        if source is None:
            assert b.envs[0].source == "recovery" and b.envs[1].source != "recovery"
            for key in ("_qpos","_qvel","_mocap",*b._model_views):
                torch.testing.assert_close(actual[key][0],before[key][0],rtol=0,atol=0)
        for i in ids:
            e, f = a.envs[i], b.envs[i]
            for key in ("source","name","index","episode","start_index","start_stage","started_holding",
                        "_lifted","_lost_steps","_stable_steps","_carried_steps","pending_recovery","recovery_rejected"):
                assert getattr(e,key)==getattr(f,key), key
            assert e.rng.bit_generator.state==f.rng.bit_generator.state
            for x,y in ((e.state,f.state),(e.context,f.context),(e.goal,f.goal),(e._initial_box,f._initial_box),
                        (e.d.qpos,f.d.qpos),(e.d.qvel,f.d.qvel),(e.clip,f.clip)):
                np.testing.assert_allclose(x,y,atol=2e-5,rtol=3e-6)
        # Make the input reference for the next step explicit, using the device frame
        # counters because active worlds can have more recent frames than their host shadow.
        targets=[]
        for w in (a,b):
            frames=w._env_index.cpu().numpy()
            targets.append(torch.as_tensor(np.stack([e.clip[min(int(t)+1,len(e.clip)-1)] for e,t in zip(w.envs,frames)]),
                                           device=w.device))
        va,ca,da,ea,ra=a.step_gpu(actions,next_ref=targets[0])
        vb,cb,db,eb,rb=b.step_gpu(actions,next_ref=targets[1])
        # Independent Warp instances may sum contact impulses in different orders.
        torch.testing.assert_close(va,vb,rtol=2e-3,atol=3e-3)
        torch.testing.assert_close(ca,cb,rtol=0,atol=0)
        np.testing.assert_array_equal(da,db)
        assert ea==eb and ra==rb
        report.append(dict(case=name, reset_worlds=len(ids), max_state_after_step=float((va-vb).abs().max()),
                           contacts_equal=True, terminal_flags_equal=True, host_rng_equal=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root",type=Path,required=True)
    parser.add_argument("--motion-dir",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path,default=os.environ.get("CARRYWM_CACHE_DIR"))
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    print(json.dumps(check(args.sim_root,args.motion_dir,args.device,args.cache_dir), indent=2))
