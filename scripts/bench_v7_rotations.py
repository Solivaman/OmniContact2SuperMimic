"""Compare V7 compiled tracking geometry and its input gradient on the same CUDA batch."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import torch

from carrywm.model import CarryGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--reps', type=int, default=16)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True  # Match the V7 trainer.
    torch.manual_seed(42)
    device = torch.device('cuda')
    state = torch.randn(args.batch, 84, device=device)
    raw = torch.randn(args.batch, 11, 111, device=device)
    ctx = torch.rand(args.batch, 16, device=device) + .1
    prev = torch.randn(args.batch, 29, device=device)
    geos = [CarryGeometry(args.sim_root, box_goal=True, fused_rotations=f).to(device)
            for f in (False, True)]

    def outputs(geo, state, prev=prev, ctx=ctx, raw=raw):
        fk = geo.forward_kinematics(state)
        return (fk, geo.observation(state, prev, ctx, fk), geo.reference(state, raw, fk),
                geo.palms_in_box(state, ctx, fk))

    def bind(geo):
        def geometry(s):
            return outputs(geo, s)
        return torch.compile(geometry, fullgraph=True)

    compiled = [bind(g) for g in geos]
    grads = []
    values = []
    with torch.no_grad():
        weights = [torch.randn_like(v) for v in outputs(geos[0], state)]
    def forward_backward(fn):
        s = state.detach().requires_grad_(True)
        vals = fn(s)
        loss = sum((v * w).mean() for v,w in zip(vals, weights))
        return vals, torch.autograd.grad(loss, s)[0]
    for fn in compiled:
        vals, grad = forward_backward(fn)
        values.append([v.detach() for v in vals])
        grads.append(grad.detach())
    names = ('link_transforms','observation','reference','palm_features')
    errors = dict()
    for name, left, right in zip(names, *values):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=1e-5)
        errors[name] = (left-right).abs().max().item()
    # TF32 in the production matmul backward can round differently from scalar
    # FMA. Compare each gradient with a separate FP64 eager geometry oracle.
    geo64 = CarryGeometry(args.sim_root, box_goal=True).to(device, torch.float64)
    s64 = state.double().requires_grad_(True)
    oracle_values = outputs(geo64, s64, prev.double(), ctx.double(), raw.double())
    oracle_grad, = torch.autograd.grad(sum((v*w.double()).mean()
                                           for v,w in zip(oracle_values, weights)), s64)
    reference_norm = oracle_grad.norm()
    rel = [((v.double()-oracle_grad).norm()/reference_norm).item() for v in grads]
    errors['baseline_gradient_rel_l2_vs_fp64'], errors['fused_gradient_rel_l2_vs_fp64'] = rel
    assert rel[1] < 1e-3 and rel[1] <= max(1e-5, 1.1*rel[0]), rel
    errors['input_gradient'] = (grads[0]-grads[1]).abs().max().item()
    reports = {}
    for train in (False, True):
        times = [[], []]
        with torch.set_grad_enabled(train):
            for repeat in range(args.reps + 3):
                for j in ((0,1) if repeat%2==0 else (1,0)):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    if train:
                        forward_backward(compiled[j])
                    else:
                        compiled[j](state)
                    torch.cuda.synchronize()
                    if repeat>=3:
                        times[j].append(1000*(time.perf_counter()-begin))
        reports['forward_backward' if train else 'forward'] = {
            'baseline_median_ms':median(times[0]),'fused_median_ms':median(times[1]),
            'baseline_ms':times[0], 'fused_ms':times[1]}
    report = dict(batch=args.batch, repetitions=args.reps, max_abs_error=errors, timing=reports,
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                  tf32=torch.backends.cuda.matmul.allow_tf32)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report['max_abs_error'], {k:(v['baseline_median_ms'],v['fused_median_ms'])
                                     for k,v in reports.items()})

if __name__=='__main__':
    main()
