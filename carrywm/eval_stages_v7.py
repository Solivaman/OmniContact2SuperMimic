"""Evaluate V7 from held-out reference phases, separately from frame-zero success."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .data import raw_window
from .sim import CarryEnv
from .stages_v7 import NoStageStart, SOURCES, STAGES, StageSampler
from .train import _CpuEvalNet


@torch.inference_mode()
def evaluate(checkpoint, sim_root, motion_dir, scenes=6, seed=42):
    if scenes < 1:
        raise ValueError("scenes must be positive")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if ck.get("version") != 7:
        raise ValueError("Expected a V7 checkpoint")
    cfg = ck["config"]
    net = _CpuEvalNet(sim_root, cfg["goal_mode"], cfg["drop_penalty"],
                      cfg["encoder_width"], cfg["decoder_width"], ck["models"][1])
    del ck
    env = CarryEnv(sim_root, motion_dir, seed=seed, split="val", cfgen_goal_noise_std_m=0., v7=True)
    # Explicit opt-in for this diagnostic. Normal val/test environments always start from frame 0.
    env.stage_sampler = StageSampler()
    paths = np.random.default_rng(seed).permutation(env.paths)[:scenes]
    records = []
    groups = defaultdict(list)
    for path in paths:
        env.paths = (path,)
        for source in SOURCES:
            for stage in STAGES:
                row = dict(scene=path.name, source=source, stage=stage)
                try:
                    env.reset(source, stage=stage)
                except NoStageStart:
                    row["eligible"] = False
                else:
                    row.update(eligible=True, start_frame=env.index, started_holding=env.started_holding)
                    history = None
                    contact_after_start = False
                    while True:
                        state = torch.from_numpy(env.state[None])
                        context = torch.from_numpy(env.context[None])
                        previous = torch.from_numpy(env.last_action[None])
                        fk = net.geo.forward_kinematics(state)
                        obs = net.geo.observation(state, previous, context, fk=fk)
                        history = (obs[:, None].expand(-1, 10, -1).clone() if history is None
                                   else torch.cat((history[:, 1:], obs[:, None]), dim=1))
                        raw = torch.from_numpy(raw_window(env.clip, env.index)[None])
                        action = net.actor(net.reference(state, raw, fk=fk), history)[0]
                        _, contacts, done, info = env.step(action.clamp(-10., 10.).numpy())
                        contact_after_start |= bool(contacts[2:4].all())
                        if done:
                            row.update(steps=env.index - row["start_frame"],
                                       bilateral=contact_after_start,
                                       new_lift=bool(info["lifted"]) and not env.started_holding,
                                       **{k: bool(info[k]) for k in ("success", "drop", "fallen", "timeout")})
                            break
                records.append(row)
                groups[(source, stage)].append(row)
    summary = {}
    for (source, stage), rows in groups.items():
        valid = [r for r in rows if r["eligible"]]
        key = f"{source}/{stage}"
        summary[key] = dict(requested=len(rows), eligible=len(valid),
                            **{k: sum(int(r[k]) for r in valid) for k in
                               ("started_holding", "bilateral", "new_lift", "drop", "fallen", "success")})
    return {"protocol": "stage_start_diagnostic", "seed": seed, "summary": summary, "episodes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "sim-root", "motion-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
