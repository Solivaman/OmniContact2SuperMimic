"""One batched policy call and one batched physical control step per vector tick."""
from collections import Counter
from functools import partial
from time import monotonic

import numpy as np
import torch

from .data import OFFSETS
from .replay_v4 import gather_reference


class VectorCollector:
    def __init__(self, env, learner, replay, rng):
        self.env, self.net, self.replay, self.rng = env, learner, replay, rng
        self.count = len(env.envs)
        if self.count != replay.num_envs:
            raise ValueError("Replay and simulator world counts differ")
        self.steps = 0
        self.episodes = Counter()
        self.fresh = np.ones(self.count, bool)
        self.world_is_teacher = np.zeros(self.count, bool)
        self.hist = None
        replay.begin_new_episodes()

        replay_dev = getattr(replay, "device", torch.device("cpu"))
        self._gpu_fast = (
            hasattr(env, "step_gpu")
            and learner.device.type == "cuda"
            and replay_dev.type == "cuda"
        )
        if self._gpu_fast:
            device = learner.device
            if hasattr(env, "_keep_recovery_on_device"):
                from .device_clips import PagedClipBank
                env._keep_recovery_on_device = isinstance(replay.clip_bank, PagedClipBank)
            self._warm_static_clips()
            self.state_gpu = torch.empty((self.count, 84), dtype=torch.float32, device=device)
            self.context_gpu = torch.empty((self.count, 16), dtype=torch.float32, device=device)
            self.previous_gpu = torch.zeros((self.count, 29), dtype=torch.float32, device=device)
            self.clip_ids_gpu = torch.zeros(self.count, dtype=torch.int64, device=device)
            self.frames_gpu = torch.zeros(self.count, dtype=torch.int32, device=device)
            self._offsets_gpu = torch.as_tensor(OFFSETS, dtype=torch.int64, device=device)
            self._one_offset_gpu = torch.tensor([1], dtype=torch.int64, device=device)
            self._gpu_gen = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**31 - 1)))
        else:
            self.previous = np.zeros((self.count, 29), np.float32)
            self.state = np.empty((self.count, 84), np.float32)
            self.context = np.empty((self.count, 16), np.float32)

    def _warm_static_clips(self):
        proto = self.env.envs[0]
        static_clips = [scene.frames for scene in getattr(proto, "scenes", {}).values()]
        static_clips.extend(frames for frames, _ in getattr(proto, "_plan_cache", {}).values())
        pool = getattr(proto, "cfgen_pool", None)
        if pool:
            static_clips.extend(entry.frames for entry in pool)
        if static_clips:
            self.replay.register_clips(static_clips, static=True)

    @torch.no_grad()
    def one_tick(self, teacher_prob=0., noise=0., update_normalizers=False, defer_reward=False,
                 episode_teacher_frac=0.0, *, record_reward=True):
        if self._gpu_fast:
            return self._one_tick_gpu(
                teacher_prob, noise, update_normalizers, defer_reward, episode_teacher_frac, record_reward
            )
        return self._one_tick_cpu(teacher_prob, noise, update_normalizers, record_reward)

    def _one_tick_gpu(self, teacher_prob, noise, update_normalizers, defer_reward,
                      episode_teacher_frac=0.0, record_reward=True):
        envs, net = self.env.envs, self.net
        device = net.device
        if episode_teacher_frac <= 0.0:
            self.world_is_teacher.fill(False)
        ids = np.flatnonzero(self.fresh)
        timing = {}
        has_defer = hasattr(self.env, "last_recovery_deferred_queued")
        if hasattr(self.env, "last_recovery_wait_sec"):
            timing = {"sim/reset_sec_per_tick": 0., "sim/recovery_wait_sec_per_tick": 0.}
        if has_defer:
            timing.update({
                "sim/recovery_deferred_queued_per_tick": 0.,
                "sim/recovery_deferred_consumed_per_tick": 0.,
                "sim/recovery_deferred_dropped_per_tick": 0.,
            })
        if len(ids):
            if timing:
                started = monotonic()
            self.env.reset(ids)
            if timing:
                timing["sim/reset_sec_per_tick"] = monotonic() - started
                timing["sim/recovery_wait_sec_per_tick"] = self.env.last_recovery_wait_sec
            if has_defer:
                timing["sim/recovery_deferred_queued_per_tick"] = float(self.env.last_recovery_deferred_queued)
                timing["sim/recovery_deferred_consumed_per_tick"] = float(self.env.last_recovery_deferred_consumed)
            if episode_teacher_frac > 0.0:
                k_teacher = int(round(episode_teacher_frac * self.count))
                self.world_is_teacher[ids] = ids < k_teacher
            else:
                self.world_is_teacher[ids] = False
            for i in ids:
                src = f"teacher_{envs[i].source}" if self.world_is_teacher[i] else envs[i].source
                self.episodes[f"{src}/started"] += 1
                if getattr(self.env, "stage_sampler", None) is not None and envs[i].source != "recovery":
                    self.episodes[f"stage/{src}/{envs[i].start_stage}/started"] += 1
            idx_t = torch.as_tensor(ids, dtype=torch.int64, device=device)
            self.state_gpu.index_copy_(0, idx_t, self.env._state_gpu.index_select(0, idx_t))
            self.context_gpu.index_copy_(0, idx_t, self.env._context_gpu.index_select(0, idx_t))
            self.previous_gpu[idx_t] = 0.0
            self.frames_gpu.index_copy_(0, idx_t, self.env._env_index.index_select(0, idx_t))
            clip_inputs = []
            for i in ids:
                gpu_clip = getattr(envs[i], "_clip_gpu", None)
                if gpu_clip is not None:
                    clip_inputs.append(gpu_clip)
                    envs[i]._clip_gpu = None
                else:
                    clip_inputs.append(envs[i].clip)
            new_clip_ids = self.replay.register_clips(
                clip_inputs, static=False,
                extra_active_ids=self.clip_ids_gpu if self.steps > 0 else None,
            )
            self.clip_ids_gpu.index_copy_(0, idx_t, new_clip_ids)

        fresh_t = torch.as_tensor(self.fresh, dtype=torch.bool, device=device)
        raw_win = self.replay.gather_reference(self.clip_ids_gpu, self.frames_gpu, self._offsets_gpu)
        if self.hist is None:
            self.hist = torch.zeros((self.count, 10, 141), dtype=torch.float32, device=device)

        if episode_teacher_frac > 0.0 or self.world_is_teacher.any():
            teacher_mask_np = self.world_is_teacher
            teacher_ids = np.flatnonzero(teacher_mask_np)
        elif teacher_prob > 0.0:
            teacher_mask_np = self.rng.random(self.count) < teacher_prob
            teacher_ids = np.flatnonzero(teacher_mask_np)
        else:
            teacher_mask_np = None
            teacher_ids = np.empty(0, dtype=np.int64)

        if not update_normalizers and hasattr(net, "collect_act"):
            obs, self.hist, actions = net.collect_act(
                self.state_gpu, self.previous_gpu, self.context_gpu, raw_win, self.hist, fresh_t
            )
        else:
            obs = net.geo.observation(self.state_gpu, self.previous_gpu, self.context_gpu)
            shifted = torch.cat((self.hist[:, 1:], obs[:, None]), dim=1)
            self.hist = torch.where(fresh_t[:, None, None], obs[:, None].expand(-1, 10, -1), shifted)
            goal = net.reference(self.state_gpu, raw_win)
            with net.autocast():
                actions = net.actor(goal, self.hist).float()
            if update_normalizers:
                net.actor.update_stats(goal, self.hist)
                if getattr(net, "critic", None) is not None and hasattr(net.critic, "update_stats"):
                    net.critic.update_stats(self.state_gpu, goal, self.context_gpu)

        teacher_mask_t = (
            torch.as_tensor(teacher_mask_np, dtype=torch.bool, device=device)
            if teacher_mask_np is not None else None
        )
        if noise:
            eps = torch.randn(
                actions.shape, generator=self._gpu_gen, device=device, dtype=torch.float32
            ) * noise
            if teacher_mask_t is not None:
                eps = eps.masked_fill(teacher_mask_t[:, None], 0.0)
            actions = actions + eps

        teacher_frac = 0.0
        if len(teacher_ids):
            teacher_frac = len(teacher_ids) / self.count
            t_idx = torch.as_tensor(teacher_ids, dtype=torch.int64, device=device)
            t_act = torch.as_tensor(
                np.stack([envs[i].teacher_action() for i in teacher_ids]),
                dtype=torch.float32, device=device,
            )
            actions.index_copy_(0, t_idx, t_act)

        actions = actions.clamp(-10.0, 10.0)
        if update_normalizers:
            net.wm.update_stats(self.state_gpu, actions, self.context_gpu)

        target = self.replay.gather_reference(self.clip_ids_gpu, self.frames_gpu, self._one_offset_gpu)[:, 0]
        after_gpu, contacts_gpu, done_np, event_counts, drops = self.env.step_gpu(
            actions, next_ref=target, teacher_ids=teacher_ids if len(teacher_ids) else None
        )
        if has_defer:
            timing["sim/recovery_deferred_dropped_per_tick"] = float(self.env.last_recovery_deferred_dropped)
        self.episodes.update(event_counts)
        self.replay.append_gpu(
            self.state_gpu, after_gpu, self.previous_gpu, actions, contacts_gpu,
            self.context_gpu, obs, self.frames_gpu, self.clip_ids_gpu, fresh_t,
            is_teacher=teacher_mask_t,
        )
        metrics = {"sim/teacher_fraction": teacher_frac, **timing}
        if record_reward:
            terms = net.geo.reward_terms(
                self.state_gpu, after_gpu, self.previous_gpu, actions, target, contacts_gpu
            )
            reward = sum(term.mean() for term in terms.values()) - net.args.drop_penalty * drops / self.count
            metrics["sim/reward"] = reward if defer_reward else reward.item()

        self.fresh = done_np
        self.state_gpu.copy_(after_gpu)
        self.previous_gpu.copy_(actions)
        self.frames_gpu.copy_(self.env._env_index)
        self.steps += self.count
        return metrics

    def _one_tick_cpu(self, teacher_prob, noise, update_normalizers, record_reward=True):
        envs, net, fresh = self.env.envs, self.net, self.fresh
        ids = np.flatnonzero(fresh)
        if len(ids):
            self.env.reset(ids)
            for i in ids:
                env = envs[i]
                self.state[i], self.context[i] = env.state, env.context
                self.episodes[f"{env.source}/started"] += 1
                if getattr(self.env, "stage_sampler", None) is not None and env.source != "recovery":
                    self.episodes[f"stage/{env.source}/{env.start_stage}/started"] += 1
            self.previous[ids] = 0

        device = net.device
        as_tensor = partial(torch.as_tensor, device=device, dtype=torch.float32)
        state = as_tensor(self.state)
        ctx = as_tensor(self.context)
        previous = as_tensor(self.previous)
        obs = net.geo.observation(state, previous, ctx)
        if self.hist is None:
            self.hist = obs[:, None].expand(-1, 10, -1).clone()
        else:
            self.hist = torch.cat((self.hist[:, 1:], obs[:, None]), dim=1)
            if len(ids):
                reset = torch.as_tensor(ids, device=device)
                self.hist[reset] = obs[reset, None].expand(-1, 10, -1)

        clips = [env.clip for env in envs]
        frames = np.fromiter((env.index for env in envs), dtype=np.int32, count=self.count)
        goal = net.reference(state, as_tensor(gather_reference(clips, frames, OFFSETS)))
        with net.autocast():
            actions = net.actor(goal, self.hist).float().cpu().numpy()
        teacher_ids = np.flatnonzero(self.rng.random(self.count) < teacher_prob) if teacher_prob else ()
        for i in teacher_ids:
            actions[i] = envs[i].teacher_action()
        if noise:
            actions += self.rng.normal(0, noise, actions.shape).astype(np.float32)
        np.clip(actions, -10, 10, out=actions)
        after, contacts, done, infos = self.env.step(actions)
        # Simulator owns terminal states until the next tick's masked reset.
        done = np.asarray(done, bool)
        executed = actions.copy()
        drops = 0
        for i, info in enumerate(infos):
            if "executed_action" in info:
                executed[i] = info["executed_action"]
            if done[i]:
                source = envs[i].source
                self.episodes[f"{source}/done"] += 1
                if getattr(self.env, "stage_sampler", None) is not None and source != "recovery":
                    stage = envs[i].start_stage
                    self.episodes[f"stage/{source}/{stage}/done"] += 1
                    self.episodes[f"stage/{source}/{stage}/success"] += int(info["success"])
                for event in ("success", "drop", "timeout", "recovered", "fallen"):
                    self.episodes[f"{source}/{event}"] += int(info[event])
            drops += int(info["drop"])
        action = as_tensor(executed)
        if update_normalizers:
            net.actor.update_stats(goal, self.hist)
            net.wm.update_stats(state, action, ctx)
        self.replay.append(self.state, after, self.previous, executed, contacts, self.context,
                           obs.cpu().numpy(), frames, clips, fresh)
        metrics = {"sim/teacher_fraction": len(teacher_ids) / self.count}
        if record_reward:
            target = gather_reference(clips, frames, (1,))[:, 0]
            terms = net.geo.reward_terms(state, as_tensor(after), previous, action,
                                         as_tensor(target), as_tensor(contacts))
            metrics["sim/reward"] = sum(term.mean() for term in terms.values()).item() - net.args.drop_penalty * drops / self.count
        self.fresh = done.copy()
        self.state[:] = after
        self.previous = executed
        self.steps += self.count
        return metrics
