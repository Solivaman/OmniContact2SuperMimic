"""Train the carry-box policy on 1024 on-policy MuJoCo Warp worlds."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import train as base
from . import train_v3
from .v3 import LearnerV3
from .v6 import teacher_schedule

NUM_ENVS = 1024
ACTOR_BATCH = 2048
REPLAY_RATIO = 16
WM_SAMPLES = train_v3.WM_UPDATES * train_v3.WM_BATCH * base.WM_STEPS


class UpdateClock:
    """Integer accounting for a fraction of an update per vector control step."""

    def __init__(self, num_envs):
        self.increment = num_envs * REPLAY_RATIO
        self.credit = 0

    def tick(self):
        updates, self.credit = divmod(self.credit + self.increment, WM_SAMPLES)
        return updates

    def ticks_until_update(self):
        return (WM_SAMPLES - self.credit + self.increment - 1) // self.increment


def _cpu_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return type(obj)((k, _cpu_clone(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [_cpu_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_clone(v) for v in obj)
    return obj


class AsyncCheckpointer:
    """Snapshot checkpoints onto CPU and write them atomically in a background thread."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._pool = ThreadPoolExecutor(max_workers=1) if self.enabled else None
        self._future = None

    def wait(self):
        if self._future is not None:
            fut, self._future = self._future, None
            fut.result()

    def save(self, payload_fn, path):
        self.wait()
        payload = payload_fn()
        if self._pool is None:
            base.atomic_save(payload, path)
        else:
            self._future = self._pool.submit(base.atomic_save, payload, path)

    def close(self):
        try:
            self.wait()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None


def configure(parser):
    train_v3.configure(parser)
    parser.description = __doc__
    cpus = os.cpu_count() or 2
    parser.add_argument('--num-envs', type=int, default=NUM_ENVS)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                        help='Compile PyTorch GPU hotspots with torch.compile')
    parser.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=True,
                        help='Capture MuJoCo Warp physics steps into a CUDA Graph')
    parser.add_argument('--async-save', action=argparse.BooleanOptionalAction, default=True,
                        help='Write checkpoints asynchronously in a background thread')
    parser.add_argument('--cfgen-cache-dir', type=Path, default=None,
                        help='Directory for caching deterministic zero-noise CF-Gen plans')
    parser.add_argument('--eval-workers', type=int, default=min(6, max(1, cpus // 4)),
                        help='CPU worker processes for held-out MuJoCo evaluation')
    parser.add_argument('--cpu-threads', type=int, default=min(4, cpus),
                        help='PyTorch intra-op CPU threads for the main process')
    parser.add_argument('--full-save-every', type=int, default=None,
                        help='Write full resumable checkpoint (including replay/optimizers) every N iterations; '
                             'intermediate --save-every points write lightweight weights.pt')
    parser.set_defaults(batch_size=ACTOR_BATCH, actor_microbatch=ACTOR_BATCH,
                        checkpoint_encoder=False, critic_chunk=16384,
                        warmup_steps=NUM_ENVS * base.WM_STEPS,
                        cfgen_goal_noise_std_m=0., collect_steps=5, eval_episodes=3)


def args_from_cli(argv=None, *, configure_parser=configure, world_count=NUM_ENVS):
    args = base.args_from_cli(argv, configure=configure_parser)
    if (args.num_envs != world_count or args.collect_steps != UpdateClock(world_count).ticks_until_update()
            or args.wm_batch_size != train_v3.WM_BATCH
            or args.batch_size != ACTOR_BATCH or args.auto_batch):
        raise ValueError(f'Use {world_count} physical worlds, 10 × 1024 WM windows and 2048 actor rollouts')
    if (args.cfgen_goal_noise_std_m != 0 or args.warmup_teacher or args.teacher_probability != 0
            or args.wm_pretrain != 0 or not 1 <= args.actor_microbatch <= ACTOR_BATCH
            or args.critic_chunk < 1):
        raise ValueError('The vector trainer uses on-policy warmup, zero CF-Gen goal noise and positive physical minibatches')
    if args.eval_workers < 1 or args.cpu_threads < 1:
        raise ValueError('eval-workers and cpu-threads must be positive')
    if getattr(args, 'full_save_every', None) is not None and args.full_save_every < 1:
        raise ValueError('full-save-every must be positive')
    if args.capacity // args.num_envs < base.WM_STEPS or torch.device(args.device).type != 'cuda':
        raise ValueError('The vector trainer requires CUDA and room for at least eight transitions per world')
    return args


def train(args, *, version=4, learner_type=LearnerV3):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args, version, learner_type)


def check_resume(ck, args, version):
    if ck['version'] != version:
        raise ValueError(f"Cannot resume V{ck['version']} using V{version}")
    if ck['iteration'] >= args.iterations:
        raise ValueError('--iterations must exceed the saved iteration')
    keys = ('num_envs', 'capacity', 'cfgen_ratio', 'action_noise', 'goal_mode', 'seed')
    if version in (5, 6, 7):
        keys += ('termination_height_m', 'termination_gravity_z', 'termination_softness', 'critic_value_max', 'drop_penalty')
    if version in (6, 7):
        keys += ('balance_reward_weight', 'action_rate_weight', 'action_l2_weight', 'obs_noise_scale',
                 'critic_norm', 'teacher_init_fraction', 'teacher_decay_iters', 'teacher_bc_weight')
    for key in keys:
        if ck['config'][key] != getattr(args, key):
            raise ValueError(f'--resume requires the saved {key} value')


def _reserve_collection_headroom(device, minimum):
    """Warp graph allocations cannot reclaim memory held by PyTorch's cache."""
    free, _ = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    released = 0
    free_after = free
    if free < minimum and reserved - allocated >= 64 * 1024**2:
        torch.cuda.empty_cache()
        released = max(0, reserved - torch.cuda.memory_reserved(device))
        free_after, _ = torch.cuda.mem_get_info(device)
    return {"gpu/free_before_cache_trim_bytes": free,
            "gpu/free_before_collect_bytes": free_after,
            "gpu/cache_released_before_collect_bytes": released,
            "gpu/torch_allocated_before_collect_bytes": allocated,
            "gpu/torch_reserved_before_collect_bytes": reserved - released}


def _run(args, version, learner_type):
    from .collector_v4 import VectorCollector
    from .replay_v4 import VectorReplay
    from .warp_sim import WarpCarryWorlds

    start = time.monotonic()
    output = args.output_dir
    if not args.resume and any((output / name).exists() for name in ('latest.pt', 'config.json', 'metrics.jsonl')):
        raise FileExistsError(f'{output} already contains a run. Select a new directory or resume its checkpoint.')
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('The selected device does not support bfloat16')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = learner_type(args)
    clock = UpdateClock(args.num_envs)
    first, elapsed, old_steps, old_ticks = 1, 0., 0, 0
    peak_alloc = peak_reserved = 0
    episodes = {}
    stage_state = None
    clip_options = (dict(clip_storage=args.clip_storage,
                         clip_max_device_bytes=int(args.clip_budget_gib * 1024**3),
                         clip_reserve_device_bytes=args.warp_headroom_mib * 1024**2)
                    if version == 7 else {})
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_resume(ck, args, version)
        for model, state in zip(net.models, ck['models']):
            model.load_state_dict(state)
        for optimizer, state in zip(net.optimizers, ck['optimizers']):
            optimizer.load_state_dict(state)
        replay = VectorReplay.from_legacy(ck['replay'], device, **clip_options)
        first, elapsed = ck['iteration'] + 1, ck['elapsed']
        old_steps, old_ticks = ck['real_steps'], ck['control_ticks']
        peak_alloc, peak_reserved = ck['peak_bytes']
        clock.credit = ck['credit']
        rng.bit_generator.state = ck['rng']
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state(ck['cuda_rng'], device)
        episodes = ck['episodes']
        if version == 7:
            stage_state = ck['stage_sampler']
        del ck
    else:
        replay = VectorReplay(args.capacity, args.seed, args.num_envs, device=device, **clip_options)

    # Recreate physical environments on restart. A fresh segment in every world
    # prevents a sampled WM sequence from straddling the restart boundary.
    teacher_init = float(getattr(args, 'teacher_init_fraction', 0.0))
    teacher_decay = int(getattr(args, 'teacher_decay_iters', 0))
    teacher_bc_weight = float(getattr(args, 'teacher_bc_weight', 0.0))
    teacher_bc_batch = int(getattr(args, 'teacher_bc_batch', 1024))
    env_kwargs = {}
    if version == 7:
        env_kwargs = {
            'v7': True,
            'gpu_recovery': args.gpu_recovery,
            'defer_recovery': getattr(args, 'defer_recovery', False),
            'recovery_workers': getattr(args, 'recovery_workers', None),
        }
    env = WarpCarryWorlds(
        args.sim_root, args.motion_dir, num_envs=args.num_envs,
        seed=args.seed + old_ticks, device=device, cfgen_ratio=args.cfgen_ratio,
        cfgen_cache_dir=args.cfgen_cache_dir, use_cuda_graph=args.cuda_graph,
        teacher_history=(teacher_init > 0.0 and first <= teacher_decay),
        **env_kwargs,
    )
    if stage_state is not None:
        env.stage_sampler.load_state_dict(stage_state)
    collector = VectorCollector(env, net, replay, rng)
    collector.episodes.update(episodes)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(version=version, replay_ratio=REPLAY_RATIO, wm_horizon=base.WM_STEPS,
                  actor_horizon=base.HORIZON, physics='mujoco_warp', torch=str(torch.__version__),
                  hardware=torch.cuda.get_device_name(device))
    if version == 7:
        config.update(replay_steps_per_env=replay.rows, goal_dim=122, wm_palm_features=6,
                      stage_uniform_fraction=0.5, teacher_free=True)
    (output / ('resume-config.json' if args.resume else 'config.json')).write_text(json.dumps(config, indent=2) + '\n')

    def record(data):
        line = json.dumps(data, allow_nan=False)
        with (output / 'metrics.jsonl').open('a') as log:
            log.write(line + '\n')
        print(line, flush=True)

    checkpointer = AsyncCheckpointer(enabled=args.async_save)

    def make_payload(iteration):
        return dict(
            version=version,
            iteration=iteration,
            elapsed=elapsed + time.monotonic() - start,
            real_steps=old_steps + collector.steps,
            control_ticks=old_ticks + collector.steps // args.num_envs,
            models=[_cpu_clone(m.state_dict()) for m in net.models],
            optimizers=[_cpu_clone(o.state_dict()) for o in net.optimizers],
            replay=replay.snapshot_cpu(),
            credit=clock.credit,
            rng=copy.deepcopy(rng.bit_generator.state),
            torch_rng=torch.get_rng_state().clone(),
            cuda_rng=torch.cuda.get_rng_state(device).clone(),
            episodes=Counter(collector.episodes),
            **({"stage_sampler": env.stage_sampler.state_dict()} if version == 7 else {}),
            config=dict(config),
            peak_bytes=(peak_alloc, peak_reserved),
        )

    full_save_every = getattr(args, 'full_save_every', None) or args.save_every

    def make_weights_payload(iteration):
        return dict(
            version=version,
            iteration=iteration,
            elapsed=elapsed + time.monotonic() - start,
            real_steps=old_steps + collector.steps,
            control_ticks=old_ticks + collector.steps // args.num_envs,
            models=[_cpu_clone(m.state_dict()) for m in net.models],
            config=dict(config),
            peak_bytes=(peak_alloc, peak_reserved),
        )

    def save(iteration, *, full=True):
        if full:
            checkpointer.save(lambda: make_payload(iteration), output / 'latest.pt')
        else:
            checkpointer.save(lambda: make_weights_payload(iteration), output / 'weights.pt')

    def warmup(minimum, *, update_normalizers, reason, episode_teacher_frac=0.0):
        initial_steps = collector.steps
        while collector.steps - initial_steps < minimum or not replay.has_window(base.WM_STEPS):
            collector.one_tick(noise=args.action_noise, update_normalizers=update_normalizers,
                               defer_reward=True, episode_teacher_frac=episode_teacher_frac,
                               record_reward=(version != 7))
            if collector.steps - initial_steps >= max(args.capacity, minimum) and not replay.has_window(base.WM_STEPS):
                raise RuntimeError('Warmup produced no continuous eight-step trajectory in any world')
        base.sync(device)
        record({'stage': 'warmup', 'reason': reason, 'real_steps': old_steps + collector.steps,
                'warmup/valid_windows': replay.valid_windows(base.WM_STEPS),
                'warmup/actor_updates': 0, 'warmup/critic_updates': 0, 'warmup/wm_updates': 0,
                'seconds': time.monotonic() - start})

    phase = 'warmup'
    iteration = first
    memory = {}
    try:
        if args.resume:
            resume_teacher = teacher_schedule(first, teacher_init, teacher_decay)
            warmup(base.WM_STEPS * args.num_envs, update_normalizers=False,
                   reason='new_physical_episodes_after_resume', episode_teacher_frac=resume_teacher)
        else:
            init_teacher = teacher_init if first <= teacher_decay else 0.0
            warmup(args.warmup_steps, update_normalizers=True, reason='initial_replay',
                   episode_teacher_frac=init_teacher)

        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(first, args.iterations + 1):
            phase = 'collection'
            ep_teacher = teacher_schedule(iteration, teacher_init, teacher_decay)
            bc_w = teacher_schedule(iteration, teacher_bc_weight, teacher_decay)
            if env.teacher_history and ep_teacher <= 0.0:
                env.teacher_history = False
            evaluate = args.eval_every and (iteration % args.eval_every == 0 or iteration == args.iterations)
            should_save = iteration in (first, args.iterations) or iteration % args.save_every == 0 or iteration % full_save_every == 0
            should_log = evaluate or iteration in (first, args.iterations) or iteration % args.log_every == 0
            if should_log:
                base.sync(device)
            tick = time.monotonic()
            memory = (_reserve_collection_headroom(device, args.warp_headroom_mib * 1024**2)
                      if version == 7 else {})
            before = collector.steps
            expected_ticks = clock.ticks_until_update()
            totals = {}
            updates = 0
            while not updates:
                for key, value in collector.one_tick(
                    noise=args.action_noise, defer_reward=not should_log,
                    episode_teacher_frac=ep_teacher, record_reward=(version != 7 or should_log),
                ).items():
                    totals[key] = totals.get(key, 0.) + value
                updates = clock.tick()
                if updates and not replay.has_window(base.WM_STEPS):
                    # Short episodes can evict every valid window. Don't count the
                    # extra collection as an update with replay ratio 16.
                    warmup(base.WM_STEPS * args.num_envs, update_normalizers=False,
                           reason='no_continuous_window', episode_teacher_frac=ep_teacher)
                    before = collector.steps
                    expected_ticks = clock.ticks_until_update()
                    totals.clear()
                    updates = 0
            new_steps = collector.steps - before
            if updates != 1 or new_steps != expected_ticks * args.num_envs:
                raise RuntimeError('Inconsistent vector-step count for one learning round')
            stats = {
                key: (value.item() if torch.is_tensor(value) else value) / (new_steps // args.num_envs)
                for key, value in totals.items()
            }
            if should_log:
                base.sync(device)
            after_collect = time.monotonic()
            phase = 'wm'
            stats.update(train_v3.wm_iteration(net, replay, args.wm_batch_size))
            if should_log:
                base.sync(device)
            after_wm = time.monotonic()
            phase = 'actor_critic'
            if bc_w > 0.0:
                stats.update(net.policy_update(replay, args.batch_size, bc_weight=bc_w, bc_batch=teacher_bc_batch))
            else:
                stats.update(net.policy_update(replay, args.batch_size))
            if should_log:
                base.sync(device)
            after_policy = time.monotonic()
            if should_log or should_save:
                peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                torch.cuda.reset_peak_memory_stats(device)
            stats.update(
                iteration=iteration, real_steps=old_steps + collector.steps, batch_size=args.batch_size,
                **{'sim/new_transitions': new_steps, 'sim/num_envs': args.num_envs,
                   'replay/ratio': WM_SAMPLES / new_steps, 'timing/collect_sec': after_collect - tick,
                   'timing/wm_sec': after_wm - after_collect, 'timing/policy_sec': after_policy - after_wm,
                   'gpu/peak_allocated_bytes': peak_alloc, 'gpu/peak_reserved_bytes': peak_reserved}
            )
            if version == 7:
                stats.update({"sim/vector_ticks": expected_ticks, "replay/ratio_target": REPLAY_RATIO})
                stats.update(memory)
                stats.update(replay.clip_bank.storage_metrics())
            if evaluate:
                phase = 'evaluation'
                stats.update(base.validate(net, args))
            if should_save:
                phase = 'checkpoint'
                is_full_save = (iteration % full_save_every == 0) or (iteration == args.iterations)
                save(iteration, full=is_full_save)
            if should_log:
                stats['wall_seconds'] = elapsed + time.monotonic() - start
                stats.update({f'collect/{k}': v for k, v in collector.episodes.items()})
                record(stats)
        checkpointer.wait()
        summary = dict(completed_iterations=args.iterations, real_steps=old_steps + collector.steps,
                       wall_seconds=elapsed + time.monotonic() - start, peak_allocated_bytes=peak_alloc,
                       peak_reserved_bytes=peak_reserved, hardware=config['hardware'])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    except RuntimeError as exc:
        if version == 7 and ('out of memory' in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)):
            diagnostic = {'stage': 'cuda_oom', 'phase': phase, 'iteration': iteration,
                          **memory, **replay.clip_bank.storage_metrics(), 'error': str(exc)}
            try:
                diagnostic['gpu/free_on_error_bytes'] = torch.cuda.mem_get_info(device)[0]
                diagnostic['gpu/torch_allocated_on_error_bytes'] = torch.cuda.memory_allocated(device)
                diagnostic['gpu/torch_reserved_on_error_bytes'] = torch.cuda.memory_reserved(device)
            except RuntimeError as query_error:
                diagnostic['gpu/query_error'] = str(query_error)
            record(diagnostic)
        raise
    finally:
        checkpointer.close()
        env.close()


if __name__ == '__main__':
    train(args_from_cli())
