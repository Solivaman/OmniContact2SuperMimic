"""Teacher-free reference state initialization and failure/return-adaptive start bins."""
from dataclasses import dataclass
from weakref import ref

import numpy as np
import torch

from .data import DT


STAGES = ("approach", "pregrasp", "lift", "carry", "place")
BINS = 20
SOURCES = ("npz", "cfgen")


@dataclass(frozen=True)
class ReturnSamplerCfg:
    """SuperMimic discounted-return reset sampler configuration."""

    gamma: float = 0.99
    horizon: int = 20
    memory: int = 256
    steepness: float = 25.0
    explore: float = 0.5
    keep: float = 0.1

    def __post_init__(self):
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.horizon < 1 or self.memory < 1:
            raise ValueError("horizon and memory must be positive")
        if not np.isfinite(self.steepness) or self.steepness <= 0.0:
            raise ValueError("steepness must be positive and finite")
        if self.explore < 0.0 or self.keep < 0.0 or self.explore + self.keep >= 1.0:
            raise ValueError("explore >= 0, keep >= 0, and explore + keep < 1 required")


class ReturnSampler:
    """Per-source SuperMimic ReturnSampler across variable-length reference clips.

    Bins each source's clip timeline into ``BINS`` intervals, tracks the moving mean
    discounted return ``V[source, b]`` from on-policy rollouts, and mixes
    ``explore`` uniform + ``(1 - explore - keep) * softmax(-steepness * V / v_max)``
    + ``keep`` historical focus squared effort.
    """

    def __init__(
        self,
        cfg: ReturnSamplerCfg | None = None,
        bin_count: int = BINS,
        num_envs: int = 1,
        max_step_reward: float = 1.0,
        num_sources: int = len(SOURCES),
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg or ReturnSamplerCfg()
        if bin_count < 1 or num_envs < 1 or num_sources < 1:
            raise ValueError("bin_count, num_envs, and num_sources must be positive")
        if not np.isfinite(max_step_reward) or max_step_reward <= 0.0:
            raise ValueError("max_step_reward must be positive and finite")
        self.bin_count = int(bin_count)
        self.num_envs = int(num_envs)
        self.num_sources = int(num_sources)
        self.device = torch.device(device)
        self.max_step_reward = float(max_step_reward)
        self.v_max = self.max_step_reward / (1.0 - float(self.cfg.gamma))

        self.value = torch.full(
            (self.num_sources, self.bin_count), self.v_max, dtype=torch.float32, device=self.device
        )
        self.count = torch.zeros(
            (self.num_sources, self.bin_count), dtype=torch.float32, device=self.device
        )
        self.effort = torch.zeros(
            (self.num_sources, self.bin_count), dtype=torch.float64, device=self.device
        )

        shape = (self.num_envs, int(self.cfg.horizon))
        self._frames = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._clip_lens = torch.ones(shape, dtype=torch.long, device=self.device)
        self._sources = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self._rewards = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._succeeded = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._step = 0

    def to(self, device):
        target = torch.device(device)
        if target == self.device:
            return self
        self.device = target
        for attr in (
            "value", "count", "effort", "_frames", "_clip_lens", "_sources",
            "_rewards", "_terminated", "_succeeded", "_clip_end", "_timed_out",
        ):
            setattr(self, attr, getattr(self, attr).to(target))
        return self

    def _bins(self, frames: torch.Tensor, clip_lens: torch.Tensor) -> torch.Tensor:
        return torch.clamp(frames * self.bin_count // clip_lens.clamp_min(1), 0, self.bin_count - 1)

    def _dones(self) -> torch.Tensor:
        return self._terminated | self._succeeded | self._clip_end | self._timed_out

    def record(
        self,
        frames: torch.Tensor,
        clip_lens: torch.Tensor,
        sources: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        succeeded: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> np.ndarray | None:
        """Buffer one env tick; every ``horizon`` ticks update V and return [sources, bins] probs."""
        col = self._step
        self._frames[:, col] = frames.to(device=self.device, dtype=torch.long)
        self._clip_lens[:, col] = clip_lens.to(device=self.device, dtype=torch.long)
        self._sources[:, col] = sources.to(device=self.device, dtype=torch.long)
        self._rewards[:, col] = rewards.to(device=self.device, dtype=torch.float32)
        self._terminated[:, col] = terminated.to(device=self.device, dtype=torch.bool)
        self._succeeded[:, col] = succeeded.to(device=self.device, dtype=torch.bool)
        self._clip_end[:, col] = clip_end.to(device=self.device, dtype=torch.bool)
        self._timed_out[:, col] = timed_out.to(device=self.device, dtype=torch.bool)
        self._step += 1
        if self._step < self.cfg.horizon:
            return None
        self._step = 0
        self._update()
        return self._distribution().detach().cpu().numpy().astype(np.float64)

    def _returns(self) -> torch.Tensor:
        gamma = float(self.cfg.gamma)
        safe_src = self._sources.clamp(min=0, max=self.num_sources - 1)
        next_frames = torch.minimum(self._frames + 1, (self._clip_lens - 1).clamp_min(0))
        next_bins = self._bins(next_frames, self._clip_lens)
        src_mean = self.value.mean(dim=1)[safe_src]
        cont_val = self.value[safe_src, next_bins]
        next_value = torch.where(
            self._terminated,
            0.0,
            torch.where(
                self._succeeded,
                self.v_max,
                torch.where(self._clip_end, src_mean, cont_val),
            ),
        )
        dones = self._dones()
        rewards = self._rewards
        returns = torch.empty_like(rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.cfg.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self) -> None:
        returns = self._returns()
        dones = self._dones()
        num_envs, horizon = dones.shape
        valid = (self._sources >= 0) & (self._sources < self.num_sources) & (self._clip_lens > 0)
        if not bool(valid.any()):
            return
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1).expand_as(episode)
        bins = self._bins(self._frames, self._clip_lens)
        stride = self.num_sources * self.bin_count
        pair = self._sources * self.bin_count + bins
        key = (envs * horizon + episode) * stride + pair

        valid_flat = valid.reshape(-1)
        key_v = key.reshape(-1)[valid_flat]
        ret_v = returns.reshape(-1)[valid_flat]
        keys, inverse, visits = torch.unique(key_v, return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), dtype=ret_v.dtype, device=ret_v.device)
        samples.scatter_add_(0, inverse, ret_v)
        samples = samples / visits

        pairs = keys % stride
        new = torch.zeros(stride, dtype=self.count.dtype, device=self.count.device).scatter_add_(
            0, pairs, torch.ones_like(samples)
        ).view(self.num_sources, self.bin_count)
        total = torch.zeros(stride, dtype=self.value.dtype, device=self.value.device).scatter_add_(
            0, pairs, samples
        ).view(self.num_sources, self.bin_count)

        self.count = self.count.clamp(max=self.cfg.memory) + new
        self.value = torch.where(
            new > 0, self.value + (total - new * self.value) / self.count.clamp_min(1.0), self.value
        )

    def _distribution(self) -> torch.Tensor:
        cfg = self.cfg
        focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=1)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum(dim=1, keepdim=True).clamp_min(1e-12)).float()
        explore = cfg.explore / self.bin_count
        return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep

    def metrics(self) -> dict[str, float]:
        vals = self.value.detach().cpu().numpy()
        out = {}
        for idx, name in enumerate(SOURCES[: self.num_sources]):
            out[f"return_sampler/{name}_mean_v"] = float(np.mean(vals[idx]))
            out[f"return_sampler/{name}_min_v"] = float(np.min(vals[idx]))
        return out

    def state_dict(self) -> dict:
        return {
            "value": self.value.detach().cpu().numpy().copy(),
            "count": self.count.detach().cpu().numpy().copy(),
            "effort": self.effort.detach().cpu().numpy().copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        value = np.asarray(state["value"], dtype=np.float32)
        count = np.asarray(state["count"], dtype=np.float32)
        effort = np.asarray(state["effort"], dtype=np.float64)
        expected = (self.num_sources, self.bin_count)
        if value.shape != expected or count.shape != expected or effort.shape != expected:
            raise ValueError("Invalid ReturnSampler checkpoint state shape")
        if not (np.isfinite(value).all() and np.isfinite(count).all() and np.isfinite(effort).all()):
            raise ValueError("ReturnSampler checkpoint state contains non-finite values")
        if np.any(count < 0.0) or np.any(effort < 0.0):
            raise ValueError("ReturnSampler counts and effort must be nonnegative")
        self.value.copy_(torch.as_tensor(value, device=self.device, dtype=torch.float32))
        self.count.copy_(torch.as_tensor(count, device=self.device, dtype=torch.float32))
        self.effort.copy_(torch.as_tensor(effort, device=self.device, dtype=torch.float64))


class NoStageStart(ValueError):
    """This reference has no physically eligible frame for the requested start."""


def _time_bin(frame, length):
    return np.minimum(np.asarray(frame) * BINS // length, BINS - 1)


def _stage_labels(frames):
    """Label individual pickups, including motions that revisit a table or loop."""
    size = len(frames)
    labels = np.zeros(size, dtype=np.int8)
    held = (frames[:, 37:39] > .5).all(axis=1)
    both = np.flatnonzero(held)
    if not len(both):
        return labels, size - 1
    z = frames[:, 106]
    box_xy = frames[:, 104:106]
    # A momentary contact-label flicker does not constitute an intermediate release.
    segments = np.split(both, np.flatnonzero(np.diff(both) > 10) + 1)
    previous_end = 0
    for segment in segments:
        first, end = int(segment[0]), int(segment[-1]) + 1
        labels[max(previous_end, first - 25):first] = 1
        labels[first:end] = 2
        high = max(float(z[first]) + .025, float(np.quantile(z[segment], .85)) - .025)
        top = segment[z[segment] >= high]
        if len(top):
            labels[int(top[0]):end] = 3
        # A loop may pass the drop-off several times. Use its *last* arrival.
        away = np.linalg.norm(box_xy[first:end] - box_xy[end - 1], axis=1) > .20
        outside = np.flatnonzero(away)
        arrival = first + int(outside[-1]) + 1 if len(outside) else max(first, end - 25)
        labels[arrival:end] = 4
        previous_end = end
    return labels, int(both[-1])


def _clip_box_metadata(frames, half, table_positions, table_half_sizes):
    """Compute admissible free-box starts, per-frame held-in-air flags, and final Z extent."""
    from .sim import _matrix

    centers = frames[:, 104:107]
    extent = np.abs(_matrix(frames[:, 107:111])) @ np.asarray(half)
    bottom = centers[:, 2] - extent[:, 2]
    # The floor and both tabletops can support the box. A few millimetres of
    # solver/retargeting tolerance avoids rejecting contacts at rest.
    supported = abs(bottom) <= .012
    penetrates = bottom < -.012
    on_support = bottom <= .005
    for pos, size in zip(table_positions, table_half_sizes):
        overlap = (abs(centers[:, :2] - pos[:2]) < size[:2] + extent[:, :2]).all(axis=1)
        above_slab = centers[:, 2] + extent[:, 2] > pos[2] - size[2]
        gap = bottom - (pos[2] + size[2])
        supported |= overlap & (abs(gap) <= .012)
        penetrates |= overlap & above_slab & (gap < -.012)
        on_support |= overlap & (bottom <= pos[2] + size[2] + .005)
    two_hands = (frames[:, 37:39] > .5).all(axis=1)
    starts_holding = two_hands & ~on_support
    starts_holding[0] = False
    return ~penetrates & (supported | two_hands), starts_holding, float(extent[-1, 2])


def _support_starts(frames, half, table_positions, table_half_sizes):
    """Reject penetrating boxes and unsupported free-box starts in the shared scene."""
    return _clip_box_metadata(frames, half, table_positions, table_half_sizes)[0]


def _angular_velocity(q):
    """World-frame angular velocity from neighboring normalized wxyz orientations."""
    q = np.asarray(q, dtype=np.float64)
    idx = np.arange(len(q))
    prev, nxt = np.maximum(idx - 1, 0), np.minimum(idx + 1, len(q) - 1)
    a, b = q[nxt], q[prev]
    w = np.sum(a * b, axis=1)
    vec = b[:, :1] * a[:, 1:] - a[:, :1] * b[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
    vec *= np.where(w < 0, -1., 1.)[:, None]
    norm = np.linalg.norm(vec, axis=1)
    angle = 2 * np.arctan2(norm, abs(w))
    return vec * (angle / np.maximum(norm, 1e-12) / ((nxt - prev) * DT))[:, None]


def generated_twists(frames):
    root, box = frames[:, 97:104], frames[:, 104:111]
    return np.stack((np.gradient(root[:, :3], DT, axis=0), _angular_velocity(root[:, 3:7]),
                     np.gradient(box[:, :3], DT, axis=0), _angular_velocity(box[:, 3:7])),
                    axis=1).astype(np.float32)


@dataclass
class ClipStarts:
    valid: np.ndarray
    labels: np.ndarray
    bins: tuple
    twists: np.ndarray | None
    starts_holding: np.ndarray | None = None
    last_box_extent_z: float | None = None


class StageSampler:
    """Per-source curriculum; supports failure-rate bins (V7) or discounted-return bins (Ultra)."""

    def __init__(
        self,
        sampling_mode: str = "failure",
        return_cfg: ReturnSamplerCfg | None = None,
        num_envs: int = 1,
        max_step_reward: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        if sampling_mode not in ("failure", "return"):
            raise ValueError(f"Unknown V7 reset sampling mode {sampling_mode!r}")
        self.sampling_mode = sampling_mode
        self.visits = np.zeros((len(SOURCES), BINS), dtype=np.int64)
        self.failures = np.zeros_like(self.visits)
        self.return_sampler = (
            ReturnSampler(
                cfg=return_cfg,
                bin_count=BINS,
                num_envs=num_envs,
                max_step_reward=max_step_reward,
                num_sources=len(SOURCES),
                device=device,
            )
            if sampling_mode == "return"
            else None
        )
        self.return_probs = None
        self._clips = {}

    def state_dict(self):
        state = {
            "visits": self.visits.copy(),
            "failures": self.failures.copy(),
            "sampling_mode": self.sampling_mode,
        }
        if self.return_sampler is not None:
            state["return_sampler"] = self.return_sampler.state_dict()
            state["return_probs"] = None if self.return_probs is None else self.return_probs.copy()
        return state

    def load_state_dict(self, state):
        saved_mode = state.get("sampling_mode", "failure")
        if saved_mode != self.sampling_mode:
            raise ValueError(f"Cannot load {saved_mode!r} stage curriculum into {self.sampling_mode!r} sampler")
        visits, failures = (np.asarray(state[k], dtype=np.int64) for k in ("visits", "failures"))
        if (visits.shape != self.visits.shape or failures.shape != visits.shape
                or np.any(failures < 0) or np.any(visits < failures)):
            raise ValueError("Invalid V7 stage curriculum checkpoint")
        self.visits[:], self.failures[:] = visits, failures
        if self.return_sampler is not None:
            if "return_sampler" not in state:
                raise ValueError("Checkpoint missing return_sampler state")
            self.return_sampler.load_state_dict(state["return_sampler"])
            rp = state.get("return_probs")
            self.return_probs = None if rp is None else np.asarray(rp, dtype=np.float64).copy()

    def catalogue(self, root, frames, half, source, phase, table_positions, table_half_sizes, *, precomputed_valid=None):
        # Fixed training clips share one catalogue across every physical world.
        # Let temporary plans from independent diagnostic resets be collected.
        key = id(frames)
        if key not in self._clips:
            if precomputed_valid is not None:
                valid = np.asarray(precomputed_valid, dtype=bool).copy()
                if valid.shape != (len(frames),):
                    raise ValueError(f"precomputed_valid shape {valid.shape} does not match clip length {len(frames)}")
            else:
                from .geometry_v7 import valid_start_mask
                valid = np.asarray(valid_start_mask(root, frames, half, source, phase), dtype=bool).copy()
            support_valid, starts_holding, last_box_extent_z = _clip_box_metadata(
                frames, half, table_positions, table_half_sizes
            )
            valid &= support_valid
            labels, final_grasp = _stage_labels(frames)
            # No suffix from a released box or starts with fewer than 8 real transitions.
            valid[min(final_grasp + 1, max(0, len(frames) - 8)):] = False
            indices = np.flatnonzero(valid)
            timed = _time_bin(indices, len(frames))
            bins = tuple(indices[timed == i] for i in range(BINS))
            entry = ClipStarts(
                indices,
                labels,
                bins,
                generated_twists(frames) if source == "cfgen" else None,
                starts_holding=starts_holding,
                last_box_extent_z=last_box_extent_z,
            )
            self._clips[key] = (ref(frames, lambda _, k=key: self._clips.pop(k, None)), entry)
        return self._clips[key][1]

    def choose(self, catalogue, source, rng, stage=None):
        if stage is not None:
            phase_id = STAGES.index(stage)
            candidates = catalogue.valid[catalogue.labels[catalogue.valid] == phase_id]
            if not len(candidates):
                raise NoStageStart(f"No eligible {stage} frame in {source} reference")
            return int(rng.choice(candidates))
        available = np.array([i for i, frames in enumerate(catalogue.bins) if len(frames)])
        if not len(available):
            raise NoStageStart(f"No eligible V7 curriculum frame in {source} reference")
        src = SOURCES.index(source)
        if self.sampling_mode == "return":
            if self.return_probs is None:
                probs = np.full(len(available), 1.0 / len(available), dtype=np.float64)
            else:
                weights = np.asarray(self.return_probs[src, available], dtype=np.float64)
                total = float(weights.sum())
                probs = (
                    weights / total
                    if np.isfinite(total) and total > 0.0
                    else np.full(len(available), 1.0 / len(available), dtype=np.float64)
                )
        else:
            rates = (1.0 + self.failures[src, available]) / (2.0 + self.visits[src, available])
            probs = .5 / len(available) + .5 * rates / rates.sum()
        chosen = int(rng.choice(available, p=probs))
        return int(rng.choice(catalogue.bins[chosen]))

    def finish(self, source, start, end, length, success):
        if source not in SOURCES:
            return
        src = SOURCES.index(source)
        a, b = int(_time_bin(start, length)), int(_time_bin(end, length))
        self.visits[src, a:b + 1] += 1
        if not success:
            self.failures[src, b] += 1
