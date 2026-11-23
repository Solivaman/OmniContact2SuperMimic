"""Teacher-free reference state initialization and failure-adaptive start bins."""
from dataclasses import dataclass
from weakref import ref

import numpy as np

from .data import DT


STAGES = ("approach", "pregrasp", "lift", "carry", "place")
BINS = 20
SOURCES = ("npz", "cfgen")


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
    """Per-source curriculum; count failure in a bin relative to opportunities to traverse it."""

    def __init__(self):
        self.visits = np.zeros((len(SOURCES), BINS), dtype=np.int64)
        self.failures = np.zeros_like(self.visits)
        self._clips = {}

    def state_dict(self):
        return {"visits": self.visits.copy(), "failures": self.failures.copy()}

    def load_state_dict(self, state):
        visits, failures = (np.asarray(state[k], dtype=np.int64) for k in ("visits", "failures"))
        if (visits.shape != self.visits.shape or failures.shape != visits.shape
                or np.any(failures < 0) or np.any(visits < failures)):
            raise ValueError("Invalid V7 stage curriculum checkpoint")
        self.visits[:], self.failures[:] = visits, failures

    def catalogue(self, root, frames, half, source, phase, table_positions, table_half_sizes):
        # Fixed training clips share one catalogue across every physical world.
        # Let temporary plans from independent diagnostic resets be collected.
        key = id(frames)
        if key not in self._clips:
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
