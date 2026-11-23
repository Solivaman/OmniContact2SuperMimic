"""Paged reference frames for the single-writer vector replay.

Each immutable clip has a stable slot ID. Small fixed-size frame blocks can span
device pages and be returned to other slots after replay has evicted the clip.
The CUDA gather follows a device-side table of block addresses. Its inputs and
outputs remain on the caller's PyTorch stream, including for recovery references.
"""
import math
import weakref

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only tests and checkpoint readers do not require Triton.
    triton = None


if triton is not None:
    @triton.jit
    def _gather_frames(meta, ids, frames, offsets, dst, dst_rows, slots, times: tl.constexpr,
                       width: tl.constexpr, block_frames: tl.constexpr,
                       tile_t: tl.constexpr, tile_d: tl.constexpr, scatter: tl.constexpr):
        row = tl.program_id(0)
        t = tl.program_id(1) * tile_t + tl.arange(0, tile_t)
        col = tl.program_id(2) * tile_d + tl.arange(0, tile_d)
        slot = tl.load(ids + row).to(tl.int64)
        slot_meta = meta + 2 * slot
        in_range = (slot >= 0) & (slot < slots)
        address = tl.load(slot_meta, mask=in_range, other=0)
        length = tl.load(slot_meta + 1, mask=in_range, other=0)
        live = in_range & (length > 0)
        tl.device_assert(live, "Missing or retired reference slot")
        table = address.to(tl.pointer_type(tl.int64))
        start = tl.load(frames + row).to(tl.int64)
        offset = tl.load(offsets + t, mask=t < times, other=0).to(tl.int64)
        local = tl.maximum(0, tl.minimum(start + offset, length - 1))
        block = tl.load(table + local // block_frames, mask=(t < times) & live,
                        other=0).to(tl.pointer_type(tl.float32))
        source = block + (local % block_frames) * width
        mask = (t[:, None] < times) & (col[None, :] < width)
        values = tl.load(source[:, None] + col[None, :], mask=mask & live, other=0.)
        target_row = tl.load(dst_rows + row) if scatter else row
        target = dst + (target_row.to(tl.int64) * times + t) * width
        tl.store(target[:, None] + col[None, :], values, mask=mask)


def _aligned(size):
    """Charge each retained tensor at CUDA's standard 512-byte allocation granularity."""
    return (size + 511) // 512 * 512


class PagedClipBank:
    def __init__(self, device, raw_dim=111, page_frames=65536, *, max_device_bytes=None, reserve_device_bytes=0):
        if raw_dim < 1 or page_frames < 1:
            raise ValueError("raw_dim and page_frames must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type not in ("cuda", "cpu") or (self.device.type == "cuda" and triton is None):
            raise RuntimeError("PagedClipBank requires Triton on CUDA or the CPU reference path")
        self.raw_dim = int(raw_dim)
        self.page_frames = int(page_frames)
        self.max_device_bytes = None if max_device_bytes is None else int(max_device_bytes)
        self.reserve_device_bytes = int(reserve_device_bytes)
        if (self.max_device_bytes is not None and self.max_device_bytes < 0) or self.reserve_device_bytes < 0:
            raise ValueError("Device budget and headroom must be nonnegative")
        self._page_charge = _aligned(self.page_frames * self.raw_dim * 4)
        self._tables_charge = 0
        self._spill = {}
        self._spill_frames = 0
        self.block_frames = math.gcd(self.page_frames, 32)
        self._blocks_per_page = self.page_frames // self.block_frames
        self.pages = []
        self._next_block = 0
        self._free_blocks = []
        self._slot_blocks = []
        self._slot_tables = []
        self._slot_lengths = []
        self._slot_static = []
        self._generations = []
        self._free_slots = []
        self._objects = {}
        self._static_hashes = {}
        self._meta = torch.empty((0, 2), dtype=torch.int64, device=self.device)
        self._frames_used = 0
        self._live_blocks = 0
        self._frames_since_gc = 0

    def _grow_slots(self, count):
        if count > len(self._meta):
            size = max(count, 64, len(self._meta) * 2)
            meta = torch.zeros((size, 2), dtype=torch.int64, device=self.device)
            meta[:len(self._meta)].copy_(self._meta)
            self._meta = meta

    def _reserve_blocks(self, count):
        fresh = max(0, count - len(self._free_blocks))
        end = self._next_block + fresh
        while len(self.pages) * self._blocks_per_page < end:
            self.pages.append(torch.empty((self.page_frames, self.raw_dim), dtype=torch.float32, device=self.device))
        blocks = [self._free_blocks.pop() for _ in range(count - fresh)]
        blocks.extend(range(self._next_block, end))
        self._next_block = end
        return blocks

    def _runs(self, blocks, length):
        """Visit contiguous physical slices within a page in logical clip order."""
        begin = 0
        while begin < len(blocks):
            first = blocks[begin]
            page, local = divmod(first, self._blocks_per_page)
            end = begin + 1
            limit = min(len(blocks), begin + self._blocks_per_page - local)
            while end < limit and blocks[end] == blocks[end - 1] + 1:
                end += 1
            source = slice(begin * self.block_frames, min(length, end * self.block_frames))
            offset = local * self.block_frames
            yield self.pages[page][offset:offset + source.stop - source.start], source
            begin = end

    def _fits(self, slot, count):
        if self.max_device_bytes is None:
            return True
        end = self._next_block + max(0, count - len(self._free_blocks))
        pages = max(len(self.pages), (end + self._blocks_per_page - 1) // self._blocks_per_page)
        capacity = len(self._meta)
        if slot >= capacity:
            capacity = max(slot + 1, 64, 2 * capacity)
            # Both slot tensors are live briefly while the small metadata table grows.
            if self._accounted() + _aligned(capacity * 16) > self.max_device_bytes:
                return False
        required = pages * self._page_charge + _aligned(capacity * 16) + self._tables_charge + _aligned(count * 8)
        return required <= self.max_device_bytes

    def _accounted(self):
        return len(self.pages) * self._page_charge + _aligned(self._meta.nbytes) + self._tables_charge

    def _has_driver_headroom(self, slot, count):
        if self.device.type != "cuda" or not self.reserve_device_bytes:
            return True
        end = self._next_block + max(0, count - len(self._free_blocks))
        added = (end + self._blocks_per_page - 1) // self._blocks_per_page - len(self.pages)
        if added <= 0:
            return True
        needed = added * self._page_charge + _aligned(count * 8) + self.reserve_device_bytes
        if slot >= len(self._meta):
            needed += _aligned(16 * max(slot + 1, 64, 2 * len(self._meta)))
        free = torch.cuda.mem_get_info(self.device)[0]
        if free < needed and torch.cuda.memory_reserved(self.device) > torch.cuda.memory_allocated(self.device):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info(self.device)[0]
        return free >= needed

    def _put(self, slot, values, static):
        length = len(values)
        count = (length + self.block_frames - 1) // self.block_frames
        self._grow_slots(slot + 1)
        old_pages, old_cursor = len(self.pages), self._next_block
        blocks = []
        try:
            blocks = self._reserve_blocks(count)
            for physical, part in self._runs(blocks, length):
                physical.copy_(values[part])
            table = None
            address = 0
            if self.device.type == "cuda":
                stride = self.block_frames * self.raw_dim * 4
                addresses = [self.pages[b // self._blocks_per_page].data_ptr() +
                             (b % self._blocks_per_page) * stride for b in blocks]
                table = torch.tensor(addresses, dtype=torch.int64, device=self.device)
                address = table.data_ptr()
            self._meta[slot].copy_(torch.tensor([address, length], dtype=torch.int64))
        except BaseException:
            # A multi-page registration can fail halfway through. No bytes from
            # its newly allocated pages belong to a published slot.
            del self.pages[old_pages:]
            self._next_block = old_cursor
            self._free_blocks.extend(b for b in reversed(blocks) if b < old_cursor)
            raise
        self._tables_charge += _aligned(count * 8)
        self._live_blocks += count
        self._publish(slot, length, static, blocks, table)

    def _publish(self, slot, length, static, blocks=None, table=None):
        if slot == len(self._slot_lengths):
            self._slot_blocks.append(blocks)
            self._slot_tables.append(table)
            self._slot_lengths.append(length)
            self._slot_static.append(bool(static))
            self._generations.append(0)
        else:
            self._slot_blocks[slot], self._slot_tables[slot] = blocks, table
            self._slot_lengths[slot], self._slot_static[slot] = length, bool(static)
        self._frames_used += length
        if not static:
            self._frames_since_gc += length

    def _remember(self, clip, slot):
        key = id(clip)
        owner = weakref.ref(self)

        def forget(ref):
            bank = owner()
            if bank is not None and bank._objects.get(key, (None,))[0] is ref:
                del bank._objects[key]

        try:
            ref = weakref.ref(clip, forget)
        except TypeError:  # Array-like input which cannot be tracked by identity.
            return
        self._objects[key] = (ref, slot, self._generations[slot])

    def reclaim(self, active_ids, *, extra=()):
        """Recycle only dynamic clips absent from replay and live simulator worlds."""
        if torch.is_tensor(active_ids):
            active_ids = active_ids.detach().cpu().reshape(-1).tolist()
        active = set(active_ids)
        active.update(extra)
        retired = []
        retired_device = []
        for slot, length in enumerate(self._slot_lengths):
            if length and not self._slot_static[slot] and slot not in active:
                if slot in self._spill:
                    del self._spill[slot]
                    self._spill_frames -= length
                else:
                    blocks = self._slot_blocks[slot]
                    self._free_blocks.extend(reversed(blocks))
                    self._live_blocks -= len(blocks)
                    self._tables_charge -= _aligned(len(blocks) * 8)
                    retired_device.append(slot)
                self._slot_blocks[slot], self._slot_tables[slot] = None, None
                self._slot_lengths[slot] = 0
                self._generations[slot] += 1
                self._free_slots.append(slot)
                self._frames_used -= length
                retired.append(slot)
        if retired_device:
            self._meta[retired_device] = 0
        self._frames_since_gc = 0
        return len(retired)

    def _try_device(self, slot, values, static):
        count = (len(values) + self.block_frames - 1) // self.block_frames
        if self._fits(slot, count) and self._has_driver_headroom(slot, count):
            try:
                self._put(slot, values, static)
                return True
            except torch.cuda.OutOfMemoryError:
                # Warp or another process may consume memory before this budget.
                pass
        return False

    def _put_spill(self, slot, values, static):
        self._spill[slot] = values.detach().to("cpu", copy=True)
        self._spill_frames += len(values)
        self._publish(slot, len(values), static)

    def register(self, clip, *, static=False, active_ids=None):
        active = None if active_ids is None else lambda: active_ids
        return self.register_many([clip], static=static, active_ids_fn=active)[0]

    def register_many(self, clips, *, static=False, active_ids_fn=None):
        from .replay_v4 import ClipBank

        result = []
        swept = False
        for clip in clips:
            seen = self._objects.get(id(clip))
            if seen is not None and seen[0]() is clip and self._generations[seen[1]] == seen[2]:
                slot = seen[1]
                if not static or self._slot_static[slot]:
                    result.append(slot)
                    continue
            else:
                slot = None
            arr = np.ascontiguousarray(clip, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.raw_dim or not len(arr):
                raise ValueError(f"Expected a nonempty [T, {self.raw_dim}] clip, got {arr.shape}")
            if static:
                fingerprint = ClipBank._fingerprint(arr)
                if slot is None:
                    slot = self._static_hashes.get(fingerprint)
            if slot is None:
                count = (len(arr) + self.block_frames - 1) // self.block_frames
                free = len(self._free_blocks) + len(self.pages) * self._blocks_per_page - self._next_block
                # A full scan of replay is amortized across at least four pages of
                # new trajectories, even if collecting a recovery on every tick.
                slot = self._free_slots[-1] if self._free_slots else len(self._slot_lengths)
                if (not swept and active_ids_fn is not None and
                        ((count > free and self._frames_since_gc >= 4 * self.page_frames)
                         or not self._fits(slot, count))):
                    self.reclaim(active_ids_fn(), extra=result)
                    swept = True
                    slot = self._free_slots[-1] if self._free_slots else len(self._slot_lengths)
                values = torch.from_numpy(arr)
                placed = self._try_device(slot, values, static)
                if not placed and not swept and active_ids_fn is not None:
                    retired = self.reclaim(active_ids_fn(), extra=result)
                    swept = True
                    slot = self._free_slots[-1] if self._free_slots else len(self._slot_lengths)
                    if retired:
                        placed = self._try_device(slot, values, static)
                if not placed:
                    self._put_spill(slot, values, static)
                if self._free_slots:
                    self._free_slots.pop()
            if static:
                self._static_hashes[fingerprint] = slot
                self._slot_static[slot] = True
            self._remember(clip, slot)
            result.append(slot)
        return result

    def gather(self, clip_ids, frames, offsets):
        if clip_ids.ndim != 1 or frames.shape != clip_ids.shape or offsets.ndim != 1:
            raise ValueError("Expected a vector of slots, one frame per slot and a vector of offsets")
        if any(t.device != self.device or t.dtype not in (torch.int64, torch.int32)
               for t in (clip_ids, frames, offsets)):
            raise ValueError("Reference query indices must be int32 or int64 on the bank device")
        if self._spill and len(clip_ids):
            ids = clip_ids.cpu().tolist()
            spilled = [row for row, slot in enumerate(ids) if slot in self._spill]
            if spilled:
                # Only the presence of a host spill enables this mixed route.
                # Repeated requests for the same clip are gathered in one operation.
                rows = torch.tensor(spilled, dtype=torch.int64, device=self.device)
                local = frames.index_select(0, rows).long().cpu()[:, None] + offsets.cpu()
                host = torch.empty((len(rows), len(offsets), self.raw_dim), dtype=torch.float32)
                groups = {}
                for i, row in enumerate(spilled):
                    groups.setdefault(ids[row], []).append(i)
                for slot, positions in groups.items():
                    host[positions] = self._spill[slot][local[positions].clamp(0, self._slot_lengths[slot] - 1)]
                spill_values = host.to(self.device)
                if len(spilled) == len(ids):
                    return spill_values
                resident = torch.tensor([i for i, slot in enumerate(ids) if slot not in self._spill],
                                        dtype=torch.int64, device=self.device)
                out = torch.empty((len(ids), len(offsets), self.raw_dim), dtype=torch.float32, device=self.device)
                out.index_copy_(0, rows, spill_values)
                return self._gather_resident(clip_ids[resident], frames[resident], offsets, out, resident)
        return self._gather_resident(clip_ids, frames, offsets)

    def _gather_resident(self, clip_ids, frames, offsets, out=None, dst_rows=None):
        if out is None:
            out = torch.empty((len(clip_ids), len(offsets), self.raw_dim), dtype=torch.float32, device=self.device)
        if not out.numel():
            return out
        if self.device.type == "cuda":
            tile = triton.next_power_of_2(min(self.raw_dim, 128))
            grid = (len(clip_ids), triton.cdiv(len(offsets), 8), triton.cdiv(self.raw_dim, tile))
            with torch.cuda.device(self.device):
                _gather_frames[grid](self._meta, clip_ids.contiguous(), frames.contiguous(),
                                     offsets.contiguous(), out, clip_ids if dst_rows is None else dst_rows,
                                     min(len(self._slot_lengths), len(self._meta)),
                                     len(offsets), self.raw_dim,
                                     self.block_frames, 8, tile, dst_rows is not None)
        else:
            for row, (slot, start) in enumerate(zip(clip_ids.tolist(), frames.tolist())):
                if slot < 0 or slot >= len(self._slot_lengths) or not self._slot_lengths[slot]:
                    raise IndexError("Missing or retired reference slot")
                length = self._slot_lengths[slot]
                times = (start + offsets).clamp(0, length - 1)
                blocks = torch.as_tensor(self._slot_blocks[slot], dtype=torch.int64)
                physical = blocks[times // self.block_frames] * self.block_frames + times % self.block_frames
                pages = physical // self.page_frames
                dest = row if dst_rows is None else int(dst_rows[row])
                for page in torch.unique(pages).tolist():
                    rows = pages == page
                    out[dest, rows] = self.pages[page][physical[rows] % self.page_frames]
        return out

    def remaining(self, clip_ids, frames):
        if self._spill and len(clip_ids):
            ids = clip_ids.cpu().tolist()
            spilled = [row for row, slot in enumerate(ids) if slot in self._spill]
            if spilled:
                lengths = torch.empty_like(clip_ids, dtype=torch.int64)
                resident = [row for row, slot in enumerate(ids) if slot not in self._spill]
                if resident:
                    lengths[resident] = self._meta[clip_ids[resident], 1]
                lengths[spilled] = torch.tensor([self._slot_lengths[ids[row]] for row in spilled], device=self.device)
                return (lengths - 1 - frames).clamp_min(0).float()
        return (self._meta[clip_ids, 1] - 1 - frames).clamp_min(0).float()

    def storage_metrics(self):
        capacity = len(self.pages) * self.page_frames
        size = capacity * self.raw_dim * 4
        return {"replay/clip_frames_used": self._frames_used, "replay/clip_frames_capacity": capacity,
                "replay/clip_device_bytes": size if self.device.type == "cuda" else 0,
                "replay/clip_host_bytes": self._spill_frames * self.raw_dim * 4 + (size if self.device.type == "cpu" else 0),
                "replay/clip_device_budget_bytes": self.max_device_bytes,
                "replay/clip_accounted_device_bytes": self._accounted(),
                "replay/clip_device_frames_used": self._frames_used - self._spill_frames,
                "replay/clip_spill_bytes": self._spill_frames * self.raw_dim * 4,
                "replay/clip_spill_frames": self._spill_frames, "replay/clip_spill_slots": len(self._spill),
                "replay/clip_pages": len(self.pages), "replay/clip_page_frames": self.page_frames,
                "replay/clip_device_index_bytes": self._meta.nbytes + 8 * self._live_blocks
                if self.device.type == "cuda" else 0}

    def snapshot_cpu(self):
        from .replay_v4 import ClipBank

        n = len(self._slot_lengths)
        host = ClipBank(device="cpu", raw_dim=self.raw_dim, init_frames=self._frames_used, init_clips=n)
        host._slot_starts = [0] * n
        cursor = 0
        for slot, length in enumerate(self._slot_lengths):
            if length:
                host._slot_starts[slot] = cursor
                dest = host.frames_buf[cursor:cursor + length]
                if slot in self._spill:
                    dest.copy_(self._spill[slot])
                else:
                    for physical, part in self._runs(self._slot_blocks[slot], length):
                        dest[part].copy_(physical)
                cursor += length
        host.starts.copy_(torch.tensor(host._slot_starts, dtype=torch.int64))
        host.lengths.copy_(torch.tensor(self._slot_lengths, dtype=torch.int64))
        host._slot_lengths = self._slot_lengths.copy()
        host._slot_static = self._slot_static.copy()
        host._free_slots = self._free_slots.copy()
        host._static_hashes = self._static_hashes.copy()
        host._used_frames = cursor
        return host

    @classmethod
    def from_host(cls, host, device, *, page_frames=65536, max_device_bytes=None, reserve_device_bytes=0):
        bank = cls(device, host.raw_dim, page_frames, max_device_bytes=max_device_bytes,
                   reserve_device_bytes=reserve_device_bytes)
        n = len(host._slot_lengths)
        bank._slot_blocks = [None] * n
        bank._slot_tables = [None] * n
        bank._generations = [0] * n
        bank._slot_static = host._slot_static.copy()
        bank._slot_lengths = [0] * n
        # Preserve original IDs, preferring the static shared library when a
        # pre-existing checkpoint already exceeds the new placement budget.
        for slot in sorted(range(n), key=lambda s: not host._slot_static[s]):
            length = host._slot_lengths[slot]
            if length:
                start = host._slot_starts[slot]
                values = host.frames_buf[start:start + length]
                if not bank._try_device(slot, values, host._slot_static[slot]):
                    bank._put_spill(slot, values, host._slot_static[slot])
        bank._free_slots = host._free_slots.copy()
        bank._static_hashes = host._static_hashes.copy()
        for slot, clip in host._slot_ref.items():
            if bank._slot_lengths[slot]:
                bank._remember(clip, slot)
        bank._frames_since_gc = 0
        return bank
