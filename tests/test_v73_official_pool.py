"""Unit and MuJoCo equivalence tests for V7.3 official CF-Gen pools and full-MoCap splits."""
from contextlib import nullcontext
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import mujoco as mj
import numpy as np
import pytest
import torch

from carrywm.data import RAW_DIM, split_paths
from carrywm.geometry_v7 import valid_start_mask
from carrywm.official_pool import (
    DEFAULT_TRAIN_SEED_A,
    DEFAULT_TRAIN_SEED_B,
    DEFAULT_VAL_SEED_A,
    DEFAULT_VAL_SEED_B,
    OfficialPackClip,
    RESERVED_SEEDS,
    TABLE_HALF_SIZES,
    build_official_pack,
    geometry_v7_sha256,
    load_official_pack,
    sample_mixed_official_specs,
    save_official_pack,
)
from carrywm.sim import CarryEnv, _compile_scene, _scene
from carrywm.stages_v7 import StageSampler
from carrywm.train import validate
from carrywm.train_v7 import args_from_cli as v7_args_from_cli


def _find_sim_root():
    paths = (
        os.environ.get("CARRY_SIM_ROOT"),
        os.environ.get("SIM2SIM_ROOT"),
        os.environ.get("OMNICONTACT_SIM2SIM"),
        str(Path(__file__).resolve().parents[2] / "papers/OmniContact/code-sim2sim"),
        "../OmniContact_sim2sim",
        "../papers/OmniContact/code-sim2sim",
    )
    for entry in paths:
        if entry and (Path(entry) / "g1_description/g1_29dof.xml").is_file():
            return Path(entry)
    return None


@pytest.fixture
def sim_root():
    root = _find_sim_root()
    if root is None:
        pytest.skip("Set CARRY_SIM_ROOT to the released MuJoCo checkout")
    return root


def _dummy_clip(idx: int, *, length: int = 32, bench_type: str = "bench_a") -> OfficialPackClip:
    frames = np.zeros((length, RAW_DIM), dtype=np.float32)
    frames[:, 99] = 0.778
    frames[:, 100] = 1.0
    frames[:, 104:107] = [1.2, 0.0, 0.5]
    frames[:, 107] = 1.0
    for j in range(5):
        frames[:, 7 * j + 3] = 1.0
    frames[10:25, 37:39] = 1.0
    frames[10:25, 106] = 0.9
    half = np.array([0.21, 0.16, 0.18], dtype=np.float64) if bench_type == "bench_a" else np.full(3, 0.15)
    tables = np.array([[1.2, 0.0, 0.5 - half[2] - 0.01], [2.2, 0.0, 0.5 - half[2] - 0.01]], dtype=np.float64)
    return OfficialPackClip(
        ep_id=idx,
        label=f"stub_{bench_type}_{idx:03d}",
        bench_type=bench_type,
        frames=frames,
        phase=np.zeros(length, dtype=np.int32),
        half=half,
        tables=tables,
        goal=np.array([2.2, 0.0, 0.5], dtype=np.float32),
        valid_mask=np.ones(length, dtype=bool),
    )


def test_split_paths_all_keeps_every_motion(tmp_path):
    for source in range(15):
        for dims in ("20-20-20", "30-20-30"):
            (tmp_path / f"carry_box_{dims}_{source}_12345_000.npz").touch()
    all_paths = split_paths(tmp_path, "all")
    assert len(all_paths) == 30
    union = set().union(*(set(split_paths(tmp_path, s)) for s in ("train", "val", "test")))
    assert set(all_paths) == union


def test_sample_mixed_official_specs_proportions_clearances_and_seeds():
    specs = sample_mixed_official_specs(40, 20, DEFAULT_VAL_SEED_A, DEFAULT_VAL_SEED_B, prefix="val")
    assert len(specs) == 60
    for k in (3, 6, 12, 24, 60):
        prefix = specs[:k]
        assert sum(s["bench_type"] == "bench_a" for s in prefix) == 2 * (k // 3)
        assert sum(s["bench_type"] == "bench_b" for s in prefix) == (k // 3)

    for s in specs:
        half = s["half"]
        ix, iy, iz = s["init_pos"]
        gx, gy, gz = s["goal_pos"]
        assert -5.0 <= ix <= 5.0 and -5.0 <= iy <= 5.0
        assert -5.0 <= gx <= 5.0 and -5.0 <= gy <= 5.0
        assert half[2] - 1e-6 <= iz <= 0.80 + 1e-6
        assert half[2] - 1e-6 <= gz <= 0.80 + 1e-6
        foot_r = float(max(half[0], half[1]))
        assert math.hypot(ix, iy) >= foot_r + 0.65 - 1e-6
        assert math.hypot(gx, gy) >= foot_r + 0.65 - 1e-6
        assert math.hypot(gx - ix, gy - iy) >= 2.0 * foot_r + 0.60 - 1e-6
        if s["bench_type"] == "bench_a":
            assert np.all((half >= 0.10 - 1e-6) & (half <= 0.25 + 1e-6))
        else:
            np.testing.assert_allclose(half, [0.15, 0.15, 0.15])

    assert RESERVED_SEEDS.isdisjoint({DEFAULT_VAL_SEED_A, DEFAULT_VAL_SEED_B, DEFAULT_TRAIN_SEED_A, DEFAULT_TRAIN_SEED_B})
    for bad in RESERVED_SEEDS:
        with pytest.raises(ValueError):
            sample_mixed_official_specs(2, 1, bad, DEFAULT_VAL_SEED_B)
        with pytest.raises(ValueError):
            sample_mixed_official_specs(2, 1, DEFAULT_VAL_SEED_A, bad)


def test_pack_roundtrip_and_stale_valid_mask_invalidation(tmp_path):
    clips = [_dummy_clip(0, bench_type="bench_a"), _dummy_clip(1, bench_type="bench_b")]
    pack_path = tmp_path / "pack.npz"
    meta = {"geometry_v7_sha256": geometry_v7_sha256(), "seed_a": 20261, "seed_b": 20262}
    save_official_pack(pack_path, clips, meta)
    assert pack_path.with_suffix(".json").is_file()

    loaded, loaded_meta = load_official_pack(pack_path)
    assert loaded_meta["count"] == 2
    assert len(loaded) == 2
    for a, b in zip(clips, loaded):
        assert a.label == b.label and a.bench_type == b.bench_type
        np.testing.assert_array_equal(a.frames, b.frames)
        np.testing.assert_array_equal(a.phase, b.phase)
        np.testing.assert_allclose(a.half, b.half)
        np.testing.assert_allclose(a.tables, b.tables)
        np.testing.assert_allclose(a.goal, b.goal)
        np.testing.assert_array_equal(a.valid_mask, b.valid_mask)

    stale_path = tmp_path / "stale_pack.npz"
    save_official_pack(stale_path, clips, {"geometry_v7_sha256": "deadbeef"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stale_loaded, _ = load_official_pack(stale_path)
    assert any("geometry_v7.py changed" in str(w.message) for w in caught)
    assert all(c.valid_mask is None for c in stale_loaded)


def test_stage_sampler_precomputed_valid_matches_dynamic(sim_root):
    clip = _dummy_clip(0, length=48, bench_type="bench_a")
    dyn_valid = valid_start_mask(sim_root, clip.frames, clip.half, "cfgen", clip.phase)

    sampler_dyn = StageSampler()
    sampler_pre = StageSampler()
    cat_dyn = sampler_dyn.catalogue(
        sim_root, clip.frames, clip.half, "cfgen", clip.phase, clip.tables, TABLE_HALF_SIZES
    )
    with patch("carrywm.geometry_v7.valid_start_mask", side_effect=AssertionError("must skip dynamic check")):
        cat_pre = sampler_pre.catalogue(
            sim_root,
            clip.frames,
            clip.half,
            "cfgen",
            clip.phase,
            clip.tables,
            TABLE_HALF_SIZES,
            precomputed_valid=dyn_valid,
        )

    np.testing.assert_array_equal(cat_dyn.valid, cat_pre.valid)
    np.testing.assert_array_equal(cat_dyn.labels, cat_pre.labels)
    for b_dyn, b_pre in zip(cat_dyn.bins, cat_pre.bins):
        np.testing.assert_array_equal(b_dyn, b_pre)
    np.testing.assert_array_equal(cat_dyn.starts_holding, cat_pre.starts_holding)
    assert cat_dyn.last_box_extent_z == cat_pre.last_box_extent_z


def test_carry_env_samples_cfgen_pool_and_all_mocap_at_3_7(sim_root, tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    for idx in range(20):
        (motion_dir / f"carry_box_20-20-20_{idx}_12345_000.npz").touch()
    pool = [_dummy_clip(i, bench_type="bench_a" if i % 3 != 2 else "bench_b") for i in range(6)]
    env = CarryEnv(
        sim_root,
        motion_dir,
        seed=17,
        split="train",
        cfgen_ratio=0.3,
        v7=True,
        motion_split="all",
        cfgen_pool=pool,
    )
    assert len(env.paths) == 20
    assert env.stage_sampler is not None

    seen_sources = []
    seen_npz = set()
    seen_pool = set()

    def fake_npz(self, path, source, stage):
        assert source == "npz"
        seen_npz.add(path.name)
        self.source, self.name, self.bench_type = "npz", path.stem, None
        self.clip = pool[0].frames
        self.index = 0
        self.half = np.full(3, 0.15)
        self.goal = np.zeros(3, dtype=np.float32)

    def fake_pool(self, entry, stage):
        seen_pool.add(entry.label)
        self.source, self.name, self.bench_type = "cfgen", entry.label, entry.bench_type
        self.clip = entry.frames
        self.index = 0
        self.half = entry.half.copy()
        self.goal = entry.goal.copy()

    env._bind_scene((0.15, 0.15, 0.15))

    with patch.object(CarryEnv, "_load_main_episode", fake_npz), patch.object(
        CarryEnv, "_load_cfgen_pool_episode", fake_pool
    ):
        for _ in range(400):
            env.reset()
            seen_sources.append(env.source)

    cfgen_frac = np.mean([s == "cfgen" for s in seen_sources])
    assert 0.24 <= cfgen_frac <= 0.36
    assert len(seen_npz) == 20
    assert len(seen_pool) == 6


def test_validate_with_val_pack_emits_cfgen_only_and_bench_breakdown(tmp_path):
    clips = [
        _dummy_clip(0, bench_type="bench_a"),
        _dummy_clip(1, bench_type="bench_a"),
        _dummy_clip(2, bench_type="bench_b"),
    ]
    pack_path = tmp_path / "val_pack.npz"
    save_official_pack(pack_path, clips, {"geometry_v7_sha256": geometry_v7_sha256()})

    class FakeEnv:
        def __init__(self, *_args, **kwargs):
            self.cfgen_pool = tuple(kwargs["cfgen_pool"])
            self.pending_recovery = False
            self._step_i = 0
            self.index = 0
            self.source = "cfgen"
            self.name = ""
            self.clip = self.cfgen_pool[0].frames
            self.state = np.zeros(84, dtype=np.float32)
            self.context = np.full(16, 0.15, dtype=np.float32)
            self.state[[3, 74]] = 1.0
            self.state[2] = 0.78

        def reset(self, source=None):
            if source == "cfgen":
                entry = self.cfgen_pool[0]
                self.source = "cfgen"
                self.name = entry.label
                self.clip = entry.frames
                self._step_i = 0
                self.index = 0
                self.pending_recovery = False
            else:
                self.source = "recovery"
                self._step_i = 0
                self.index = 0
                self.pending_recovery = False

        def step(self, _action):
            self._step_i += 1
            self.index = min(self._step_i, len(self.clip) - 1)
            done = self._step_i >= 2
            # Clip 0: initial success; Clip 1: drop then recovery success; Clip 2: fallen.
            if self.source == "recovery":
                info = dict(success=done, drop=False, timeout=False, fallen=False, lifted=done, grounded=False, supported=done)
            elif self.name.endswith("000"):
                info = dict(success=done, drop=False, timeout=False, fallen=False, lifted=done, grounded=False, supported=done)
            elif self.name.endswith("001"):
                info = dict(success=False, drop=done, timeout=False, fallen=False, lifted=True, grounded=done, supported=done)
                self.pending_recovery = bool(done)
            else:
                info = dict(success=False, drop=False, timeout=False, fallen=done, lifted=False, grounded=False, supported=False)
            return self.state.copy(), np.ones(6, dtype=np.float32), done, info

    class FakeNet:
        device = torch.device("cpu")
        box_features = True
        args = SimpleNamespace(drop_penalty=1.0, goal_mode="full")

        class _Geo:
            @staticmethod
            def observation(s, _p, _c):
                return torch.zeros(len(s), 141)

            @staticmethod
            def reward_terms(s, *_args):
                return {"dummy": torch.zeros(len(s))}

        geo = _Geo()

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def reference(s, _raw):
            return torch.zeros(len(s), 11, 122)

        @staticmethod
        def actor(goal, _hist):
            return torch.zeros(len(goal), 29)

        @staticmethod
        def wm(state, _act, _ctx):
            return state.clone(), torch.zeros(len(state), 6)

    args = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", str(tmp_path),
        "--val-pack", str(pack_path), "--eval-episodes", "3", "--eval-workers", "1",
    ])
    with patch("carrywm.train.CarryEnv", FakeEnv):
        metrics = validate(FakeNet(), args)

    assert not any(k.startswith("eval/npz/") for k in metrics)
    assert metrics["eval/cfgen/done"] == 3
    assert metrics["eval/cfgen/success_rate"] == pytest.approx(1 / 3)
    assert metrics["eval/cfgen/end_to_end_success_rate"] == pytest.approx(2 / 3)
    assert metrics["eval/cfgen_a/done"] == 2
    assert metrics["eval/cfgen_a/success_rate"] == pytest.approx(0.5)
    assert metrics["eval/cfgen_a/end_to_end_success_rate"] == pytest.approx(1.0)
    assert metrics["eval/cfgen_b/done"] == 1
    assert metrics["eval/cfgen_b/success_rate"] == pytest.approx(0.0)
    assert metrics["eval/recovery/done"] == 1
    assert metrics["eval/recovery/success_rate"] == pytest.approx(1.0)
    assert "eval/cfgen/dynamics/h01/all/box_position_m" in metrics
    assert "eval/dynamics/h01/all/box_position_m" in metrics


def test_v7_cli_accepts_v73_flags_with_v72_defaults():
    default_args = v7_args_from_cli(["--sim-root", ".", "--motion-dir", ".", "--output-dir", "."])
    assert default_args.motion_split == "train"
    assert default_args.cfgen_train_pack is None
    assert default_args.val_pack is None

    v73_args = v7_args_from_cli([
        "--sim-root", ".", "--motion-dir", ".", "--output-dir", ".",
        "--motion-split", "all", "--cfgen-ratio", "0.3",
        "--cfgen-train-pack", "train.npz", "--val-pack", "val.npz",
    ])
    assert v73_args.motion_split == "all"
    assert v73_args.cfgen_ratio == pytest.approx(0.3)
    assert v73_args.cfgen_train_pack == Path("train.npz")
    assert v73_args.val_pack == Path("val.npz")


@pytest.mark.parametrize("headless", [True, False])
def test_fast_scene_resize_matches_full_xml_compilation(sim_root, headless):
    half = (0.23, 0.12, 0.19)
    m_fast = _scene(sim_root, half, headless=headless)
    m_full = _compile_scene(sim_root, half, headless=headless)

    g_fast = m_fast.geom("box_geom").id
    g_full = m_full.geom("box_geom").id
    b_fast = m_fast.body("box").id
    b_full = m_full.body("box").id

    np.testing.assert_allclose(m_fast.geom_size[g_fast], m_full.geom_size[g_full], atol=1e-7)
    np.testing.assert_allclose(m_fast.geom_aabb[g_fast], m_full.geom_aabb[g_full], atol=1e-7)
    np.testing.assert_allclose(m_fast.geom_rbound[g_fast], m_full.geom_rbound[g_full], atol=1e-7)
    np.testing.assert_allclose(m_fast.body_inertia[b_fast], m_full.body_inertia[b_full], atol=1e-7)
    np.testing.assert_allclose(m_fast.body_invweight0, m_full.body_invweight0, atol=1e-6)
    np.testing.assert_allclose(m_fast.dof_invweight0, m_full.dof_invweight0, atol=1e-6)
    if not headless:
        gg_fast = m_fast.geom("ghost_box_geom").id
        gg_full = m_full.geom("ghost_box_geom").id
        np.testing.assert_allclose(m_fast.geom_size[gg_fast], m_full.geom_size[gg_full], atol=1e-7)
        np.testing.assert_allclose(m_fast.geom_aabb[gg_fast], m_full.geom_aabb[gg_full], atol=1e-7)

    d_fast, d_full = mj.MjData(m_fast), mj.MjData(m_full)
    bp = int(m_fast.joint("box").qposadr[0])
    for d in (d_fast, d_full):
        d.qpos[:3] = [0.0, 0.0, 0.778]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[bp : bp + 3] = [0.35, 0.0, half[2] + 0.002]
        d.qpos[bp + 3 : bp + 7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(m_fast if d is d_fast else m_full, d)

    dist_fast = mj.mj_geomDistance(m_fast, d_fast, g_fast, m_fast.geom("floor").id, 0.2, None)
    dist_full = mj.mj_geomDistance(m_full, d_full, g_full, m_full.geom("floor").id, 0.2, None)
    assert abs(dist_fast - dist_full) < 1e-7

    for _ in range(10):
        mj.mj_step(m_fast, d_fast)
        mj.mj_step(m_full, d_full)

    np.testing.assert_allclose(d_fast.qpos, d_full.qpos, atol=1e-6)
    np.testing.assert_allclose(d_fast.qvel, d_full.qvel, atol=1e-6)


def test_build_official_pack_real_planner_smoke(sim_root, tmp_path):
    out_pack = tmp_path / "mini_pack.npz"
    summary = build_official_pack(
        sim_root,
        out_pack,
        n_a=2,
        n_b=1,
        seed_a=20261,
        seed_b=20262,
        workers=1,
        compute_valid_mask=True,
        require_valid_starts=True,
        prefix="smoke",
    )
    assert summary["count"] == 3
    clips, _ = load_official_pack(out_pack)
    assert [c.bench_type for c in clips] == ["bench_a", "bench_a", "bench_b"]
    assert all(c.valid_mask is not None for c in clips)
