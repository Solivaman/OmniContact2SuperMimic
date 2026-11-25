"""Pack only this tick's V7 resets; static reference trajectories remain in their existing banks."""
import numpy as np

from .sim import _matrix


def _normal_poses(episodes):
    """Populate precisely the same host shadows as individual _Episode resets."""
    if not episodes:
        return
    frames = np.stack([e._reset_pose[0] for e in episodes])
    twists = np.stack([e._reset_pose[1] for e in episodes])
    first = episodes[0]
    bp, bv = first.bp, first.bv
    qp = np.broadcast_to(first.m.qpos0, (len(frames), first.m.nq)).copy()
    qv = np.zeros((len(frames), first.m.nv), dtype=first.d.qvel.dtype)
    qp[:, :7], qp[:, 7:36], qp[:, bp:bp+7] = frames[:, 97:104], frames[:, 39:68], frames[:, 104:111]
    qv[:, :3], qv[:, 6:35], qv[:, bv:bv+3] = twists[:, 0], frames[:, 68:97], twists[:, 2]
    for dst, q, omega in ((slice(3,6), slice(100,104), 1), (slice(bv+3,bv+6), slice(107,111), 3)):
        rot = _matrix(frames[:, q])
        qv[:, dst] = (rot.transpose(0,2,1) @ twists[:, omega, :, None]).squeeze(-1)
    for i, e in enumerate(episodes):
        e.d.qpos[:] = qp[i]
        e.d.qvel[:] = qv[i]
        e._reset_pose = None


def _geometry_table(world, episodes):
    import torch

    slots = world._reset_geometry_slots
    before = len(slots)
    result = []
    for e in episodes:
        key = tuple(e.half)
        if key not in slots:
            slots[key] = len(slots)
        result.append(slots[key])
    if len(slots) != before:
        # Build at most one small shared table per geometry field; rebuild only when
        # the run encounters a new box size. References, worlds and frames are not cached here.
        geometries = [world._geometry(half) for half in slots]
        world._reset_geometry_table = {
            name: torch.as_tensor(np.stack([g[name] for g in geometries]), device=world.device, dtype=dst.dtype)
            for name, dst in world._model_views.items()
        }
    return result


def sync_box_context(world, episode):
    """Synchronize episode.context[3:10] (box mass, inertia, friction) with episode DR state."""
    geom = world._geometry(episode.half)
    m_scale = float(getattr(episode, "box_mass_scale", 1.0))
    base_mass = float(world._host.body_mass[world._box_body])
    episode.context[3] = np.float32(base_mass * m_scale)
    episode.context[4:7] = geom["body_inertia"][world._box_body] * m_scale
    fric = getattr(episode, "box_friction", None)
    episode.context[7:10] = (
        np.asarray(fric, dtype=np.float32)
        if fric is not None
        else np.asarray(episode._base_box_friction, dtype=np.float32)
    )


def apply_box_dr_to_stacked(world, key, vals, episodes):
    """Apply per-episode box mass/inertia/invweight0/friction randomization to stacked tensors."""
    if not getattr(world, "_has_box_dr", False) or not episodes:
        return vals
    import torch

    m_scales = torch.as_tensor(
        [float(ep.box_mass_scale) for ep in episodes],
        dtype=torch.float32,
        device=world.device,
    )
    if key in ("body_mass", "body_subtreemass"):
        vals = vals.clone()
        vals[:, world._box_body] *= m_scales
    elif key == "body_inertia":
        vals = vals.clone()
        vals[:, world._box_body] *= m_scales[:, None]
    elif key == "body_invweight0":
        vals = vals.clone()
        vals[:, world._box_body] /= m_scales[:, None]
    elif key == "dof_invweight0":
        vals = vals.clone()
        vals[:, world._bv:world._bv + 6] /= m_scales[:, None]
    elif key == "geom_friction" and getattr(world, "_box_friction_dr", False):
        vals = vals.clone()
        frics = np.stack([
            ep.box_friction if ep.box_friction is not None else ep._base_box_friction
            for ep in episodes
        ]).astype(np.float32)
        vals[:, world._box] = torch.as_tensor(frics, dtype=vals.dtype, device=world.device)
    return vals


def reset_batch(world, ids, source):
    import mujoco_warp as mjw
    import torch
    import warp as wp

    episodes = [world.envs[i] for i in ids]
    normal_rows = []
    pose_episodes = []
    for row, e in enumerate(episodes):
        snap = getattr(e, "_deferred_snapshot", None)
        was_deferred = False
        if snap is not None and source is None and e.pending_recovery:
            e.d.qpos[:] = snap["qpos"]
            e.d.qvel[:] = snap["qvel"]
            e.d.mocap_pos[:] = snap["mocap_pos"]
            e.half = snap["half"].copy()
            e.goal = snap["goal"].copy()
            e.name = snap["name"]
            if "box_mass_scale" in snap:
                e.box_mass_scale = float(snap["box_mass_scale"])
            if "box_friction" in snap:
                sf = snap["box_friction"]
                e.box_friction = None if sf is None else sf.copy()
            was_deferred = True
        e._deferred_snapshot = None
        e._defer_main_pose = e.in_reset = True
        try:
            e.reset(source)
        finally:
            e._defer_main_pose = e.in_reset = False
        if e.source != "recovery":
            normal_rows.append(row)
            pose_episodes.append(e)
        elif was_deferred:
            normal_rows.append(row)
        sync_box_context(world, e)

    _normal_poses(pose_episodes)
    normal_episodes = [episodes[i] for i in normal_rows]
    variants = _geometry_table(world, normal_episodes)
    # Pack all per-episode float fields into a single H2D copy, including the
    # live state of successful recoveries. MjWarp resets ordinary and deferred rows.
    model = world._host
    n = len(ids)
    dm = 3 * model.nmocap
    cuts = np.cumsum([model.nq, model.nv, dm, 16, 3, 3]).tolist()
    floats = np.empty((n, cuts[-1]), np.float32)
    host = np.split(floats, cuts[:-1], axis=1)
    ints = np.zeros((5, n), np.int64)
    ints[0] = ids
    for row, e in enumerate(episodes):
        host[0][row] = e.d.qpos
        host[1][row] = e.d.qvel
        host[2][row] = e.d.mocap_pos.ravel()
        host[3][row] = e.context
        host[4][row] = e._initial_box
        host[5][row] = e.goal
        cat = getattr(e, "_clip_catalogue", None)
        if e.source != "recovery" and cat is not None and cat.last_box_extent_z is not None:
            extent_z = cat.last_box_extent_z
        else:
            extent_z = float(np.abs(_matrix(e.clip[-1, 107:111])[2]) @ e.half)
        goal_bottom = float(e.goal[2] - extent_z)
        tabletop = float(e.d.mocap_pos[world._table2_id, 2] + world._table2_half_z)
        ints[1:, row] = len(e.clip), e.index, e._lifted, abs(goal_bottom) < abs(goal_bottom - tabletop)

    buf = torch.as_tensor(floats, device=world.device)
    qp, qv, mocap, ctx, initial, goal = torch.tensor_split(buf, cuts[:-1], dim=1)
    idx, length, frame, held, floor = torch.as_tensor(ints, device=world.device).unbind(0)
    with wp.ScopedDevice(str(world.device)), world._stream():
        if normal_rows:
            norm, slots = torch.as_tensor(np.stack((normal_rows, variants)), device=world.device).unbind(0)
            changed = idx[norm]
            world._reset_mask_torch.zero_()
            world._reset_mask_torch[changed] = True
            stream = torch.cuda.current_stream(world.device)
            world._cuda_stream.wait_stream(stream)
            mjw.reset_data(world.m, world.d, world._reset_mask_wp)
            stream.wait_stream(world._cuda_stream)
            for key, dst in world._model_views.items():
                vals = apply_box_dr_to_stacked(
                    world, key, world._reset_geometry_table[key][slots], normal_episodes
                )
                dst.index_copy_(0, changed, vals)
            world._qpos.index_copy_(0, changed, qp[norm])
            world._qvel.index_copy_(0, changed, qv[norm])
            world._mocap.index_copy_(0, changed, mocap[norm].reshape(-1, model.nmocap, 3))
        state = torch.cat((qp[:, :36], qv[:, :35], qp[:, world._bp:world._bp+7], qv[:, world._bv:world._bv+6]), dim=1)
        for dst, rows in ((world._state_gpu, state), (world._context_gpu, ctx), (world._initial_box, initial),
                          (world._goal_gpu, goal), (world._goal_on_floor, floor.bool()), (world._clip_len, length.int()),
                          (world._env_index, frame.int()), (world._lifted, held.bool()), (world._carried_steps, held.int()*2)):
            dst.index_copy_(0, idx, rows)
        world._lost_steps[idx] = 0
        world._stable_steps[idx] = 0
