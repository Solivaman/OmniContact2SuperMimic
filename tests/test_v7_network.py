import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from carrywm.export import export_actor
from carrywm.model import BOX_GOAL_DIM, CarryGeometry, WorldModel
from carrywm.train import _CpuEvalNet, HORIZON, OBS_NOISE_BASE, WM_STEPS, frozen
from carrywm.v7 import LearnerV7


@pytest.fixture
def root():
    paths = (os.environ.get("SIM2SIM_ROOT"), os.environ.get("OMNICONTACT_SIM2SIM"),
             "../OmniContact_sim2sim", "../papers/OmniContact/code-sim2sim")
    for entry in paths:
        if entry and (Path(entry) / "g1_description/g1_29dof.xml").is_file():
            return Path(entry)
    pytest.skip("Set SIM2SIM_ROOT to the released MuJoCo model")


def pose():
    state = torch.zeros(1, 84)
    state[:, 2] = .8
    state[:, 3] = state[:, 74] = 1.
    state[:, 71:74] = torch.tensor([.3, .12, .6])
    return state


def test_box_reference_appends_expected_local_pose(root):
    state = pose()
    # Robot faces +Y. A box one metre away in world +Y is straight ahead.
    state[:, 3:7] = torch.tensor([.5**.5, 0., 0., .5**.5])
    target = torch.zeros(1, 11, 111)
    target[..., 104:107] = state[..., None, :3] + torch.tensor([0., 1., .25])
    target[..., 107:111] = state[..., None, 3:7]
    original = CarryGeometry(root).reference(state, target)
    goal = CarryGeometry(root, box_goal=True).reference(state, target)
    assert goal.shape == (1, 11, 122)
    torch.testing.assert_close(goal[..., :113], original)
    expected = torch.tensor([1., 0., .25, 1., 0., 0., 1., 0., 0.])
    torch.testing.assert_close(goal[..., 113:], expected.expand_as(goal[..., 113:]))


def test_wm_pad_positions_match_physical_palm_centres(root):
    import mujoco
    import numpy as np

    state = pose()
    state[:, 7:36] = torch.linspace(-.12, .35, 29)
    state[:, 74:78] = torch.tensor([.5**.5, 0., .5**.5, 0.])
    ctx = torch.ones(1, 16)
    ctx[:, :3] = torch.tensor([.1, .15, .2])
    geo = CarryGeometry(root, box_goal=True)
    m = mujoco.MjModel.from_xml_path(str(root / "g1_description/g1_29dof.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = state[0, :36].numpy()
    mujoco.mj_forward(m, d)
    box_rot = np.empty(9)
    mujoco.mju_quat2Mat(box_rot, state[0, 74:78].double().numpy())
    expected = []
    # Distal contact edges from the released, mirrored G1 hand collision boxes.
    for side, dy in (("left", -.02895897), ("right", .02895897)):
        wrist = d.body(side + "_wrist_yaw_link")
        pad = wrist.xpos + wrist.xmat.reshape(3, 3) @ [.16911956, dy, 0.]
        expected.extend((box_rot.reshape(3, 3).T @ (pad - state[0, 71:74].numpy())) / ctx[0, :3].numpy())
    local = geo.palms_in_box(state, ctx)
    torch.testing.assert_close(local[0], torch.tensor(expected).float(), rtol=1e-5, atol=1e-5)
    fk = geo.forward_kinematics(state)
    wm = WorldModel(16, width=16, palm_geometry=geo)
    with patch.object(geo, "forward_kinematics", side_effect=AssertionError("unexpected FK recomputation")):
        action = torch.ones(1, 29, requires_grad=True)
        with frozen(wm):
            next_state, _ = wm(state, action, ctx, fk=fk)
            next_state[:, 80].sum().backward()
        assert action.grad is not None and action.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in wm.parameters())


def test_wm_recurrence_uses_predicted_states_and_export_uses_122(root, tmp_path):
    args = SimpleNamespace(device="cpu", sim_root=root, goal_mode="full", lr=1e-3, precision="fp32",
                           encoder_width=16, decoder_width=16, wm_width=16, drop_penalty=1.,
                           critic_norm=True, obs_noise_scale=1.)
    net = LearnerV7(args)
    assert net.wm.trunk[0].in_features == 84 + 29 + 16 + 6
    assert net.critic.value[0].in_features == 84 + 11 * 122 + 16
    assert net.critic.inputs.mean.numel() == net.critic.value[0].in_features
    torch.testing.assert_close(net._obs_noise_std, torch.tensor(OBS_NOISE_BASE))
    truth = pose()[:, None].repeat(1, WM_STEPS + 1, 1)
    truth[:, 1:, 71] = 4.
    actions = torch.zeros(1, WM_STEPS, 29)
    ctx = torch.ones(1, 16) * .15
    labels = torch.zeros(1, WM_STEPS, 6)
    palm_states = []
    original = net.geo.palms_in_box
    def record(state, ctx, fk=None):
        palm_states.append(state)
        return original(state, ctx, fk)
    with patch.object(net.geo, "palms_in_box", side_effect=record):
        loss, _ = net._wm_rollout_loss(truth, actions, labels, ctx)
        loss.backward()
    assert len(palm_states) == WM_STEPS
    torch.testing.assert_close(palm_states[0], truth[:, 0])
    assert palm_states[1].requires_grad
    assert all(s[0, 71].item() < 1. for s in palm_states)
    assert torch.isfinite(net.wm.box.weight.grad).all()

    # Backpropagate a complete imagined return through the frozen V7 dynamics.
    net.wm.zero_grad(set_to_none=True)
    state = pose()
    raw = torch.zeros(1, HORIZON + 51, 111)
    raw[..., 97:104] = state[:, None, :7]
    raw[..., 104:111] = state[:, None, 71:78]
    for i in range(5):
        raw[..., 7*i + 3] = 1.
    inputs = (state, actions[:, 0], ctx, torch.zeros(1, 10, 141), raw, torch.tensor([80.]))
    with frozen(net.wm, net.critic):
        returns, _, _, goals, *_ = net._rollout(inputs)
        (-returns.mean()).backward()
    assert len(goals) == HORIZON + 1 and goals[-1].shape == (1, 11, 122)
    assert torch.isfinite(net.actor.decoder[-1].weight.grad).all()
    assert net.actor.decoder[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.wm.parameters())

    # Standalone inference and multiprocess evaluation choose the checkpoint's goal width.
    state_dict = net.actor.state_dict()
    cpu = _CpuEvalNet(root, "full", 1., 16, 16, state_dict)
    assert cpu.geo.box_goal and cpu.actor.goals.mean.numel() == BOX_GOAL_DIM
    ckpt = tmp_path / "checkpoint.pt"
    output = tmp_path / "actor.pt"
    torch.save(dict(config=vars(args), models=[{}, state_dict, {}]), ckpt)
    export_actor(ckpt, output)
    x, h = torch.randn(1, 11, 122), torch.randn(1, 10, 141)
    torch.testing.assert_close(torch.jit.load(str(output))(x, h), cpu.actor(x, h).clamp(-10, 10))
