"""V5 survival-weighted policy objective for the V4 carry-box trainer."""
import torch

from .model import DT, TRACKING_REWARD_WEIGHTS
from .train import GAMMA
from .v3 import LearnerV3


HEIGHT_THRESHOLD_M = .25
GRAVITY_Z_THRESHOLD = .8
TERMINATION_SOFTNESS = .05
VALUE_MAX = DT * sum(TRACKING_REWARD_WEIGHTS.values()) / (1. - GAMMA)


def soft_termination_probability(after, target, *,
                                 height_threshold=HEIGHT_THRESHOLD_M,
                                 gravity_z_threshold=GRAVITY_Z_THRESHOLD,
                                 softness=TERMINATION_SOFTNESS):
    """Score the post-step pelvis against its reference. Both quaternions are unit wxyz."""
    height_error = (after[..., 2] - target[..., 99]).abs()
    # Local gravity z is 2*(x²+y²)-1. Yaw alone leaves it unchanged.
    gravity_z = 2 * after[..., 4:6].square().sum(-1) - 1
    ref_gravity_z = 2 * target[..., 101:103].square().sum(-1) - 1
    tilt_error = (gravity_z - ref_gravity_z).abs()
    normalized = torch.maximum(height_error / height_threshold, tilt_error / gravity_z_threshold)
    return torch.sigmoid((normalized - 1.) / softness)


class LearnerV5(LearnerV3):
    def _continuation(self, after, target, task_alive):
        args = getattr(self, "args", None)
        p = soft_termination_probability(
            after,
            target,
            height_threshold=float(getattr(args, "termination_height_m", HEIGHT_THRESHOLD_M)),
            gravity_z_threshold=float(getattr(args, "termination_gravity_z", GRAVITY_Z_THRESHOLD)),
            softness=float(getattr(args, "termination_softness", TERMINATION_SOFTNESS)),
        )
        # Only a hard carry-task boundary invalidates subsequent critic examples.
        return (1. - p) * task_alive, task_alive

    def _limit_return(self, value):
        vmax = float(getattr(getattr(self, "args", None), "critic_value_max", VALUE_MAX))
        return value.clamp(max=vmax)
