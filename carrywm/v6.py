"""V6 survival-focused objective, reward balancing, observation noise, and teacher schedule."""
from .model import DT, TRACKING_REWARD_WEIGHTS
from .train import GAMMA
from .v5 import LearnerV5


HEIGHT_THRESHOLD_M = 0.25
GRAVITY_Z_THRESHOLD = 0.45
TERMINATION_SOFTNESS = 0.08
BALANCE_REWARD_WEIGHT = 0.5
ACTION_RATE_WEIGHT = -0.25
ACTION_L2_WEIGHT = -0.005
OBS_NOISE_SCALE = 1.0
TEACHER_INIT_FRACTION = 0.25
TEACHER_DECAY_ITERS = 400
TEACHER_BC_WEIGHT = 0.5
TEACHER_BC_BATCH = 1024


def v6_tracking_weights(balance_weight=BALANCE_REWARD_WEIGHT):
    """Build tracking reward weights with elevated whole-body balance terms."""
    weights = dict(TRACKING_REWARD_WEIGHTS)
    for key in ("anchor_pos", "anchor_rot", "body_pos", "body_rot"):
        weights[key] = float(balance_weight)
    return weights


def v6_value_max(balance_weight=BALANCE_REWARD_WEIGHT):
    """Upper bound on discounted infinite-horizon tracking return for V6 rewards."""
    weights = v6_tracking_weights(balance_weight)
    return DT * sum(weights.values()) / (1.0 - GAMMA)


def teacher_schedule(iteration, init_value, decay_iters):
    """Linearly decay a teacher schedule from init_value at iter 1 to 0 after decay_iters."""
    if init_value <= 0.0 or decay_iters <= 0 or iteration > decay_iters:
        return 0.0
    return float(init_value) * max(0.0, 1.0 - (iteration - 1) / decay_iters)


class LearnerV6(LearnerV5):
    """V6 learner inheriting V5 soft termination, Critic normalization, obs noise, and BC support."""
