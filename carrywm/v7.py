"""Box-conditioned carry policy trained through a palm-aware world model."""
from .v5 import LearnerV5


class LearnerV7(LearnerV5):
    """Use V6's configured reward and survival objective with a randomly initialized actor."""

    box_features = True
