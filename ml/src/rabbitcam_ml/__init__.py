"""RabbitCam posture-classification tools.

The package deliberately predicts a single-frame posture only. Persistence,
health interpretation, alerting, and notifications belong to the homeserver.
"""

from .labels import CLASS_NAMES, CLASS_TO_INDEX

__all__ = ["CLASS_NAMES", "CLASS_TO_INDEX"]
__version__ = "0.1.0"

