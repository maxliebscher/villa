from __future__ import annotations

import numpy as np


def looks_like_channel_first_map(array: np.ndarray, num_classes: int) -> bool:
    """Return True for 4D ``(B, C, H, W)`` one-hot/probability label maps.

    Four-dimensional inputs are ambiguous in this package: they can be 2D
    channel-first maps or 3D hard-label volumes. Use the ground-truth tensor as
    the tie-breaker because hard-label volumes should not be collapsed across
    depth just because ``Z == num_classes``.
    """
    if array.ndim != 4 or array.shape[1] != num_classes:
        return False
    if num_classes <= 1:
        return True

    finite = np.isfinite(array)
    if not np.all(finite):
        return False

    in_probability_range = np.logical_and(array >= 0.0, array <= 1.0)
    if not np.all(in_probability_range):
        return False

    channel_sum = np.sum(array, axis=1)
    return bool(np.allclose(channel_sum, 1.0, atol=1e-5))
