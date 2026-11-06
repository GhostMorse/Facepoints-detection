"""Tests for the facial landmark permutation used by horizontal flipping."""

import numpy as np

from facepoints.data.augment import swap_keypoints


def test_swap_keypoints_preserves_the_original_mapping():
    """Verify the original 14-point index mapping."""
    points = np.arange(28).reshape(14, 2)
    expected = points[[3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]]
    np.testing.assert_array_equal(swap_keypoints(points), expected)
