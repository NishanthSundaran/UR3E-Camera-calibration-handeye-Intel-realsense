"""Rigid-transform utilities used by the hand-eye solvers.

All transforms here are stored as (R, t) tuples where:
  - R is a 3x3 numpy rotation matrix
  - t is a 3x1 numpy translation column vector
Applying the transform to a point p (3x1) is: p_out = R @ p + t.
"""

from __future__ import annotations
from typing import Tuple

import math
import numpy as np
import cv2


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R


def transform_msg_to_R_t(tf_msg) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (R, t) from a geometry_msgs/TransformStamped."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    tvec = np.array([[t.x], [t.y], [t.z]])
    return R, tvec


def matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix -> (x, y, z, w) quaternion."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Rotation matrix -> roll, pitch, yaw (ZYX convention, URDF style)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def invert_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a rigid transform: (R, t) -> (R^T, -R^T @ t)."""
    R_inv = R.T
    t_inv = -R.T @ t
    return R_inv, t_inv


def compose_transforms(
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two transforms: returns T1 * T2."""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t
