"""Pose math for the PolicyIK MuJoCo port.

Quaternions are (w, x, y, z), matching MuJoCo and the original PolicyIK code.
Rotation errors use the matrix/quaternion logarithm, ``Log(R)^vee``, the same
representation the paper feeds to the policy and scores success against.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_conjugate",
    "quat_multiply",
    "quat_to_mat",
    "quat_error_rotvec_body",
    "rotvec_from_quat",
    "pose_error",
]


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a (w, x, y, z) quaternion."""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, dtype=float)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """``Log(R)^vee`` of a (w, x, y, z) quaternion: an axis-angle 3-vector."""
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[0] < 0.0:  # shortest rotation
        q = -q
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return 2.0 * v  # small-angle limit
    angle = 2.0 * np.arctan2(nv, q[0])
    return v * (angle / nv)


def quat_error_rotvec_body(curr: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """End-effector-frame rotation error ``Log(R_curr^T R_goal)^vee``.

    Port of ``quat_error_rotvec_body`` in the original ``mdp.py``.
    """
    return rotvec_from_quat(quat_multiply(quat_conjugate(curr), goal))


def pose_error(
    curr_pos: np.ndarray,
    curr_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
) -> tuple[float, float]:
    """Scalar position (m) and orientation (rad) errors used for success scoring.

    ``e_p = ||p_curr - p_goal||`` and ``e_R = ||Log(R_curr^T R_goal)^vee||``,
    i.e. the shortest rotation angle between the two frames (Equation 1).
    """
    e_p = float(np.linalg.norm(np.asarray(goal_pos) - np.asarray(curr_pos)))
    e_R = float(np.linalg.norm(quat_error_rotvec_body(curr_quat, goal_quat)))
    return e_p, e_R
