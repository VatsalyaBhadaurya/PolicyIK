"""Piper forward kinematics and 6D-rotation helpers (batched, CPU MuJoCo)."""

from __future__ import annotations

import os

import mujoco
import numpy as np

_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "piper", "scene.xml",
)
N_ARM = 6


class PiperFK:
    """Forward kinematics of the Piper TCP (`tcp` site on link6)."""

    def __init__(self, model_path: str = _MODEL, tcp_site: str = "tcp") -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if self.site < 0:
            raise ValueError(f"TCP site {tcp_site!r} not found")

    def fk(self, q6: np.ndarray):
        """TCP (pos (3,), R (3,3)) for one 6-joint configuration."""
        self.data.qpos[:N_ARM] = q6
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.site].copy()
        R = self.data.site_xmat[self.site].reshape(3, 3).copy()
        return pos, R

    def fk_batch(self, Q6: np.ndarray):
        """TCP (pos (N,3), rot6d (N,6)) for N configurations. rot6d = first two R columns."""
        n = Q6.shape[0]
        pos = np.empty((n, 3))
        r6 = np.empty((n, 6))
        for i in range(n):
            p, R = self.fk(Q6[i])
            pos[i] = p
            r6[i] = np.concatenate([R[:, 0], R[:, 1]])  # continuous 6D rotation
        return pos, r6


def rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix from a 6D rotation (Gram-Schmidt; Zhou et al. 2019)."""
    a1, a2 = r6[:3], r6[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / max(np.linalg.norm(a2), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns = basis vectors
