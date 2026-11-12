"""Damped-least-squares inverse kinematics for the Piper 6-DoF arm.

Converts an end-effector pose goal (as produced by ACT) into a 6-joint
configuration for the Piper, the classical ("normal IK") executor in the
ACT -> EE-pose -> IK -> Piper pipeline. The learned PolicyIK could drop in here
in place of DLS once trained for the Piper.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .piper_fk import PiperFK

N_ARM = 6


def rotvec_from_mat(R: np.ndarray) -> np.ndarray:
    """Axis-angle 3-vector of a rotation matrix (world-frame orientation error)."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-7:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


class PiperIK:
    def __init__(self, fk: PiperFK | None = None, damping: float = 0.05,
                 max_iters: int = 120, tol_pos: float = 1e-4, tol_rot: float = 1e-3,
                 step_gain: float = 1.0, max_step: float = 0.2) -> None:
        self.fk = fk or PiperFK()
        self.damping = damping
        self.max_iters = max_iters
        self.tol_pos = tol_pos
        self.tol_rot = tol_rot
        self.step_gain = step_gain
        self.max_step = max_step
        m = self.fk.model
        self.lo = m.jnt_range[:N_ARM, 0].copy()
        self.hi = m.jnt_range[:N_ARM, 1].copy()

    def _pose(self):
        m, d, s = self.fk.model, self.fk.data, self.fk.site
        return d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()

    def solve(self, goal_pos: np.ndarray, goal_R: np.ndarray, q_init: np.ndarray):
        """Return (q6, pos_residual_m, rot_residual_rad) for the goal TCP pose."""
        m, d, s = self.fk.model, self.fk.data, self.fk.site
        q = np.clip(np.asarray(q_init, float)[:N_ARM], self.lo, self.hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(self.max_iters):
            d.qpos[:N_ARM] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos, R = self._pose()
            e_p = np.asarray(goal_pos) - pos
            e_r = rotvec_from_mat(goal_R @ R.T)
            if np.linalg.norm(e_p) < self.tol_pos and np.linalg.norm(e_r) < self.tol_rot:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, s)
            J = np.vstack([jacp[:, :N_ARM], jacr[:, :N_ARM]])
            err = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(6), err)
            q = np.clip(q + np.clip(self.step_gain * dq, -self.max_step, self.max_step),
                        self.lo, self.hi)
        d.qpos[:N_ARM] = q
        mujoco.mj_kinematics(m, d)
        pos, R = self._pose()
        return q, float(np.linalg.norm(np.asarray(goal_pos) - pos)), \
            float(np.linalg.norm(rotvec_from_mat(goal_R @ R.T)))
