"""Resolved-rate quadratic-programming (QP) inverse kinematics.

The second classical baseline in the paper (the QP row of Table 1). Each control
step it solves a box-constrained least-squares problem for the joint increment

    min_dq  || J dq - e ||^2 + lambda^2 ||dq||^2
    s.t.    dq_min <= dq <= dq_max   (joint velocity limits)
            q_lo <= q + dq <= q_hi   (joint position limits)

so, unlike plain DLS, it respects joint limits explicitly -- the case where the
paper reports classical controllers stalling with a joint pinned at its limit.
Solved with ``scipy.optimize.lsq_linear`` (a bounded linear least squares).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .. import kinematics as K
from ..arm import ArmModel
from .dls_ik import world_twist_error


class QPController:
    """Closed-loop resolved-rate QP IK emitting a joint reference for the PD plant.

    Parameters
    ----------
    damping:
        Tikhonov weight ``lambda`` on ``dq``.
    dq_max:
        Per-step joint velocity bound (rad), i.e. ``v_max * dt``.
    """

    def __init__(
        self,
        arm: ArmModel,
        damping: float = 0.02,
        dq_max: float = 0.1,
    ) -> None:
        self.arm = arm
        self.damping = float(damping)
        self.dq_max = float(dq_max)
        self.z = arm.z.copy()

    def reset(self) -> None:
        self.z = self.arm.q.copy()

    def compute(self, goal_pos: np.ndarray, goal_quat: np.ndarray) -> np.ndarray:
        pos, quat = self.arm.fk()
        e = world_twist_error(pos, quat, goal_pos, goal_quat)
        J = self.arm.jacobian()
        n = J.shape[1]

        # Resolved-rate velocity scaling: uniformly scale the task twist so the
        # natural (unconstrained) joint step respects the velocity limit while
        # preserving the Cartesian direction. The box constraints then bind only
        # for genuine joint-limit avoidance, not to cap transport speed.
        dq_unc = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(n), e)
        peak = float(np.max(np.abs(dq_unc)))
        if peak > self.dq_max:
            e = e * (self.dq_max / peak)

        A = np.vstack([J, self.damping * np.eye(n)])
        b = np.concatenate([e, np.zeros(n)])
        # bounds: remaining room to the joint limits, capped at a velocity bound
        q = self.arm.q
        vcap = 1.5 * self.dq_max
        lo = np.clip(self.arm.joint_range[:, 0] - q, -vcap, 0.0)
        hi = np.clip(self.arm.joint_range[:, 1] - q, 0.0, vcap)
        sol = lsq_linear(A, b, bounds=(lo, hi), method="bvls", max_iter=50)
        self.z = self.arm.clip_to_limits(self.z + sol.x)
        return self.z

    def run(self, goal_pos: np.ndarray, goal_quat: np.ndarray, steps: int = 160) -> dict:
        self.reset()
        e_pos = np.empty(steps)
        e_ori = np.empty(steps)
        for t in range(steps):
            z = self.compute(goal_pos, goal_quat)
            self.arm.step(z)
            e_pos[t], e_ori[t] = self.arm.pose_error(goal_pos, goal_quat)
        return {"e_pos": e_pos, "e_ori": e_ori}
