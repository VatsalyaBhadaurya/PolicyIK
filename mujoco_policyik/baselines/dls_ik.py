"""Damped-least-squares (DLS) differential inverse kinematics.

This is the classical "normal IK" baseline PolicyIK compares against (the DLS
row of Table 1 in the paper). Each control step it turns the world-frame pose
error into a joint-reference increment,

    dq = J^T (J J^T + lambda^2 I)^-1 * [e_p; e_R],
    z  = clip(z + clip_step(alpha * dq)),

and the arm's joint PD controller tracks ``z``. Damping ``lambda`` trades
accuracy for robustness near singularities (Wampler 1986; Deo & Walker 1995),
exactly the per-robot tuning knob PolicyIK removes.
"""

from __future__ import annotations

import numpy as np

from .. import kinematics as K
from ..arm import ArmModel


def world_twist_error(
    curr_pos: np.ndarray,
    curr_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
) -> np.ndarray:
    """6-vector [position error; world-frame orientation error] for the Jacobian.

    The angular part is ``Log(R_goal R_curr^T)^vee`` expressed in the world
    frame, consistent with MuJoCo's world-frame site Jacobian.
    """
    e_p = np.asarray(goal_pos, dtype=float) - np.asarray(curr_pos, dtype=float)
    # body-frame error rotated into the world frame: R_curr @ Log(R_curr^T R_goal)
    e_body = K.quat_error_rotvec_body(curr_quat, goal_quat)
    e_r = K.quat_to_mat(curr_quat) @ e_body
    return np.concatenate([e_p, e_r])


class DLSController:
    """Closed-loop DLS IK that emits a joint reference for the arm's PD controller.

    Parameters
    ----------
    damping:
        Levenberg-Marquardt damping ``lambda``. Larger is more robust, less precise.
    step_gain:
        Scalar ``alpha`` applied to ``dq`` each step.
    max_step:
        Per-joint cap on the reference increment (rad), for stability.
    """

    def __init__(
        self,
        arm: ArmModel,
        damping: float = 0.05,
        step_gain: float = 1.0,
        max_step: float = 0.1,
    ) -> None:
        self.arm = arm
        self.damping = float(damping)
        self.step_gain = float(step_gain)
        self.max_step = float(max_step)
        self.z = arm.z.copy()

    def reset(self) -> None:
        """Anchor the reference at the arm's current configuration."""
        self.z = self.arm.q.copy()

    def compute(self, goal_pos: np.ndarray, goal_quat: np.ndarray) -> np.ndarray:
        """One DLS update: returns the new joint reference ``z``."""
        pos, quat = self.arm.fk()
        err = world_twist_error(pos, quat, goal_pos, goal_quat)
        J = self.arm.jacobian()
        JJt = J @ J.T
        lam2 = self.damping ** 2
        dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), err)
        dq = np.clip(self.step_gain * dq, -self.max_step, self.max_step)
        self.z = self.arm.clip_to_limits(self.z + dq)
        return self.z

    def run(
        self,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        steps: int = 160,
    ) -> dict:
        """Closed-loop reach: step the arm for ``steps`` control periods.

        Returns a dict with per-step position/orientation error traces.
        """
        self.reset()
        e_pos = np.empty(steps)
        e_ori = np.empty(steps)
        for t in range(steps):
            z = self.compute(goal_pos, goal_quat)
            self.arm.step(z)
            e_pos[t], e_ori[t] = self.arm.pose_error(goal_pos, goal_quat)
        return {"e_pos": e_pos, "e_ori": e_ori}
