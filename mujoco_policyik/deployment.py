"""Deployment protocol: controller-agnostic retries that raise holding success.

The paper lifts single-attempt success (93.5%) to 98.9% with a fixed retry
protocol, *without changing the controller*. This implements the same idea so it
can wrap DLS, QP or the learned policy:

* **corrected-goal retry** -- a failure that ends within ``correct_radius`` of
  the goal is retried against a goal that cancels the measured residual, so the
  controller's repeatable steady-state bias lands the end effector on the goal;
* **transient-goal retry** -- a failure that ends far away first commands a
  transient (offset + rotated) goal to break out of the stalled configuration
  (e.g. a joint pinned at its limit), then re-commands the original goal.

Every attempt is scored against the *original* goal.
"""

from __future__ import annotations

import numpy as np

from . import kinematics as K
from . import TOL_ORI_RAD, TOL_POS_M
from .arm import ArmModel
from .metrics import FINAL_WINDOW


def _corrected_goal(orig_p, orig_q, real_p, real_q):
    """Goal that cancels the residual: realized(bias) then lands on the original."""
    cmd_p = 2.0 * orig_p - real_p
    q_res = K.quat_multiply(real_q, K.quat_conjugate(orig_q))      # realized = q_res * original
    cmd_q = K.quat_multiply(K.quat_conjugate(q_res), orig_q)        # command q_res^-1 * original
    return cmd_p, cmd_q


def run_deployment(
    arm: ArmModel,
    make_ctrl,
    q_start: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    steps: int = 160,
    max_attempts: int = 4,
    correct_radius: float = 0.01,
    transient_offset: float = 0.30,
    transient_rot_deg: float = 90.0,
    transient_steps: int = 40,
    rng: np.random.Generator | None = None,
) -> tuple[dict, int]:
    """Run a controller under the deployment protocol. Returns (best_trace, n_attempts).

    ``make_ctrl(arm)`` builds a fresh controller anchored at the arm's current
    configuration (so retries continue from where the last attempt left off).
    ``best_trace`` holds per-step errors vs the original goal for the first
    strictly successful attempt, or the closest attempt if none succeed.
    """
    rng = rng or np.random.default_rng(0)
    orig_p = np.asarray(goal_pos, dtype=float)
    orig_q = np.asarray(goal_quat, dtype=float)
    arm.reset(q_start)

    best_trace, best_key = None, np.inf
    mode = "initial"          # how the *next* attempt retries
    transient_p = transient_q = None
    n_attempts = 0

    for _ in range(max_attempts):
        n_attempts += 1
        cmd_p, cmd_q = orig_p, orig_q
        if mode == "corrected":
            real_p, real_q = arm.fk()
            cmd_p, cmd_q = _corrected_goal(orig_p, orig_q, real_p, real_q)

        ctrl = make_ctrl(arm)   # anchors its reference at the current config
        e_pos = np.empty(steps)
        e_ori = np.empty(steps)
        for t in range(steps):
            if mode == "transient" and t < transient_steps:
                g_p, g_q = transient_p, transient_q
            else:
                g_p, g_q = cmd_p, cmd_q
            arm.step(ctrl.compute(g_p, g_q))
            e_pos[t], e_ori[t] = arm.pose_error(orig_p, orig_q)

        mp = float(np.mean(e_pos[-FINAL_WINDOW:]))
        mo = float(np.mean(e_ori[-FINAL_WINDOW:]))
        key = max(mp / TOL_POS_M, mo / TOL_ORI_RAD)
        if key < best_key:
            best_key, best_trace = key, {"e_pos": e_pos, "e_ori": e_ori}
        if mp <= TOL_POS_M and mo <= TOL_ORI_RAD:
            break   # strict success

        # choose the retry mode for the next attempt from the realized pose
        real_p, real_q = arm.fk()
        if float(np.linalg.norm(real_p - orig_p)) <= correct_radius:
            mode = "corrected"
        else:
            mode = "transient"
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            transient_p = orig_p + transient_offset * d
            ax = rng.normal(size=3)
            transient_q = K.quat_multiply(
                K.quat_from_axis_angle(ax, np.deg2rad(transient_rot_deg)), orig_q)

    return best_trace, n_attempts
