"""The PolicyIK formulation (Sections 4.1-4.3), ported to 6-DoF in NumPy.

Three design pillars, all operating in the world frame (with a fixed base this
equals the paper's base frame up to a constant transform):

4.1  Reference accumulation + command-state feedback
     ``z_t = clip(z_{t-1} + eta * beta(e_p,e_R) * a_t)`` and the observation
     exposes ``d_t = z_{t-1} - q_t`` so the policy can retain the rest offset.
4.2  Multiscale pose feedback + error-dependent action gain ``beta``.
4.3  Tolerance-normalised progress and dwell rewards.

Constants default to the values in the released ``env_cfg.py`` / ``actions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import TOL_ORI_RAD, TOL_POS_M
from . import kinematics as K


@dataclass
class PolicyIKConfig:
    # -- 4.1 action: accumulated joint reference -------------------------
    action_scale: float = 0.0625        # eta
    # -- 4.2 error-dependent gain beta -----------------------------------
    g_min: float = 0.1
    d_lo: float = 0.0005                 # position ramp lower (m)
    d_hi: float = 0.00931                # position ramp upper (m)
    ori_lo: float = np.deg2rad(0.415)    # orientation ramp lower (rad)
    ori_hi: float = np.deg2rad(2.328)    # orientation ramp upper (rad)
    # -- 4.2 multiscale pose channels ------------------------------------
    d0_pos_coarse: float = 0.01          # 1 cm
    d0_ori_coarse: float = np.deg2rad(20.0)
    d0_pos_fine: float = 0.0005          # 0.5 mm
    d0_ori_fine: float = np.deg2rad(0.25)
    kp_size_obs: float = 0.30            # keypoint arm length for the observation
    kp_size_rew: float = 0.45            # keypoint arm length for the tracking reward
    # -- 4.3 reward weights ----------------------------------------------
    w_improvement: float = 26.831
    w_shape: float = 1.0
    w_backslide: float = 0.2
    w_dwell: float = 1.0
    w_track: float = 1.0
    w_action_rate: float = 5.0e-4
    w_joint_vel: float = 5.0e-4
    w_joint_margin: float = 1.0
    improvement_beta: float = 0.5
    shape_gamma: float = 0.99
    shape_cap: float = 50.0
    backslide_cap: float = 3.0
    joint_margin_eps: float = 0.05
    rho_gate_limit: float = 10.0
    h_scale: float = 20.0                # error-history normaliser d0
    # -- tolerances ------------------------------------------------------
    tol_pos: float = TOL_POS_M
    tol_ori: float = TOL_ORI_RAD


# ---------------------------------------------------------------------------
# 4.2  Error-dependent action gain beta(e_p, e_R)
# ---------------------------------------------------------------------------
def _ramp(e: float, lo: float, hi: float, g_min: float) -> float:
    t = np.clip((e - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return g_min + (1.0 - g_min) * t


def action_gain(e_p: float, e_R: float, cfg: PolicyIKConfig) -> float:
    """``beta`` in [g_min, 1]: full step far from the goal, shrunk near it (Eq. 6)."""
    return max(
        _ramp(e_p, cfg.d_lo, cfg.d_hi, cfg.g_min),
        _ramp(e_R, cfg.ori_lo, cfg.ori_hi, cfg.g_min),
    )


# ---------------------------------------------------------------------------
# 4.2  Keypoints and multiscale pose channels
# ---------------------------------------------------------------------------
def _keypoint_offsets(kp_size: float, zero_sum: bool) -> np.ndarray:
    offs = np.eye(3) * kp_size
    return offs - offs.mean(axis=0, keepdims=True) if zero_sum else offs


def keypoint_offsets_world(curr_pos, curr_quat, goal_pos, goal_quat, kp_size: float) -> np.ndarray:
    """Goal-minus-current offsets of three EE keypoints, world frame (9-vector)."""
    Rc, Rd = K.quat_to_mat(curr_quat), K.quat_to_mat(goal_quat)
    offs = _keypoint_offsets(kp_size, zero_sum=False)
    out = []
    for i in range(3):
        kp_cur = curr_pos + Rc @ offs[i]
        kp_des = goal_pos + Rd @ offs[i]
        out.append(kp_des - kp_cur)
    return np.concatenate(out)


def keypoint_mean_distance(curr_pos, curr_quat, goal_pos, goal_quat, kp_size: float) -> float:
    """Mean distance of three zero-sum EE keypoints to their goal (``eps`` in Eq. 8)."""
    Rc, Rd = K.quat_to_mat(curr_quat), K.quat_to_mat(goal_quat)
    offs = _keypoint_offsets(kp_size, zero_sum=True)
    d = [np.linalg.norm((curr_pos + Rc @ offs[i]) - (goal_pos + Rd @ offs[i])) for i in range(3)]
    return float(np.mean(d))


def pose_channel(curr_pos, curr_quat, goal_pos, goal_quat, d0_pos, d0_ori) -> np.ndarray:
    """6-vector ``[clip(R_ee^T (p_g-p)/d0_pos), clip(Log(R_ee^T R_g)/d0_ori)]`` in [-1,1]."""
    Rc = K.quat_to_mat(curr_quat)
    p_err = Rc.T @ (np.asarray(goal_pos) - np.asarray(curr_pos))
    r_err = K.quat_error_rotvec_body(curr_quat, goal_quat)
    return np.concatenate(
        [np.clip(p_err / d0_pos, -1.0, 1.0), np.clip(r_err / d0_ori, -1.0, 1.0)]
    )


# ---------------------------------------------------------------------------
# 4.3  Tolerance-normalised error state and reward terms
# ---------------------------------------------------------------------------
def rho_of(e_p: float, e_R: float, cfg: PolicyIKConfig) -> float:
    """``rho = max(e_p / tol_pos, e_R / tol_ori)``; ``rho <= 1`` iff within tolerance."""
    return max(e_p / cfg.tol_pos, e_R / cfg.tol_ori)


def _U(rho: float, beta: float) -> float:
    return 0.5 * (1.0 + np.tanh((1.0 - rho) / beta))


@dataclass
class RewardState:
    """Per-episode running state for the progress terms."""
    best: float = np.inf          # b_t, running min of rho
    prev_phi: float | None = None  # Phi_{t-1} for potential shaping


def reward_terms(
    e_p: float,
    e_R: float,
    rho: float,
    rstate: RewardState,
    kp_eps: float,
    q: np.ndarray,
    joint_range: np.ndarray,
    qd: np.ndarray,
    action_rate_sq: float,
    cfg: PolicyIKConfig,
) -> tuple[float, dict]:
    """Full PolicyIK reward (Eq. 7) and a per-term breakdown. Advances ``rstate``.

    ``rho`` is this step's tolerance-normalised error; ``rstate.best`` holds
    ``b_{t-1}`` (updated in place to ``b_t``).
    """
    best_before = rstate.best

    # -- tracking (Eq. 8) --
    K_ = lambda x, s: 1.0 - np.tanh(x / s)
    r_track = -0.2 * e_p + 0.1 * K_(e_p, 0.1) + K_(kp_eps, 0.2) + K_(kp_eps, 0.05)

    # -- progress (Eq. 10) --
    r_imp = max(_U(rho, cfg.improvement_beta) - _U(best_before, cfg.improvement_beta), 0.0)
    phi = -min(cfg.shape_cap, rho)
    r_shape = 0.0 if rstate.prev_phi is None else cfg.shape_gamma * phi - rstate.prev_phi
    c_back = min(cfg.backslide_cap, max(rho - best_before, 0.0))

    # -- dwell --
    r_dwell = 1.0 if rho <= 1.0 else 0.0

    # -- costs --
    c_motion = cfg.w_action_rate * action_rate_sq + cfg.w_joint_vel * float(np.sum(qd ** 2))
    lo, hi = joint_range[:, 0], joint_range[:, 1]
    margin = np.min(np.minimum(q - lo, hi - q) / np.maximum(hi - lo, 1e-6))
    c_limit = np.clip(1.0 - margin / cfg.joint_margin_eps, 0.0, 1.0)
    c_limit = c_limit if rho > cfg.rho_gate_limit else 0.0

    terms = {
        "track": cfg.w_track * r_track,
        "improvement": cfg.w_improvement * r_imp,
        "shape": cfg.w_shape * r_shape,
        "backslide": -cfg.w_backslide * c_back,
        "dwell": cfg.w_dwell * r_dwell,
        "motion_cost": -c_motion,
        "limit_cost": -cfg.w_joint_margin * c_limit,
    }
    total = float(sum(terms.values()))

    rstate.best = min(best_before, rho)
    rstate.prev_phi = phi
    return total, terms


def error_history(rho: float, best_before: float, cfg: PolicyIKConfig) -> np.ndarray:
    """``h_t = [clip(b_{t-1}/d0,0,1), clip((rho_t-b_{t-1})/d0,0,1)]``."""
    return np.array(
        [
            np.clip(best_before / cfg.h_scale, 0.0, 1.0),
            np.clip((rho - best_before) / cfg.h_scale, 0.0, 1.0),
        ]
    )
