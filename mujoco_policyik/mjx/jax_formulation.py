"""PolicyIK formulation in JAX (functional, vmap/jit-friendly).

Mirrors ``mujoco_policyik.formulation`` and ``.kinematics`` but with jax.numpy and
no in-place state, so it runs batched on the GPU. All angles in radians,
quaternions in (w, x, y, z). Constants match the released PolicyIK config.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

TOL_POS = 1.0e-3
TOL_ORI = jnp.deg2rad(0.2)


class Cfg(NamedTuple):
    action_scale: float = 0.0625
    g_min: float = 0.1
    d_lo: float = 0.0005
    d_hi: float = 0.00931
    ori_lo: float = jnp.deg2rad(0.415)
    ori_hi: float = jnp.deg2rad(2.328)
    d0_pos_coarse: float = 0.01
    d0_ori_coarse: float = jnp.deg2rad(20.0)
    d0_pos_fine: float = 0.0005
    d0_ori_fine: float = jnp.deg2rad(0.25)
    kp_size_obs: float = 0.30
    kp_size_rew: float = 0.45
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
    h_scale: float = 20.0
    tol_pos: float = TOL_POS
    tol_ori: float = TOL_ORI


# --- quaternion / rotation math -------------------------------------------
def quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = w * w + x * x + y * y + z * z
    s = 2.0 / jnp.maximum(n, 1e-12)
    return jnp.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """Rotation matrix (3x3) -> (w, x, y, z) quaternion (Shepperd's method, branchless)."""
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    m21, m02, m10 = R[2, 1], R[0, 2], R[1, 0]
    m12, m20, m01 = R[1, 2], R[2, 0], R[0, 1]
    t = m00 + m11 + m22
    # four candidate quaternions, one per branch; pick the numerically best
    qw = jnp.array([t + 1.0, m21 - m12, m02 - m20, m10 - m01])
    qx = jnp.array([m21 - m12, 1.0 + m00 - m11 - m22, m01 + m10, m02 + m20])
    qy = jnp.array([m02 - m20, m01 + m10, 1.0 - m00 + m11 - m22, m12 + m21])
    qz = jnp.array([m10 - m01, m02 + m20, m12 + m21, 1.0 - m00 - m11 + m22])
    diors = jnp.array([t, m00, m11, m22])
    cand = jnp.stack([qw, qx, qy, qz], axis=0)
    q = cand[jnp.argmax(diors)]
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-12)


def rotvec_from_quat(q):
    q = q / jnp.maximum(jnp.linalg.norm(q), 1e-12)
    q = jnp.where(q[0] < 0.0, -q, q)
    v = q[1:]
    nv = jnp.linalg.norm(v)
    nv_safe = jnp.where(nv > 1e-8, nv, 1.0)
    angle = 2.0 * jnp.arctan2(nv, q[0])
    scale = jnp.where(nv > 1e-8, angle / nv_safe, 2.0)
    return v * scale


def rotvec_err_body(curr, goal):
    return rotvec_from_quat(quat_mul(quat_conj(curr), goal))


def pose_error(pos, quat, gpos, gquat):
    e_p = jnp.linalg.norm(gpos - pos)
    e_R = jnp.linalg.norm(rotvec_err_body(quat, gquat))
    return e_p, e_R


# --- gain, channels, keypoints --------------------------------------------
def _ramp(e, lo, hi, g_min):
    t = jnp.clip((e - lo) / jnp.maximum(hi - lo, 1e-9), 0.0, 1.0)
    return g_min + (1.0 - g_min) * t


def action_gain(e_p, e_R, c: Cfg):
    return jnp.maximum(_ramp(e_p, c.d_lo, c.d_hi, c.g_min),
                       _ramp(e_R, c.ori_lo, c.ori_hi, c.g_min))


_EYE3 = jnp.eye(3)


def keypoint_offsets_world(pos, quat, gpos, gquat, kp_size):
    Rc, Rd = quat_to_mat(quat), quat_to_mat(gquat)
    offs = _EYE3 * kp_size
    kc = pos + offs @ Rc.T
    kd = gpos + offs @ Rd.T
    return (kd - kc).reshape(-1)


def keypoint_mean_distance(pos, quat, gpos, gquat, kp_size):
    Rc, Rd = quat_to_mat(quat), quat_to_mat(gquat)
    offs = (_EYE3 - _EYE3.mean(axis=0, keepdims=True)) * kp_size
    kc = pos + offs @ Rc.T
    kd = gpos + offs @ Rd.T
    return jnp.mean(jnp.linalg.norm(kc - kd, axis=-1))


def pose_channel(pos, quat, gpos, gquat, d0_pos, d0_ori):
    Rc = quat_to_mat(quat)
    p_err = Rc.T @ (gpos - pos)
    r_err = rotvec_err_body(quat, gquat)
    return jnp.concatenate([jnp.clip(p_err / d0_pos, -1.0, 1.0),
                            jnp.clip(r_err / d0_ori, -1.0, 1.0)])


def rho_of(e_p, e_R, c: Cfg):
    return jnp.maximum(e_p / c.tol_pos, e_R / c.tol_ori)


def error_history(rho, best_before, c: Cfg):
    return jnp.array([jnp.clip(best_before / c.h_scale, 0.0, 1.0),
                      jnp.clip((rho - best_before) / c.h_scale, 0.0, 1.0)])


def _U(rho, beta):
    return 0.5 * (1.0 + jnp.tanh((1.0 - rho) / beta))


def reward(e_p, e_R, rho, best_before, prev_phi, kp_eps, q, jr_lo, jr_hi, qd,
           action_rate_sq, c: Cfg):
    """Full PolicyIK reward; returns (total, new_best, new_phi)."""
    K = lambda x, s: 1.0 - jnp.tanh(x / s)
    r_track = -0.2 * e_p + 0.1 * K(e_p, 0.1) + K(kp_eps, 0.2) + K(kp_eps, 0.05)

    r_imp = jnp.maximum(_U(rho, c.improvement_beta) - _U(best_before, c.improvement_beta), 0.0)
    phi = -jnp.minimum(c.shape_cap, rho)
    r_shape = c.shape_gamma * phi - prev_phi
    c_back = jnp.minimum(c.backslide_cap, jnp.maximum(rho - best_before, 0.0))

    r_dwell = jnp.where(rho <= 1.0, 1.0, 0.0)

    c_motion = c.w_action_rate * action_rate_sq + c.w_joint_vel * jnp.sum(qd ** 2)
    rng = jnp.maximum(jr_hi - jr_lo, 1e-6)
    margin = jnp.min(jnp.minimum(q - jr_lo, jr_hi - q) / rng)
    c_limit = jnp.clip(1.0 - margin / c.joint_margin_eps, 0.0, 1.0)
    c_limit = jnp.where(rho > c.rho_gate_limit, c_limit, 0.0)

    total = (c.w_track * r_track + c.w_improvement * r_imp + c.w_shape * r_shape
             - c.w_backslide * c_back + c.w_dwell * r_dwell
             - c_motion - c.w_joint_margin * c_limit)
    return total, jnp.minimum(best_before, rho), phi
