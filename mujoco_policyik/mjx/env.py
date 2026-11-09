"""Batched PolicyIK environment on MJX, exposed as a Brax ``Env`` for PPO.

The UR5e plant is stepped with ``mjx`` on the GPU; the PolicyIK action
(accumulated joint reference with error-dependent gain, tanh-squashed) and the
50-D observation / reward come from ``jax_formulation``. Goals are poses of
random reachable configs (FK of a sampled ``q*``); each episode starts a random
distance from its goal (a mixed difficulty curriculum), so every batch carries
dense near-goal signal while still covering the workspace.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

from . import jax_formulation as F

HOME_Q = jnp.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "ur5e", "scene.xml",
)


class PolicyIKMJX(Env):
    def __init__(self, model_path: str = _MODEL_PATH, control_dt: float = 1.0 / 30.0,
                 ee_site: str = "attachment_site", start_std_range=(0.01, 0.5)):
        mj = mujoco.MjModel.from_xml_path(model_path)
        self._m = mjx.put_model(mj)
        self._site = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        self._nu = mj.nu
        self._n_sub = max(1, int(round(control_dt / mj.opt.timestep)))
        self._jr_lo = jnp.array(mj.jnt_range[:, 0])
        self._jr_hi = jnp.array(mj.jnt_range[:, 1])
        self._q_nom = HOME_Q
        self._std_lo, self._std_hi = start_std_range
        self._cfg = F.Cfg()

    @property
    def observation_size(self) -> int:
        return 50

    @property
    def action_size(self) -> int:
        return 6

    @property
    def backend(self) -> str:
        return "mjx"

    # -- helpers ----------------------------------------------------------
    def _fk(self, data):
        pos = data.site_xpos[self._site]
        quat = F.mat_to_quat(data.site_xmat[self._site].reshape(3, 3))
        return pos, quat

    def _obs(self, data, gpos, gquat, z, best_before, rho, prev_a):
        q = data.qpos[: self._nu]
        qd = data.qvel[: self._nu]
        pos, quat = self._fk(data)
        c = self._cfg
        return jnp.concatenate([
            q - self._q_nom,
            qd,
            gpos,
            F.keypoint_offsets_world(pos, quat, gpos, gquat, c.kp_size_obs),
            F.pose_channel(pos, quat, gpos, gquat, c.d0_pos_coarse, c.d0_ori_coarse),
            F.pose_channel(pos, quat, gpos, gquat, c.d0_pos_fine, c.d0_ori_fine),
            F.error_history(rho, best_before, c),
            z - q,                       # command-state feedback d_t
            prev_a,
        ])

    # -- Brax API ---------------------------------------------------------
    def reset(self, rng) -> State:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        margin = 0.05 * (self._jr_hi - self._jr_lo)
        q_star = jax.random.uniform(k1, (self._nu,), minval=self._jr_lo + margin,
                                    maxval=self._jr_hi - margin)
        # goal = FK(q_star)
        dg = mjx.kinematics(self._m, mjx.make_data(self._m).replace(qpos=q_star))
        gpos, gquat = self._fk(dg)
        # start a random distance from the goal config, biased toward near-goal
        # (u**3 concentrates mass near std_lo) so most envs give dense dwell/
        # improvement signal while a tail still covers the workspace.
        u = jax.random.uniform(k2, ())
        std = self._std_lo + (self._std_hi - self._std_lo) * u ** 3
        q0 = jnp.clip(q_star + std * jax.random.normal(k3, (self._nu,)),
                      self._jr_lo, self._jr_hi)
        data = mjx.forward(self._m, mjx.make_data(self._m).replace(qpos=q0, ctrl=q0))

        pos, quat = self._fk(data)
        e_p, e_R = F.pose_error(pos, quat, gpos, gquat)
        rho = F.rho_of(e_p, e_R, self._cfg)
        prev_phi = -jnp.minimum(self._cfg.shape_cap, rho)
        prev_a = jnp.zeros(self._nu)
        obs = self._obs(data, gpos, gquat, q0, rho, rho, prev_a)
        info = {"z": q0, "best": rho, "prev_phi": prev_phi, "gpos": gpos,
                "gquat": gquat, "prev_a": prev_a, "rng": rng}
        metrics = {"e_pos": e_p, "e_ori": e_R, "within_tol": (rho <= 1.0).astype(jnp.float32)}
        return State(data, obs, jnp.float32(0.0), jnp.float32(0.0), metrics, info)

    def reset_to(self, q_start, goal_pos, goal_quat) -> State:
        """Deterministic reset to a fixed task (for evaluation on a held-out set)."""
        q0 = jnp.asarray(q_start)
        gpos, gquat = jnp.asarray(goal_pos), jnp.asarray(goal_quat)
        data = mjx.forward(self._m, mjx.make_data(self._m).replace(qpos=q0, ctrl=q0))
        pos, quat = self._fk(data)
        e_p, e_R = F.pose_error(pos, quat, gpos, gquat)
        rho = F.rho_of(e_p, e_R, self._cfg)
        prev_phi = -jnp.minimum(self._cfg.shape_cap, rho)
        prev_a = jnp.zeros(self._nu)
        obs = self._obs(data, gpos, gquat, q0, rho, rho, prev_a)
        info = {"z": q0, "best": rho, "prev_phi": prev_phi, "gpos": gpos,
                "gquat": gquat, "prev_a": prev_a, "rng": jax.random.PRNGKey(0)}
        metrics = {"e_pos": e_p, "e_ori": e_R, "within_tol": (rho <= 1.0).astype(jnp.float32)}
        return State(data, obs, jnp.float32(0.0), jnp.float32(0.0), metrics, info)

    def step(self, state: State, action) -> State:
        c = self._cfg
        data, info = state.pipeline_state, state.info
        gpos, gquat = info["gpos"], info["gquat"]

        # error-dependent gain from the pre-step pose, then accumulate reference
        pos0, quat0 = self._fk(data)
        e_p0, e_R0 = F.pose_error(pos0, quat0, gpos, gquat)
        beta = F.action_gain(e_p0, e_R0, c)
        # Brax's NormalTanhDistribution already squashes actions to (-1, 1) with
        # the correct log-prob correction, so use them directly (clip is a no-op guard).
        a_eff = jnp.clip(action, -1.0, 1.0)
        z = jnp.clip(info["z"] + beta * c.action_scale * a_eff, self._jr_lo, self._jr_hi)

        # advance the PD plant one control period
        data = data.replace(ctrl=z)
        data = jax.lax.fori_loop(0, self._n_sub, lambda _, d: mjx.step(self._m, d), data)

        pos, quat = self._fk(data)
        e_p, e_R = F.pose_error(pos, quat, gpos, gquat)
        rho = F.rho_of(e_p, e_R, c)
        best_before = info["best"]
        kp_eps = F.keypoint_mean_distance(pos, quat, gpos, gquat, c.kp_size_rew)
        q = data.qpos[: self._nu]
        qd = data.qvel[: self._nu]
        arate = jnp.sum((a_eff - info["prev_a"]) ** 2)
        rew, new_best, new_phi = F.reward(e_p, e_R, rho, best_before, info["prev_phi"],
                                          kp_eps, q, self._jr_lo, self._jr_hi, qd, arate, c)

        obs = self._obs(data, gpos, gquat, z, best_before, rho, a_eff)
        info = {**info, "z": z, "best": new_best, "prev_phi": new_phi, "prev_a": a_eff}
        # preserve any metrics keys the brax wrappers add (e.g. 'reward')
        metrics = {**state.metrics, "e_pos": e_p, "e_ori": e_R,
                   "within_tol": (rho <= 1.0).astype(jnp.float32)}
        return state.replace(pipeline_state=data, obs=obs, reward=rew,
                             done=jnp.float32(0.0), metrics=metrics, info=info)
