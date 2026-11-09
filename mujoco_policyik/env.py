"""PolicyIK goal-conditioned environment on a 6-DoF arm (UR5e) in MuJoCo.

One control step:

1. compute ``beta`` from the *pre-step* pose error (Eq. 6);
2. accumulate the reference ``z_t = clip(z_{t-1} + beta * eta * a_t)`` (Eq. 2);
3. advance the joint PD plant one control period;
4. score the *post-step* pose: reward (Eq. 7) and the 50-D observation (Eq. 3).

The observation layout (50-D) is::

    [ q-q_nom(6), qdot(6), goal_pos(6?no->3), keypoint_offsets(9),
      pose_coarse(6), pose_fine(6), error_history(2),
      command_state d_t(6), prev_action(6) ]
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .arm import ArmModel, HOME_Q
from .formulation import (
    PolicyIKConfig,
    RewardState,
    action_gain,
    error_history,
    keypoint_mean_distance,
    keypoint_offsets_world,
    pose_channel,
    reward_terms,
    rho_of,
)

OBS_DIM = 50
ACT_DIM = 6


class PolicyIKEnv(gym.Env):
    """Goal-conditioned high-precision IK env, PolicyIK formulation, 6-DoF."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: PolicyIKConfig | None = None,
        horizon: int = 160,
        q_nom: np.ndarray | None = None,
        start_near_goal_std: float | None = None,
        seed: int | None = None,
        **arm_kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PolicyIKConfig()
        self.arm = ArmModel(**arm_kwargs)
        self.horizon = int(horizon)
        self.q_nom = HOME_Q.copy() if q_nom is None else np.asarray(q_nom, dtype=float)
        # reverse-curriculum knob: if set, starts are drawn near the goal config
        self.start_near_goal_std = start_near_goal_std
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)

        self._goal_pos = np.zeros(3)
        self._goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._prev_action = np.zeros(ACT_DIM)
        self._rstate = RewardState()
        self._t = 0

    # -- helpers ----------------------------------------------------------
    def _sample_start(self, q_star: np.ndarray) -> np.ndarray:
        lo, hi = self.arm.joint_range[:, 0], self.arm.joint_range[:, 1]
        if self.start_near_goal_std is not None:
            q0 = q_star + self._rng.normal(scale=self.start_near_goal_std, size=ACT_DIM)
        else:
            margin = 0.05 * (hi - lo)
            q0 = self._rng.uniform(lo + margin, hi - margin)
        return self.arm.clip_to_limits(q0)

    def _observation(self, best_before: float, rho: float, d_cmd: np.ndarray) -> np.ndarray:
        pos, quat = self.arm.fk()
        kp = keypoint_offsets_world(pos, quat, self._goal_pos, self._goal_quat, self.cfg.kp_size_obs)
        coarse = pose_channel(pos, quat, self._goal_pos, self._goal_quat,
                              self.cfg.d0_pos_coarse, self.cfg.d0_ori_coarse)
        fine = pose_channel(pos, quat, self._goal_pos, self._goal_quat,
                            self.cfg.d0_pos_fine, self.cfg.d0_ori_fine)
        obs = np.concatenate([
            self.arm.q - self.q_nom,        # 6
            self.arm.qd,                    # 6
            self._goal_pos,                 # 3
            kp,                             # 9
            coarse,                         # 6
            fine,                           # 6
            error_history(rho, best_before, self.cfg),  # 2
            d_cmd,                          # 6
            self._prev_action,              # 6
        ])
        return obs.astype(np.float32)

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "goal_pos" in options and "goal_quat" in options:
            self._goal_pos = np.asarray(options["goal_pos"], dtype=float)
            self._goal_quat = np.asarray(options["goal_quat"], dtype=float)
            q_star = options.get("q_star", self.q_nom)
        else:
            q_star, self._goal_pos, self._goal_quat = self.arm.sample_reachable_goal(self._rng)
        q0 = np.asarray(options["q_start"], dtype=float) if "q_start" in options else self._sample_start(q_star)

        self.arm.reset(q0)                       # z anchored at q0
        self._prev_action = np.zeros(ACT_DIM)
        self._rstate = RewardState()
        self._t = 0

        e_p, e_R = self.arm.pose_error(self._goal_pos, self._goal_quat)
        rho = rho_of(e_p, e_R, self.cfg)
        self._rstate.best = rho                  # b_0 = rho_0
        d_cmd = self.arm.z - self.arm.q          # ~0 at reset
        obs = self._observation(best_before=rho, rho=rho, d_cmd=d_cmd)
        info = {"e_pos": e_p, "e_ori": e_R, "rho": rho}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).reshape(ACT_DIM)

        # 1-2. error-dependent gain from the pre-step pose, then accumulate reference
        e_p0, e_R0 = self.arm.pose_error(self._goal_pos, self._goal_quat)
        beta = action_gain(e_p0, e_R0, self.cfg)
        z = self.arm.clip_to_limits(self.arm.z + beta * self.cfg.action_scale * action)

        # 3. advance the PD plant one control period
        self.arm.step(z)

        # 4. score the post-step pose
        e_p, e_R = self.arm.pose_error(self._goal_pos, self._goal_quat)
        rho = rho_of(e_p, e_R, self.cfg)
        best_before = self._rstate.best
        kp_eps = keypoint_mean_distance(*self.arm.fk(), self._goal_pos, self._goal_quat, self.cfg.kp_size_rew)
        action_rate_sq = float(np.sum((action - self._prev_action) ** 2))
        reward, terms = reward_terms(
            e_p, e_R, rho, self._rstate, kp_eps,
            self.arm.q, self.arm.joint_range, self.arm.qd, action_rate_sq, self.cfg,
        )

        d_cmd = self.arm.z - self.arm.q
        obs = self._observation(best_before=best_before, rho=rho, d_cmd=d_cmd)
        self._prev_action = action

        self._t += 1
        truncated = self._t >= self.horizon
        info = {"e_pos": e_p, "e_ori": e_R, "rho": rho, "beta": beta,
                "within_tol": rho <= 1.0, "reward_terms": terms}
        return obs, float(reward), False, truncated, info
