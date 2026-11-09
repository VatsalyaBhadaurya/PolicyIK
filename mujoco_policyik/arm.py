"""A MuJoCo 6-DoF arm (UR5e) wrapped as a PolicyIK-style plant.

The arm is driven through a *joint reference* ``z`` that a finite-gain joint PD
controller tracks for one control period -- the same interface the policy and
the classical IK baselines command. Because the gains are finite and gravity is
on, the arm settles slightly away from ``z``; retaining that offset is the whole
point of PolicyIK's reference accumulation.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from . import kinematics as K

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "ur5e", "scene.xml",
)

# A comfortable, elbow-up "home" well inside the joint limits.
HOME_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


class ArmModel:
    """UR5e plant with a joint-reference PD interface, FK and Jacobian.

    Parameters
    ----------
    ee_site:
        Name of the end-effector site. UR5e exposes ``attachment_site``.
    control_dt:
        Policy/controller period in seconds. Defaults to 1/30 s, the PolicyIK
        control period. The physics is substepped to the nearest multiple of the
        model timestep.
    kp, kv:
        Optional per-joint PD gains. When given they overwrite the model's
        native position-servo gains, exposing stiffness as a tunable quantity.
    """

    def __init__(
        self,
        model_path: str | None = None,
        ee_site: str = "attachment_site",
        control_dt: float = 1.0 / 30.0,
        kp: np.ndarray | float | None = None,
        kv: np.ndarray | float | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path or _DEFAULT_MODEL)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        self.nv = self.model.nv

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id < 0:
            raise ValueError(f"end-effector site {ee_site!r} not found in model")

        # Actuators are ordered to match the joints; capture the dof/qpos ids.
        self.actuator_ids = np.arange(self.nu)
        self.joint_ids = np.array(
            [self.model.actuator_trnid[i, 0] for i in range(self.nu)]
        )
        self.qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_ids = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.joint_range = self.model.jnt_range[self.joint_ids].copy()

        if kp is not None or kv is not None:
            self.set_pd_gains(kp, kv)

        self.control_dt = float(control_dt)
        self.n_substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.reset(HOME_Q)

    # -- PD gains ---------------------------------------------------------
    def set_pd_gains(self, kp: np.ndarray | float | None, kv: np.ndarray | float | None) -> None:
        """Overwrite the native position-servo gains (kp, kv) per joint."""
        if kp is not None:
            kp = np.broadcast_to(np.asarray(kp, dtype=float), (self.nu,))
            self.model.actuator_gainprm[:, 0] = kp
            self.model.actuator_biasprm[:, 1] = -kp
        if kv is not None:
            kv = np.broadcast_to(np.asarray(kv, dtype=float), (self.nu,))
            self.model.actuator_biasprm[:, 2] = -kv

    @property
    def kp(self) -> np.ndarray:
        return self.model.actuator_gainprm[:, 0].copy()

    @property
    def kv(self) -> np.ndarray:
        return -self.model.actuator_biasprm[:, 2].copy()

    # -- state ------------------------------------------------------------
    def reset(self, q: np.ndarray, z: np.ndarray | None = None) -> None:
        """Place the arm at configuration ``q`` at rest, reference ``z`` (=q by default)."""
        q = np.asarray(q, dtype=float)
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel[self.dof_ids] = 0.0
        self.data.ctrl[self.actuator_ids] = q if z is None else np.asarray(z, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self.dof_ids].copy()

    @property
    def z(self) -> np.ndarray:
        """Current joint reference held by the PD controller."""
        return self.data.ctrl[self.actuator_ids].copy()

    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1])

    # -- stepping ---------------------------------------------------------
    def step(self, z: np.ndarray) -> None:
        """Command joint reference ``z`` and advance one control period."""
        self.data.ctrl[self.actuator_ids] = self.clip_to_limits(np.asarray(z, dtype=float))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def settle(self, z: np.ndarray, periods: int = 200) -> None:
        """Hold reference ``z`` until the arm comes to rest (for FK-of-a-config)."""
        for _ in range(periods):
            self.step(z)

    # -- kinematics -------------------------------------------------------
    def fk(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics of the end-effector site: (pos (3,), quat wxyz (4,)).

        With ``q=None`` it reads the current state; otherwise it evaluates FK at
        ``q`` without disturbing the live simulation state.
        """
        if q is None:
            pos = self.data.site_xpos[self.ee_site_id].copy()
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
            return pos, quat
        saved = self.data.qpos.copy()
        self.data.qpos[self.qpos_ids] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.ee_site_id].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        self.data.qpos[:] = saved
        mujoco.mj_kinematics(self.model, self.data)
        return pos, quat

    def jacobian(self) -> np.ndarray:
        """World-frame 6xN end-effector Jacobian [linear; angular] at the current state."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return np.vstack([jacp[:, self.dof_ids], jacr[:, self.dof_ids]])

    def pose_error(self, goal_pos: np.ndarray, goal_quat: np.ndarray) -> tuple[float, float]:
        """Scalar (position_m, orientation_rad) error of the current pose to the goal."""
        pos, quat = self.fk()
        return K.pose_error(pos, quat, goal_pos, goal_quat)

    # -- goal sampling ----------------------------------------------------
    def sample_reachable_goal(
        self, rng: np.random.Generator, settle: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random in-limits configuration and return (q*, goal_pos, goal_quat).

        Mirrors PolicyIK's data generation: goals are poses of actual reachable
        configurations, so a feasible solution exists. When ``settle`` is True the
        pose is taken after the arm settles under gravity at ``z = q*`` -- the pose
        it would actually hold -- otherwise the kinematic pose of ``q*`` is used.
        """
        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]
        # keep a small margin off the hard limits
        margin = 0.05 * (hi - lo)
        q_star = rng.uniform(lo + margin, hi - margin)
        if settle:
            saved_q, saved_v, saved_c = (
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.data.ctrl.copy(),
            )
            self.reset(q_star)
            self.settle(q_star)
            goal_pos, goal_quat = self.fk()
            self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = saved_q, saved_v, saved_c
            mujoco.mj_forward(self.model, self.data)
        else:
            goal_pos, goal_quat = self.fk(q_star)
        return q_star, goal_pos, goal_quat
