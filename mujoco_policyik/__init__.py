"""A MuJoCo re-implementation of the PolicyIK setting for a 6-DoF arm.

Phase A provides the pieces a learned or classical controller needs:

* ``kinematics`` -- SO(3)/SE(3) pose math (quaternion <-> rotation-vector,
  pose error) following the conventions of the original PolicyIK code.
* ``arm``        -- a MuJoCo UR5e wrapper with a finite-gain joint PD
  controller, forward kinematics and the end-effector Jacobian.
* ``baselines.dls_ik`` -- a damped-least-squares differential IK controller,
  the classical ("normal") IK baseline PolicyIK is compared against.

The original paper and reference code (Isaac Lab, 7-DoF OpenArm) target
1 mm / 0.2 deg full-pose precision; those tolerances are reused here.
"""

TOL_POS_M = 1.0e-3        # 1 mm position tolerance
TOL_ORI_RAD = 3.4906585e-3  # 0.2 deg orientation tolerance

__all__ = ["TOL_POS_M", "TOL_ORI_RAD"]
