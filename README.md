# PolicyIK — MuJoCo 6-DoF port

A MuJoCo re-implementation of the **PolicyIK** setting
([Liu et al., 2026](https://arxiv.org/abs/2611.05372)) targeting a **6-DoF arm
(UR5e)**, so the learned high-precision IK policy can be benchmarked against
classical ("normal") IK in the same simulator.

The original work trains a goal-conditioned RL policy that reaches and holds
full end-effector poses within **1 mm / 0.2°**, beating damped-least-squares
(DLS), QP and MPC IK. The reference code is built on Isaac Lab for a 7-DoF
OpenArm; this repo rebuilds the plant and baselines in MuJoCo for a 6-DoF arm.

## Status — Phase A (plant + classical baseline) ✅

| Piece | Module | Notes |
|---|---|---|
| Pose math (quat/rotvec, pose error) | `mujoco_policyik/kinematics.py` | ports `quat_error_rotvec_body`, Eq. 1 errors |
| UR5e plant w/ finite-gain joint PD, FK, Jacobian, goal sampling | `mujoco_policyik/arm.py` | reference-tracking interface `z` |
| Classical IK baseline (DLS) | `mujoco_policyik/baselines/dls_ik.py` | the "normal IK" comparator |
| Success metrics (final-1 s window) | `mujoco_policyik/metrics.py` | strict 1 mm/0.2°, coarse 1 cm/2° |
| Benchmark harness / smoke test | `mujoco_policyik/scripts/demo_dls.py` | — |

### Quick start

```bash
pip install -r requirements.txt
python -m mujoco_policyik.scripts.demo_dls --tasks 200
```

The arm is driven through a joint reference `z` tracked by a finite-gain PD
controller; under gravity it settles slightly off `z`, the behaviour PolicyIK's
reference accumulation is designed around.

### Baseline sanity check

On 100 random reachable full-pose tasks the DLS baseline reproduces the paper's
reported DLS behaviour closely:

| | strict | median pos |
|---|---|---|
| This port (UR5e) | 51.0% | 0.95 mm |
| Paper, Table 1 (OpenArm 7-DoF) | 50.4% | 0.94 mm |

Large-error tasks correspond to goals near joint limits / singularities, the
classical-IK failure mode PolicyIK is designed to overcome.

## Next phases

- **Phase B** — port the PolicyIK formulation to 6-DoF: accumulated
  joint-reference action with error-dependent gain, the 50-D observation
  (incl. command-state feedback `d_t = z_{t-1} − q_t` and multiscale pose
  channels), and the progress/dwell reward stack.
- **Phase C** — vectorised Gym env + PPO training; add a QP IK baseline.
- **Phase D** — benchmark PolicyIK vs. DLS/QP on held-out 6-DoF tasks,
  stressing joint-limit / singularity cases.

## Credits

Method and reference implementation: *PolicyIK: High-Precision Inverse
Kinematics for Robot Arms with Reinforcement Learning*, Liu, Liu, Dong, Ross,
Wan (2026). UR5e model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(BSD-3), under `models/ur5e/`.
