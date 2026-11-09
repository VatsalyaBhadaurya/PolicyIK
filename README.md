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

## Status — Phase B (PolicyIK formulation) ✅

The three design pillars (Sections 4.1–4.3), ported to 6-DoF:

| Piece | Module | Notes |
|---|---|---|
| Error-dependent action gain, multiscale pose channels, keypoints, reward stack | `mujoco_policyik/formulation.py` | Eq. 2, 6, 7, 8, 10; all constants from the released config |
| Goal-conditioned env (50-D obs, 6-D action) | `mujoco_policyik/env.py` | accumulated reference + command-state feedback `d_t = z_t − q_t` |
| Env verification | `mujoco_policyik/scripts/check_env.py` | shapes, gain schedule, hold-from-goal, DLS-oracle plumbing check |

The **50-D observation**: `[q−q_nom(6), q̇(6), goal_pos(3), keypoint_offsets(9),
pose_coarse(6), pose_fine(6), error_history(2), command_state(6), prev_action(6)]`.
The accumulated-reference action is verified to reproduce the DLS plant to
machine precision (`check_env` test 4), confirming the formulation is wired
correctly and the task is solvable before any training.

## Status — Phase C (training + QP baseline) ✅

| Piece | Module | Notes |
|---|---|---|
| Resolved-rate QP IK baseline | `mujoco_policyik/baselines/qp_ik.py` | box-constrained (joint limits) via `lsq_linear`; velocity-scaled |
| PPO training (SB3) | `mujoco_policyik/train.py` | shared [256,256] ELU MLP, VecNormalize obs, γ0.99/GAE0.95 |
| Reverse curriculum | `mujoco_policyik/curriculum.py` | start spread grows from near-goal to full-random |
| Policy vs DLS vs QP evaluation | `mujoco_policyik/scripts/evaluate_policy.py` | 6-DoF analogue of Table 1 |

Both classical baselines reproduce the paper's ordering (QP slightly beats DLS,
lower median error). Training throughput is ~400–800 env-steps/s on CPU; the
paper's precision needs a large budget (it trained 4,096 GPU-parallel envs for
13k PPO epochs), so the included run is a single-workstation demonstration.

### Quick start

```bash
pip install -r requirements.txt
python -m mujoco_policyik.scripts.demo_dls --tasks 200    # classical DLS baseline
python -m mujoco_policyik.scripts.check_env --tasks 60    # verify the PolicyIK env
python -m mujoco_policyik.train --timesteps 1_200_000 --n-envs 16   # train PolicyIK
python -m mujoco_policyik.scripts.evaluate_policy --model outputs/policyik_ppo --tasks 200
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

- **Phase C** — vectorised env + PPO training (reverse curriculum via
  `start_near_goal_std`, near-goal sampling); add a QP IK baseline.
- **Phase D** — benchmark the trained PolicyIK policy vs. DLS/QP on held-out
  6-DoF tasks, stressing joint-limit / singularity cases.

## Credits

Method and reference implementation: *PolicyIK: High-Precision Inverse
Kinematics for Robot Arms with Reinforcement Learning*, Liu, Liu, Dong, Ross,
Wan (2026). UR5e model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(BSD-3), under `models/ur5e/`.
