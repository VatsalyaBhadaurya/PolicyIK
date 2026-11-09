"""Phase B verification: exercise the PolicyIK env mechanics on the UR5e.

Checks, in order:

1. shapes            -- observation is 50-D, action 6-D, all obs finite;
2. gain schedule     -- beta ramps from g_min near the goal to 1 far from it;
3. reference holding  -- from the goal config a zero action holds within tol
                         (reference accumulation retains the PD rest offset);
4. DLS oracle         -- feeding DLS increments *through the env's action
                         interface* reproduces standalone DLS to machine
                         precision on identical tasks, confirming the
                         action/gain/reference plumbing is exact and the task is
                         solvable; also reports the env's own success rate.

    python -m mujoco_policyik.scripts.check_env --tasks 100
"""

from __future__ import annotations

import argparse

import numpy as np

from ..arm import ArmModel
from ..baselines.dls_ik import DLSController, world_twist_error
from ..env import ACT_DIM, OBS_DIM, PolicyIKEnv
from ..formulation import action_gain
from ..metrics import success, terminal_errors


def _dls_action(env: PolicyIKEnv, damping=0.05, step_gain=1.0, max_step=0.1) -> np.ndarray:
    """The action that makes the env's accumulated reference take one DLS step."""
    pos, quat = env.arm.fk()
    err = world_twist_error(pos, quat, env._goal_pos, env._goal_quat)
    J = env.arm.jacobian()
    dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
    delta_z = np.clip(step_gain * dq, -max_step, max_step)
    e_p, e_R = env.arm.pose_error(env._goal_pos, env._goal_quat)
    beta = action_gain(e_p, e_R, env.cfg)
    return delta_z / (beta * env.cfg.action_scale)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=100)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = PolicyIKEnv(horizon=args.steps, seed=args.seed)

    # 1. shapes / finiteness
    obs, info = env.reset(seed=args.seed)
    assert obs.shape == (OBS_DIM,), obs.shape
    assert env.action_space.shape == (ACT_DIM,)
    assert np.all(np.isfinite(obs)), "non-finite observation"
    print(f"[1] obs {obs.shape}, action {env.action_space.shape}, all finite  OK")

    # 2. gain schedule: far vs near the goal
    far = action_gain(0.5, np.deg2rad(90), env.cfg)
    near = action_gain(0.0, 0.0, env.cfg)
    print(f"[2] gain beta: far={far:.3f} (expect ~1.0), at-goal={near:.3f} (expect g_min=0.1)  "
          f"{'OK' if abs(far - 1.0) < 1e-6 and abs(near - env.cfg.g_min) < 1e-6 else 'FAIL'}")

    # 3. hold-from-goal: start at the goal config, zero action, check it stays in tol
    rng = np.random.default_rng(args.seed)
    q_star, gpos, gquat = env.arm.sample_reachable_goal(rng)
    obs, info = env.reset(options={"goal_pos": gpos, "goal_quat": gquat, "q_star": q_star, "q_start": q_star})
    held = []
    for _ in range(args.steps):
        obs, r, term, trunc, info = env.step(np.zeros(ACT_DIM))
        held.append(info["within_tol"])
    print(f"[3] hold-from-goal: within-tol on {100 * np.mean(held):.0f}% of steps, "
          f"final {1e3 * info['e_pos']:.3f} mm / {np.degrees(info['e_ori']):.3f} deg")

    # 4. DLS oracle through the env, compared to standalone DLS on identical tasks
    arm = ArmModel()
    task_rng = np.random.default_rng(args.seed + 1)
    tasks = []
    for _ in range(args.tasks):
        q_star, gpos, gquat = arm.sample_reachable_goal(task_rng)
        lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
        q0 = arm.clip_to_limits(task_rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        tasks.append((q0, gpos, gquat))

    max_diff_mm = 0.0
    env_results = []
    rew_terms_example = None
    for i, (q0, gpos, gquat) in enumerate(tasks):
        # standalone DLS
        arm.reset(q0)
        s = DLSController(arm, damping=0.05).run(gpos, gquat, steps=args.steps)
        s_pos = terminal_errors(s["e_pos"], s["e_ori"])[0]
        # same task through the env's action interface
        env.reset(options={"goal_pos": gpos, "goal_quat": gquat, "q_star": q0, "q_start": q0})
        e_pos = np.empty(args.steps)
        e_ori = np.empty(args.steps)
        for t in range(args.steps):
            obs, r, term, trunc, info = env.step(_dls_action(env))
            e_pos[t], e_ori[t] = info["e_pos"], info["e_ori"]
            if i == 0 and t == args.steps // 2:
                rew_terms_example = info["reward_terms"]
        env_results.append({"e_pos": e_pos, "e_ori": e_ori})
        max_diff_mm = max(max_diff_mm, 1e3 * abs(s_pos - terminal_errors(e_pos, e_ori)[0]))

    strict = 100.0 * np.mean([success(r["e_pos"], r["e_ori"]) for r in env_results])
    mp = np.median([terminal_errors(r["e_pos"], r["e_ori"])[0] for r in env_results])
    ok = max_diff_mm < 1e-6
    print(f"[4] env-vs-standalone on identical tasks: max |pos diff| = {max_diff_mm:.2e} mm  "
          f"{'OK (plumbing exact)' if ok else 'FAIL'}")
    print(f"    env DLS-oracle on random starts: strict {strict:.1f}%, median pos {1e3 * mp:.3f} mm "
          f"(random starts are harder than the HOME-start demo)")
    print(f"    reward terms at a mid-episode step: "
          f"{ {k: round(v, 4) for k, v in rew_terms_example.items()} }")


if __name__ == "__main__":
    main()
