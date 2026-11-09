"""Benchmark a trained PolicyIK policy against classical IK on identical tasks.

Runs the learned policy, DLS and QP on the same held-out start/goal set and
reports PolicyIK-style metrics (strict/coarse success, median terminal error) --
the 6-DoF MuJoCo analogue of Table 1.

    python -m mujoco_policyik.scripts.evaluate_policy --model outputs/policyik_ppo --tasks 200
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..arm import ArmModel
from ..baselines.dls_ik import DLSController
from ..baselines.qp_ik import QPController
from ..env import PolicyIKEnv
from ..metrics import summarize


def build_tasks(n: int, seed: int):
    arm = ArmModel()
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n):
        q_star, gpos, gquat = arm.sample_reachable_goal(rng)
        lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
        q0 = arm.clip_to_limits(rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        tasks.append((q0, gpos, gquat))
    return tasks


def run_baseline(make_ctrl, tasks, steps):
    arm = ArmModel()
    out = []
    for q0, gpos, gquat in tasks:
        arm.reset(q0)
        out.append(make_ctrl(arm).run(gpos, gquat, steps=steps))
    return out


def run_policy(model, obs_rms, clip_obs, tasks, steps):
    env = PolicyIKEnv(horizon=steps)
    eps = 1e-8
    out = []
    for q0, gpos, gquat in tasks:
        obs, _ = env.reset(options={"goal_pos": gpos, "goal_quat": gquat, "q_star": q0, "q_start": q0})
        e_pos = np.empty(steps)
        e_ori = np.empty(steps)
        for t in range(steps):
            norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip_obs, clip_obs)
            action, _ = model.predict(norm.astype(np.float32), deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            e_pos[t], e_ori[t] = info["e_pos"], info["e_ori"]
        out.append({"e_pos": e_pos, "e_ori": e_ori})
    return out


def _row(name, s):
    print(f"  {name:12s} | {s['strict_pct']:6.1f} | {s['coarse_pct']:6.1f} | "
          f"{s['median_pos_mm']:9.3f} | {s['median_ori_deg']:8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="directory with model.zip and vecnormalize.pkl")
    p.add_argument("--tasks", type=int, default=200)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    tasks = build_tasks(args.tasks, args.seed)

    model = PPO.load(f"{args.model}/model", device="cpu")
    vn = VecNormalize.load(f"{args.model}/vecnormalize.pkl", DummyVecEnv([lambda: PolicyIKEnv()]))

    results = {
        "PolicyIK": run_policy(model, vn.obs_rms, vn.clip_obs, tasks, args.steps),
        "DLS": run_baseline(lambda a: DLSController(a, damping=0.05), tasks, args.steps),
        "QP": run_baseline(lambda a: QPController(a, damping=0.02), tasks, args.steps),
    }

    print(f"\n=== PolicyIK vs classical IK on {args.tasks} UR5e tasks (MuJoCo) ===")
    print("  controller   | strict | coarse | med pos mm | med ori deg")
    print("  " + "-" * 56)
    for name, res in results.items():
        _row(name, summarize(res))


if __name__ == "__main__":
    main()
