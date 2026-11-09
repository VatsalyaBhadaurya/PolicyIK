"""Measure the deployment protocol: single attempt vs retries, on DLS and QP.

Demonstrates the controller-agnostic improvement from `mujoco_policyik.deployment`
on the 6-DoF UR5e -- the analogue of the paper's "+ deployment" row, which it
reuses to lift PolicyIK from 93.5% to 98.9%.

    python -m mujoco_policyik.scripts.evaluate_deployment --tasks 200
"""

from __future__ import annotations

import argparse

import numpy as np

from ..arm import ArmModel
from ..baselines.dls_ik import DLSController
from ..baselines.qp_ik import QPController
from ..deployment import run_deployment
from ..metrics import summarize


def build_tasks(arm, n, seed):
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n):
        q_star, gp, gq = arm.sample_reachable_goal(rng)
        lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
        q0 = arm.clip_to_limits(rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        tasks.append((q0, gp, gq))
    return tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=150)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arm = ArmModel()
    tasks = build_tasks(arm, args.tasks, args.seed)
    controllers = {
        "DLS": lambda a: DLSController(a, damping=0.05),
        "QP": lambda a: QPController(a, damping=0.05),
    }

    print(f"\n=== Deployment protocol on {args.tasks} UR5e tasks "
          f"(up to {args.max_attempts} attempts) ===")
    print("  controller         | strict | coarse | med pos mm | E[N]")
    print("  " + "-" * 58)
    for name, mk in controllers.items():
        single, deploy, n_att = [], [], []
        for i, (q0, gp, gq) in enumerate(tasks):
            arm.reset(q0)
            single.append(mk(arm).run(gp, gq, steps=args.steps))
            tr, n = run_deployment(arm, mk, q0, gp, gq, steps=args.steps,
                                   max_attempts=args.max_attempts,
                                   rng=np.random.default_rng(args.seed + 1 + i))
            deploy.append(tr)
            n_att.append(n)
        s0, s1 = summarize(single), summarize(deploy)
        print(f"  {name:<12s} single | {s0['strict_pct']:6.1f} | {s0['coarse_pct']:6.1f} "
              f"| {s0['median_pos_mm']:9.3f} | 1.00")
        print(f"  {name:<8s} +deployment | {s1['strict_pct']:6.1f} | {s1['coarse_pct']:6.1f} "
              f"| {s1['median_pos_mm']:9.3f} | {np.mean(n_att):.2f}")


if __name__ == "__main__":
    main()
