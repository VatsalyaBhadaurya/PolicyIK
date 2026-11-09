"""Phase A smoke test: benchmark the classical DLS IK baseline on the UR5e.

Samples reachable full-pose goals, runs closed-loop DLS through the arm's joint
PD controller, and reports PolicyIK-style success metrics. This is the harness
the learned PolicyIK policy will later plug into for an apples-to-apples
comparison.

    python -m mujoco_policyik.scripts.demo_dls --tasks 200
"""

from __future__ import annotations

import argparse

import numpy as np

from ..arm import ArmModel, HOME_Q
from ..baselines.dls_ik import DLSController
from ..metrics import summarize, terminal_errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=100, help="number of random reach tasks")
    p.add_argument("--steps", type=int, default=160, help="control steps per task (5.3 s at 30 Hz)")
    p.add_argument("--damping", type=float, default=0.05, help="DLS damping lambda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arm = ArmModel()
    rng = np.random.default_rng(args.seed)
    print(
        f"UR5e: {arm.nu} DoF | control_dt={arm.control_dt:.4f}s "
        f"({arm.n_substeps} substeps) | kp={arm.kp} kv={arm.kv}"
    )

    results = []
    for i in range(args.tasks):
        q_star, goal_pos, goal_quat = arm.sample_reachable_goal(rng)
        arm.reset(HOME_Q)
        ctrl = DLSController(arm, damping=args.damping)
        res = ctrl.run(goal_pos, goal_quat, steps=args.steps)
        results.append(res)
        if i < 5:
            mp, mo = terminal_errors(res["e_pos"], res["e_ori"])
            print(f"  task {i}: final err {1e3 * mp:7.3f} mm / {np.degrees(mo):6.3f} deg")

    s = summarize(results)
    print("\n=== DLS baseline (classical IK) on UR5e, MuJoCo ===")
    print(f"  tasks            : {s['n']}")
    print(f"  strict success   : {s['strict_pct']:.1f}%  (<=1 mm, 0.2 deg over final 1 s)")
    print(f"  coarse success   : {s['coarse_pct']:.1f}%  (<=1 cm, 2 deg)")
    print(f"  median pos error : {s['median_pos_mm']:.3f} mm")
    print(f"  median ori error : {s['median_ori_deg']:.3f} deg")


if __name__ == "__main__":
    main()
