"""Generate figures/renders for the upstream issue (saved to assets/)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ..arm import ArmModel, HOME_Q
from ..baselines.dls_ik import DLSController
from ..baselines.qp_ik import QPController
from ..metrics import success, terminal_errors

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
os.makedirs(ASSETS, exist_ok=True)
HZ = 30.0


def convergence_fig(arm):
    """A successful DLS reach: position/orientation error vs time (mirrors paper Fig 1b)."""
    rng = np.random.default_rng(0)
    trace = q_star = None
    for _ in range(300):
        qs, gp, gq = arm.sample_reachable_goal(rng)
        arm.reset(HOME_Q)
        r = DLSController(arm, damping=0.05).run(gp, gq, steps=160)
        if success(r["e_pos"], r["e_ori"]):
            trace, q_star = r, qs
            break
    t = np.arange(160) / HZ
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, 1e3 * trace["e_pos"], color="#1f77b4", label="position error (mm)")
    ax.plot(t, np.degrees(trace["e_ori"]), color="#ff7f0e", label="orientation error (deg)")
    ax.axhline(1.0, ls="--", color="#1f77b4", alpha=0.5)
    ax.axhline(0.2, ls="--", color="#ff7f0e", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)"); ax.set_ylabel("error (log scale)")
    ax.set_title("PolicyIK MuJoCo port — DLS reach & hold on UR5e (1 mm / 0.2°)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "convergence.png"), dpi=140)
    plt.close(fig)
    return q_star


def deployment_fig(arm):
    """Single-attempt vs +deployment strict success for DLS and QP."""
    rng = np.random.default_rng(0)
    tasks = []
    for _ in range(120):
        qs, gp, gq = arm.sample_reachable_goal(rng)
        lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
        q0 = arm.clip_to_limits(rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        tasks.append((q0, gp, gq))
    from ..deployment import run_deployment
    out = {}
    for name, mk in [("DLS", lambda a: DLSController(a, damping=0.05)),
                     ("QP", lambda a: QPController(a, damping=0.05))]:
        s = d = 0
        for i, (q0, gp, gq) in enumerate(tasks):
            arm.reset(q0)
            r = mk(arm).run(gp, gq, steps=160)
            s += success(r["e_pos"], r["e_ori"])
            tr, _ = run_deployment(arm, mk, q0, gp, gq, steps=160, max_attempts=4,
                                   rng=np.random.default_rng(100 + i))
            d += success(tr["e_pos"], tr["e_ori"])
        out[name] = (100 * s / len(tasks), 100 * d / len(tasks))
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(out)); w = 0.35
    ax.bar(x - w / 2, [out[k][0] for k in out], w, label="single attempt", color="#9ecae1")
    ax.bar(x + w / 2, [out[k][1] for k in out], w, label="+ deployment protocol", color="#3182bd")
    for i, k in enumerate(out):
        ax.text(i - w / 2, out[k][0] + 1, f"{out[k][0]:.0f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, out[k][1] + 1, f"{out[k][1]:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(list(out)); ax.set_ylabel("strict success (%)")
    ax.set_title("Deployment protocol (retries) raises holding success — UR5e")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "deployment.png"), dpi=140)
    plt.close(fig)


def render_fig(arm, q_star):
    """Render the UR5e at a reached goal configuration."""
    m, d = arm.model, arm.data
    arm.reset(HOME_Q if q_star is None else q_star)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 120.0, -12.0, 1.25
    cam.lookat[:] = [0.0, -0.1, 0.45]
    r = mujoco.Renderer(m, 600, 800)
    r.update_scene(d, camera=cam)
    img = r.render()
    plt.imsave(os.path.join(ASSETS, "ur5e_reach.png"), img)


def main():
    arm = ArmModel()
    q_star = convergence_fig(arm)
    deployment_fig(arm)
    render_fig(arm, q_star)
    print("wrote assets/convergence.png, assets/deployment.png, assets/ur5e_reach.png")


if __name__ == "__main__":
    main()
