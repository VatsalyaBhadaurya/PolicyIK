"""Phase D: benchmark an MJX-trained PolicyIK policy against DLS and QP.

Runs the learned policy (Brax PPO params) and the two classical IK baselines on
the *same* held-out start/goal set and reports PolicyIK-style metrics -- the
6-DoF MuJoCo analogue of the paper's Table 1. Runs inside WSL (GPU JAX), where
mujoco + scipy are also available for the CPU baselines.

    PYTHONPATH=. python -m mujoco_policyik.mjx.evaluate_mjx \
        --params outputs/mjx_ppo/params.pkl --tasks 256
"""

from __future__ import annotations

import argparse
import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from ..arm import ArmModel
from ..baselines.dls_ik import DLSController
from ..baselines.qp_ik import QPController
from ..metrics import summarize
from .env import PolicyIKMJX


def build_tasks(n: int, seed: int):
    """Fixed held-out tasks: random in-limits start, goal = FK(q*) (matches MJX training)."""
    arm = ArmModel()
    rng = np.random.default_rng(seed)
    q0s, gps, gqs = [], [], []
    for _ in range(n):
        q_star, gp, gq = arm.sample_reachable_goal(rng, settle=False)
        lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
        q0 = arm.clip_to_limits(rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        q0s.append(q0); gps.append(gp); gqs.append(gq)
    return np.array(q0s), np.array(gps), np.array(gqs)


def run_baseline(make_ctrl, q0s, gps, gqs, steps):
    arm = ArmModel()
    out = []
    for q0, gp, gq in zip(q0s, gps, gqs):
        arm.reset(q0)
        out.append(make_ctrl(arm).run(gp, gq, steps=steps))
    return out


def run_policy(params, q0s, gps, gqs, steps):
    env = PolicyIKMJX()
    normalize = running_statistics.normalize
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=(256, 256), value_hidden_layer_sizes=(256, 256))
    inference = ppo_networks.make_inference_fn(net)(params, deterministic=True)

    reset_to = jax.jit(jax.vmap(env.reset_to))
    step = jax.jit(jax.vmap(env.step))
    qs = jnp.asarray(q0s); gp = jnp.asarray(gps); gq = jnp.asarray(gqs)

    @jax.jit
    def rollout(qs, gp, gq):
        st = reset_to(qs, gp, gq)
        def body(carry, _):
            st, key = carry
            key, k = jax.random.split(key)
            act, _ = inference(st.obs, k)
            st = step(st, act)
            return (st, key), (st.metrics["e_pos"], st.metrics["e_ori"])
        _, (ep, eo) = jax.lax.scan(body, (st, jax.random.PRNGKey(0)), None, length=steps)
        return ep, eo  # (steps, N)

    ep, eo = rollout(qs, gp, gq)
    ep, eo = np.array(ep).T, np.array(eo).T  # (N, steps)
    return [{"e_pos": ep[i], "e_ori": eo[i]} for i in range(ep.shape[0])]


def _row(name, s):
    print(f"  {name:12s} | {s['strict_pct']:6.1f} | {s['coarse_pct']:6.1f} | "
          f"{s['median_pos_mm']:9.3f} | {s['median_ori_deg']:8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", required=True, help="MJX Brax PPO params.pkl")
    p.add_argument("--tasks", type=int, default=256)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    q0s, gps, gqs = build_tasks(args.tasks, args.seed)
    with open(args.params, "rb") as f:
        params = pickle.load(f)

    results = {
        "PolicyIK-MJX": run_policy(params, q0s, gps, gqs, args.steps),
        "DLS": run_baseline(lambda a: DLSController(a, damping=0.05), q0s, gps, gqs, args.steps),
        "QP": run_baseline(lambda a: QPController(a, damping=0.05), q0s, gps, gqs, args.steps),
    }

    print(f"\n=== Phase D: PolicyIK-MJX vs classical IK on {args.tasks} UR5e tasks ===")
    print("  controller   | strict | coarse | med pos mm | med ori deg")
    print("  " + "-" * 56)
    for name, res in results.items():
        _row(name, summarize(res))


if __name__ == "__main__":
    main()
