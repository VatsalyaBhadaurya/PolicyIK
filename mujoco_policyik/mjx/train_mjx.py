"""GPU-parallel PolicyIK training on MJX with Brax PPO.

Thousands of MJX environments give the low-variance advantages that the CPU
SB3 run lacked. Brax's PPO uses a tanh-squashed Normal policy (correct log-prob
correction), observation normalisation, and GAE -- the pieces that make
high-precision regulation learnable.

Run inside WSL with the CUDA JAX venv:
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python -m mujoco_policyik.mjx.train_mjx --timesteps 50000000 --num-envs 2048
"""

from __future__ import annotations

import argparse
import functools
import os
import pickle
import time

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from .env import PolicyIKMJX

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs", "mjx_ppo", "params.pkl",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=50_000_000)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--episode-length", type=int, default=160)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--updates-per-batch", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=1e-3)
    p.add_argument("--reward-scaling", type=float, default=0.1)
    p.add_argument("--num-evals", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=_DEFAULT_OUT)
    args = p.parse_args()

    env = PolicyIKMJX()
    net_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )

    t0 = time.time()

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        ep = metrics.get("eval/episode_e_pos", float("nan"))
        wt = metrics.get("eval/episode_within_tol", float("nan"))
        # metrics are summed over the episode by the eval wrapper
        print(f"[{time.time() - t0:6.0f}s] step {int(step):>10} | reward {float(r):8.2f} "
              f"| mean e_pos {1e3 * float(ep) / args.episode_length:8.2f} mm "
              f"| in-tol steps {float(wt):6.1f}/{args.episode_length}", flush=True)

    make_inference, params, _ = ppo.train(
        environment=env,
        num_timesteps=args.timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.updates_per_batch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        entropy_cost=args.entropy,
        discounting=0.99,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        reward_scaling=args.reward_scaling,
        normalize_observations=True,
        network_factory=net_factory,
        num_evals=args.num_evals,
        seed=args.seed,
        progress_fn=progress,
    )

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(params, f)
    print(f"[train_mjx] saved params to {args.out}", flush=True)


if __name__ == "__main__":
    main()
