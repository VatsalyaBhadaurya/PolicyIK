"""Train PolicyIK on the 6-DoF UR5e with PPO (Stable-Baselines3).

Mirrors the paper's PPO recipe where it maps cleanly onto SB3: a shared
[256, 256] ELU MLP, gamma 0.99, GAE 0.95, 5 epochs, clip 0.2, entropy 0.01, and
input normalisation. A reverse curriculum grows the start spread over training.

    python -m mujoco_policyik.train --timesteps 2_000_000 --n-envs 16 --subproc

Full precision needs a large budget (the paper trains 4,096 envs for 13k PPO
epochs); the defaults here are sized for a single workstation. Outputs a model
and the observation-normalisation statistics under ``--out``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .curriculum import ReverseCurriculumCallback
from .env import PolicyIKEnv


def make_env(seed: int, horizon: int, start_std: float | None):
    def _init():
        env = PolicyIKEnv(horizon=horizon, start_near_goal_std=start_std,
                          squash_actions=True, seed=seed)
        return env
    return _init


def build_vec_env(n_envs: int, horizon: int, start_std: float | None, subproc: bool,
                  seed: int, norm_reward: bool = True):
    fns = [make_env(seed + i, horizon, start_std) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if subproc else DummyVecEnv(fns)
    # reward normalisation stabilises PPO here: the progress term (weight ~27)
    # otherwise produces high-variance returns and saturates the policy mean.
    return VecNormalize(venv, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=128, help="rollout steps per env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--horizon", type=int, default=160)
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (true parallelism)")
    p.add_argument("--device", default="cpu", help="cpu is usually faster for a tiny MLP")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/policyik_ppo")
    p.add_argument("--no-curriculum", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    start_std = None if args.no_curriculum else 0.1
    venv = build_vec_env(args.n_envs, args.horizon, start_std, args.subproc, args.seed)

    policy_kwargs = dict(activation_fn=nn.ELU, net_arch=[256, 256])
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
        max_grad_norm=1.0, policy_kwargs=policy_kwargs, device=args.device,
        seed=args.seed, verbose=1,
    )

    callback = None if args.no_curriculum else ReverseCurriculumCallback(
        args.timesteps, std_start=0.1, std_end=1.5, warm_frac=0.8, verbose=1
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    model.save(os.path.join(args.out, "model"))
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"[train] saved model and normaliser to {args.out}")


if __name__ == "__main__":
    main()
