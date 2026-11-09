# GPU-parallel PolicyIK (MJX + Brax PPO)

A JAX/MJX re-implementation of the PolicyIK env so thousands of UR5e
environments step on the GPU, giving the low-variance advantages that the
CPU SB3 PPO run lacked. This is the path toward the paper's precision on this
6-DoF port.

## Files

| File | Purpose |
|---|---|
| `jax_formulation.py` | PolicyIK reward / kinematics in functional JAX (matches the CPU `formulation.py`) |
| `env.py` | `PolicyIKMJX` — batched Brax `Env`: MJX plant + accumulated-reference action + 50-D obs/reward |
| `train_mjx.py` | Brax PPO trainer (tanh-squashed Normal policy, obs normalization, GAE) |
| `requirements_mjx.txt` | pinned, mutually compatible GPU stack |
| `setup_wsl.sh` | one-shot venv + install helper |

## Why WSL2

Native-Windows JAX is **CPU-only**; GPU JAX needs Linux or **WSL2**. This repo
was validated in WSL2 Ubuntu with an RTX 4050 (CUDA via `/usr/lib/wsl/lib`).

## Setup (once)

```bash
# inside WSL2 Ubuntu
python3 -m venv ~/pik && source ~/pik/bin/activate
pip install -U pip
pip install -r /mnt/<drive>/.../Policy-IK/mujoco_policyik/mjx/requirements_mjx.txt
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

## Train

```bash
source ~/pik/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd /mnt/<drive>/.../Policy-IK
PYTHONPATH=. python -m mujoco_policyik.mjx.train_mjx \
    --timesteps 30000000 --num-envs 2048 --num-evals 30
```

Progress prints eval reward, mean terminal `e_pos` (mm) and in-tolerance step
count. Params are saved to `outputs/mjx_ppo/params.pkl`.

## Measured throughput

On an RTX 4050 Laptop (6 GB), ~1.8k control-steps/s at 2048 envs (each control
step = 17 MJX substeps). Rough budgets: ~10M steps ≈ 1.5 h, ~30M ≈ 4.5 h,
~50M ≈ 7.5 h. A bigger GPU (more VRAM → more envs) scales this up substantially.

## Status

Pipeline verified end-to-end on GPU (env jit+vmap, Brax PPO trains/evals/saves);
reward improves early. Reaching the paper's 1 mm / 0.2° precision is a
multi-hour run on this laptop GPU and is left as a user-launched job.
