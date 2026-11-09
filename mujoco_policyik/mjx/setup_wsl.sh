#!/usr/bin/env bash
# Set up a CUDA JAX + MJX + Brax environment inside WSL for GPU-parallel training.
set -e
python3 -m venv ~/pik
source ~/pik/bin/activate
python -m pip install -U pip
python -m pip install "jax[cuda12]" mujoco mujoco-mjx brax
echo INSTALL_DONE
python - <<'PY'
import jax, mujoco, brax
print("jax", jax.__version__)
print("devices", jax.devices())
print("mujoco", mujoco.__version__)
PY
