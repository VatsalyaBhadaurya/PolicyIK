"""GPU-parallel PolicyIK on MJX (JAX) for large-scale PPO training.

This package mirrors the CPU formulation in ``mujoco_policyik`` but in JAX so the
UR5e plant and the PolicyIK reward/observation run batched on the GPU, giving the
thousands of parallel environments PPO needs for low-variance advantages.

Run inside WSL (Linux) with a CUDA-enabled JAX; native Windows JAX is CPU only.
"""
