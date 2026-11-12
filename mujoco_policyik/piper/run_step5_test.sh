#!/usr/bin/env bash
# Ensure mujoco is available, then run the Step 5 offline test (CPU, no GPU contention).
source ~/lerobot-venv/bin/activate
python -c "import mujoco" 2>/dev/null || pip install -q mujoco 2>&1 | tail -2
echo "--- checkpoint ---"
ls ~/act_piper_eepose/checkpoints/last/pretrained_model/ 2>/dev/null || echo "NO CHECKPOINT"
echo "--- run test ---"
PYTHONPATH=/mnt/d/vatty/work/Policy-IK python -m mujoco_policyik.piper.test_step5 --device cpu --n 40 2>&1 | grep -vE "libnvrtc|torchcodec|pyav" | tail -25
