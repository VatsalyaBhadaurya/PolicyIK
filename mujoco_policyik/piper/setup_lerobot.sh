#!/usr/bin/env bash
# Install LeRobot (ACT training) in a WSL Python 3.10+ venv with CUDA torch.
set -e
python3 -m venv ~/lerobot-venv
source ~/lerobot-venv/bin/activate
python -m pip install -U pip
python -m pip install lerobot
echo LEROBOT_DONE
python - <<'PY'
import lerobot, torch
print("lerobot", getattr(lerobot, "__version__", "ok"), "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
