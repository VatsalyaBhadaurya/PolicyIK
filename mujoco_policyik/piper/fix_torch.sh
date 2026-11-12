#!/usr/bin/env bash
# Robustly install stable torch 2.8 / CUDA-12.8 (retries survive network blips).
source ~/lerobot-venv/bin/activate
ok=0
for a in 1 2 3 4 5 6 7 8; do
  echo "=== install attempt $a ==="
  pip install "torch==2.8.0" "torchvision==0.23.0" \
      --index-url https://download.pytorch.org/whl/cu128 \
      --retries 10 --timeout 180 && { ok=1; break; }
  echo "attempt $a failed; sleeping"; sleep 8
done
echo "FIXDONE ok=$ok"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "TORCH_FAILED=$?"
python -c "import lerobot; print('lerobot import ok')" || echo "LEROBOT_FAILED=$?"
