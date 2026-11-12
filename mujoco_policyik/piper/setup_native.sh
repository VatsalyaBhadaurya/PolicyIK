#!/usr/bin/env bash
# Move the Piper ACT dataset into the native Ubuntu filesystem and check torch.
source ~/lerobot-venv/bin/activate
echo "=== torch import check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "TORCH_FAILED=$?"
echo "=== copy dataset to ~/piper_train (native ext4, fast I/O) ==="
mkdir -p ~/piper_train
cp -rL /mnt/d/vatty/work/Policy-IK/piper_train/. ~/piper_train/
echo "=== result ==="
du -sh ~/piper_train
find ~/piper_train -maxdepth 2 -type d | sort
find ~/piper_train -name "*.mp4" | wc -l
echo DONE
