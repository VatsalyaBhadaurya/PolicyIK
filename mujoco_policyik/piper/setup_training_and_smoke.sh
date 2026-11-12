#!/usr/bin/env bash
# Install lerobot[training] (accelerate), then run the 2-step ACT smoke.
source ~/lerobot-venv/bin/activate
pip install 'lerobot[training]' --retries 10 --timeout 180 2>&1 | tail -5
echo TRAINEXTRA_DONE
export HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot
rm -rf ~/act_piper_smoke
lerobot-train \
  --dataset.repo_id=aiengineer56/piper_dataset_1100 \
  --dataset.root=$HOME/piper_train \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=2 \
  --steps=2 \
  --num_workers=2 \
  --save_checkpoint=false \
  --log_freq=1 \
  --output_dir=$HOME/act_piper_smoke \
  --job_name=piper_act_smoke 2>&1 | tail -45
echo "SMOKE_EXIT=${PIPESTATUS[0]}"
