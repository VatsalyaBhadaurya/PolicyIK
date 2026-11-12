#!/usr/bin/env bash
# 2-step ACT smoke: does native-res (720p+480p) fit 6 GB? (OOM -> we downscale videos)
source ~/lerobot-venv/bin/activate
export HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot
lerobot-train \
  --dataset.repo_id=aiengineer56/piper_dataset_1100 \
  --dataset.root=$HOME/piper_train \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.push_to_hub=false \
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
  --job_name=piper_act_smoke 2>&1 | tail -40
echo "SMOKE_EXIT=${PIPESTATUS[0]}"
