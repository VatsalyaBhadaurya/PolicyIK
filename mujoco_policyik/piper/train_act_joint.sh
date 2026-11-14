#!/usr/bin/env bash
# Baseline ACT: direct joint-angle output, same config/images as the EE-pose run.
source ~/lerobot-venv/bin/activate
export HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot
export PYTHONUNBUFFERED=1
lerobot-train \
  --dataset.repo_id=aiengineer56/piper_dataset_1100 \
  --dataset.root=$HOME/piper_train_joint \
  --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=true \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=16 \
  --num_workers=4 \
  --dataloader_multiprocessing_context=spawn \
  --steps=15000 \
  --save_freq=5000 \
  --save_checkpoint=true \
  --log_freq=50 \
  --output_dir=$HOME/act_piper_joint \
  --job_name=piper_act_joint
