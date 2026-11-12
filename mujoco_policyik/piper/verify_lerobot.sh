#!/usr/bin/env bash
# Confirm torch+CUDA, that LeRobot loads the local EE-pose dataset, and the ACT CLI.
source ~/lerobot-venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== load dataset ==="
python - <<'PY'
import importlib
LeRobotDataset = None
for mod in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
    try:
        LeRobotDataset = importlib.import_module(mod).LeRobotDataset
        print("import path:", mod); break
    except Exception as e:
        print("miss", mod, repr(e)[:80])
ds = LeRobotDataset("aiengineer56/piper_dataset_1100", root="/home/vatty/piper_train")
print("episodes", ds.num_episodes, "frames", ds.num_frames)
s = ds[0]
print("sample keys:", [k for k in s])
import numpy as np, torch
def sh(x): return tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
print("action", sh(s["action"]), "state", sh(s["observation.state"]))
for k in s:
    if k.startswith("observation.images"): print(k, sh(s[k]), s[k].dtype)
PY
echo "=== lerobot-train --help (first fields) ==="
lerobot-train --help 2>&1 | head -50
