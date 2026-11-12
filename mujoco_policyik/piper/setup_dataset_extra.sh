#!/usr/bin/env bash
# Install LeRobot's [dataset] extra, then verify dataset load + ACT CLI.
source ~/lerobot-venv/bin/activate
pip install 'lerobot[dataset]' --retries 10 --timeout 180 2>&1 | tail -6
echo EXTRADONE
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("aiengineer56/piper_dataset_1100", root="/home/vatty/piper_train")
print("episodes", ds.num_episodes, "frames", ds.num_frames)
s = ds[0]
print("sample keys:", [k for k in s])
def sh(x): return tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
print("action", sh(s["action"]), "state", sh(s["observation.state"]))
for k in s:
    if k.startswith("observation.images"):
        print(k, sh(s[k]), getattr(s[k], "dtype", ""))
PY
echo "=== lerobot-train --help (head) ==="
lerobot-train --help 2>&1 | head -45
