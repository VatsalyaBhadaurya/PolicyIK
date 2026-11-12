"""Offline test of Step 5 (ACT -> EE-pose -> IK -> Piper joints).

Runs the trained ACT policy on real dataset observations and checks three things:
  1. ACT task-space accuracy  -- predicted EE pose vs the ground-truth EE-pose action
  2. IK executor accuracy      -- FK(IK(pose)) reproduces the commanded pose (residual ~0)
  3. the full chain produces a valid 6-joint reference + gripper per frame.

Run in WSL (lerobot venv):
    PYTHONPATH=/mnt/d/vatty/work/Policy-IK python -m mujoco_policyik.piper.test_step5
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/vatty/work/Policy-IK")

from mujoco_policyik.piper.act_ik_executor import ACTtoJoint
from mujoco_policyik.piper.piper_fk import rot6d_to_mat
from mujoco_policyik.piper.piper_ik import rotvec_from_mat


def _load_dataset(root):
    for mod in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
        try:
            LeRobotDataset = importlib.import_module(mod).LeRobotDataset
            return LeRobotDataset("aiengineer56/piper_dataset_1100", root=root)
        except Exception:
            continue
    raise RuntimeError("could not import LeRobotDataset")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/vatty/act_piper_eepose/checkpoints/last/pretrained_model")
    p.add_argument("--root", default="/home/vatty/piper_train_s")
    p.add_argument("--model", default="/mnt/d/vatty/work/Policy-IK/models/piper/scene.xml")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu avoids contending with a running training")
    args = p.parse_args()

    ex = ACTtoJoint(args.ckpt, device=args.device, model_path=args.model)
    ds = _load_dataset(args.root)
    print(f"dataset frames: {len(ds)}; testing {args.n} random frames")

    rng = random.Random(args.seed)
    idx = rng.sample(range(len(ds)), args.n)
    act_pos, act_rot, ik_pos, ik_rot, jok = [], [], [], [], 0
    for i in idx:
        s = ds[i]
        obs = {k: v for k, v in s.items() if k.startswith("observation.")}
        q_cur = s["observation.state"].numpy()[:6]
        ex.reset()
        q, grip, (pos, R), (ikp, ikr) = ex.act(obs, q_cur)
        gt = s["action"].numpy()
        act_pos.append(np.linalg.norm(pos - gt[:3]))
        Rg = rot6d_to_mat(gt[3:9])
        act_rot.append(np.linalg.norm(rotvec_from_mat(Rg @ R.T)))
        ik_pos.append(ikp)
        ik_rot.append(ikr)
        jok += int(np.all(np.isfinite(q)))

    med = lambda x: float(np.median(x))
    print("\n=== Step 5 offline test (ACT -> EE-pose -> IK -> Piper joints) ===")
    print(f"  1. ACT task-space error vs GT : {1e3*med(act_pos):6.1f} mm | {np.degrees(med(act_rot)):5.2f} deg  (median)")
    print(f"  2. IK executor residual       : {1e3*med(ik_pos):6.3f} mm | {np.degrees(med(ik_rot)):5.3f} deg  (median)")
    print(f"  3. valid joint refs produced  : {jok}/{args.n}")


if __name__ == "__main__":
    main()
