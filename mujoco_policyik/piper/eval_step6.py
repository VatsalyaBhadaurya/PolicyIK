"""Step 6: ACT-joint (direct) vs ACT-EE-pose + IK, on held-out Piper frames.

Both policies see identical observations (images + joint state); only the action
representation differs. We score each against the ground-truth joints in
joint-space and against the ground-truth end-effector pose in task-space (via FK)
-- the paper's question: does acting through IK give better task-space fidelity?

Run in WSL (lerobot venv):
    PYTHONPATH=/mnt/d/vatty/work/Policy-IK python -m mujoco_policyik.piper.eval_step6 \
        --ee_ckpt .../act_piper_eepose/checkpoints/last/pretrained_model \
        --jt_ckpt .../act_piper_joint/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/vatty/work/Policy-IK")

from mujoco_policyik.piper.act_ik_executor import ACTtoJoint
from mujoco_policyik.piper.piper_fk import PiperFK
from mujoco_policyik.piper.piper_ik import rotvec_from_mat


def _ds(root):
    for mod in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
        try:
            return importlib.import_module(mod).LeRobotDataset("aiengineer56/piper_dataset_1100", root=root)
        except Exception:
            continue
    raise RuntimeError("no LeRobotDataset")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ee_ckpt", default="/home/vatty/act_piper_eepose/checkpoints/last/pretrained_model")
    p.add_argument("--jt_ckpt", default="/home/vatty/act_piper_joint/checkpoints/last/pretrained_model")
    p.add_argument("--ee_root", default="/home/vatty/piper_train_s")
    p.add_argument("--jt_root", default="/home/vatty/piper_train_joint")
    p.add_argument("--model", default="/mnt/d/vatty/work/Policy-IK/models/piper/scene.xml")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    ee = ACTtoJoint(args.ee_ckpt, device=args.device, model_path=args.model)
    jt = ACTtoJoint(args.jt_ckpt, device=args.device, model_path=args.model)
    ee_ds, jt_ds = _ds(args.ee_root), _ds(args.jt_root)
    fk = PiperFK(args.model)

    idx = random.Random(args.seed).sample(range(len(jt_ds)), args.n)
    m = {"ee_j": [], "ee_p": [], "ee_r": [], "jt_j": [], "jt_p": [], "jt_r": []}
    for i in idx:
        s = ee_ds[i]
        obs = {k: v for k, v in s.items() if k.startswith("observation.")}
        q_cur = s["observation.state"].numpy()[:6]
        q_gt = jt_ds[i]["action"].numpy()[:6]
        p_gt, R_gt = fk.fk(q_gt)

        ee.reset()
        q_ik, _, _, _ = ee.act(obs, q_cur)
        jt.reset()
        q_dir = jt.predict_pose(obs)[:6]

        for q, kp, kr, kj in ((q_ik, "ee_p", "ee_r", "ee_j"), (q_dir, "jt_p", "jt_r", "jt_j")):
            pos, R = fk.fk(q)
            m[kj].append(float(np.mean(np.abs(q - q_gt))))
            m[kp].append(float(np.linalg.norm(pos - p_gt)))
            m[kr].append(float(np.linalg.norm(rotvec_from_mat(R_gt @ R.T))))

    md = lambda k: float(np.median(m[k]))
    print(f"\n=== Step 6: ACT-joint vs ACT-EE-pose+IK on {args.n} Piper frames (median) ===")
    print("  path                | joint MAE (rad) | task pos (mm) | task rot (deg)")
    print("  " + "-" * 64)
    print(f"  ACT-joint (direct)  | {md('jt_j'):14.4f} | {1e3*md('jt_p'):12.1f} | {np.degrees(md('jt_r')):12.2f}")
    print(f"  ACT-EEpose + IK     | {md('ee_j'):14.4f} | {1e3*md('ee_p'):12.1f} | {np.degrees(md('ee_r')):12.2f}")


if __name__ == "__main__":
    main()
