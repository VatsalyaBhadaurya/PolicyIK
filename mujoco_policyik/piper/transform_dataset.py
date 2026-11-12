"""Transform a Piper LeRobot dataset: joint-angle action -> EE-pose action.

Reads a LeRobot v3 dataset whose `action` is [joint1-6, gripper] and writes a new
dataset whose `action` is [ee_pos(3), ee_rot6d(6), gripper(1)] (10-dim, absolute
TCP pose in the base frame, via Piper FK). `observation.state` (joints) and the
videos are kept unchanged. This is Step 3 of the ACT -> EE-pose -> PolicyIK pipeline.

    python -m mujoco_policyik.piper.transform_dataset --src piper_data --dst piper_eepose
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from .piper_fk import PiperFK

ACTION_NAMES = ["ee_x", "ee_y", "ee_z",
                "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "rot6d_6",
                "gripper"]


def transform_action(joints_action: np.ndarray, fk: PiperFK) -> np.ndarray:
    """[N,7] joint+gripper action -> [N,10] ee_pos + rot6d + gripper."""
    pos, r6 = fk.fk_batch(joints_action[:, :6].astype(np.float64))
    grip = joints_action[:, 6:7]
    return np.concatenate([pos, r6, grip], axis=1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="piper_data")
    p.add_argument("--dst", default="piper_eepose")
    p.add_argument("--link-videos", action="store_true", help="symlink videos/ from src (for training)")
    args = p.parse_args()

    fk = PiperFK()
    os.makedirs(args.dst, exist_ok=True)

    # --- data parquets: replace the action column ---
    all_actions = []
    srcs = sorted(glob.glob(os.path.join(args.src, "data", "**", "*.parquet"), recursive=True))
    for sp in srcs:
        rel = os.path.relpath(sp, args.src)
        dp = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        df = pd.read_parquet(sp)
        A = np.stack(df["action"].values)
        new_A = transform_action(A, fk)
        df["action"] = list(new_A)
        df.to_parquet(dp, index=False)
        all_actions.append(new_A)
        print(f"  {rel}: {len(df)} frames -> action {A.shape[1]}d -> {new_A.shape[1]}d")
    actions = np.concatenate(all_actions, axis=0)

    # --- meta/info.json: update the action feature ---
    info = json.load(open(os.path.join(args.src, "meta", "info.json")))
    info["features"]["action"] = {"dtype": "float32", "shape": [10],
                                  "names": ACTION_NAMES, "fps": info["fps"]}
    os.makedirs(os.path.join(args.dst, "meta"), exist_ok=True)
    json.dump(info, open(os.path.join(args.dst, "meta", "info.json"), "w"), indent=1)

    # --- meta/stats.json: recompute action stats ---
    stats = json.load(open(os.path.join(args.src, "meta", "stats.json")))
    stats["action"] = {
        "min": actions.min(0).tolist(), "max": actions.max(0).tolist(),
        "mean": actions.mean(0).tolist(), "std": (actions.std(0) + 1e-8).tolist(),
        "count": [int(len(actions))],
    }
    json.dump(stats, open(os.path.join(args.dst, "meta", "stats.json"), "w"), indent=1)

    # --- copy remaining meta (episodes, tasks) ---
    for sub in ("episodes", "tasks.parquet"):
        s = os.path.join(args.src, "meta", sub)
        d = os.path.join(args.dst, "meta", sub)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif os.path.exists(s):
            shutil.copy2(s, d)

    # --- videos: optional symlink (kept out of the transform) ---
    if args.link_videos:
        sv = os.path.abspath(os.path.join(args.src, "videos"))
        if os.path.isdir(sv):
            dv = os.path.join(args.dst, "videos")
            if not os.path.exists(dv):
                os.symlink(sv, dv)

    print(f"[transform] wrote EE-pose dataset -> {args.dst}  ({len(actions)} frames)")
    print(f"  action = [ee_pos(3), ee_rot6d(6), gripper(1)]; videos {'linked' if args.link_videos else 'NOT copied'}")


if __name__ == "__main__":
    main()
