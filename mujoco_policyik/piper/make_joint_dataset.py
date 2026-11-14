"""Build the joint-action variant of the downscaled Piper dataset for Step 6.

Same episodes / downscaled videos / joint observation as `piper_train_s`, but the
`action` is the original [joint1-6, gripper] (from the untransformed dataset) --
so an ACT trained here outputs joints directly, the baseline to compare against
ACT-EE-pose + IK. Videos are shared (symlinked); only the action column differs.

    python -m mujoco_policyik.piper.make_joint_dataset \
        --ee_root ~/piper_train_s --orig /mnt/d/.../piper_data --dst ~/piper_train_joint
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ee_root", default=os.path.expanduser("~/piper_train_s"))
    p.add_argument("--orig", default="/mnt/d/vatty/work/Policy-IK/piper_data")
    p.add_argument("--dst", default=os.path.expanduser("~/piper_train_joint"))
    args = p.parse_args()

    os.makedirs(os.path.join(args.dst, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "meta"), exist_ok=True)

    # data: take the EE dataset rows, swap action for the original joint action
    ee_pq = sorted(glob.glob(os.path.join(args.ee_root, "data", "**", "*.parquet"), recursive=True))[0]
    orig_pq = sorted(glob.glob(os.path.join(args.orig, "data", "**", "*.parquet"), recursive=True))[0]
    df = pd.read_parquet(ee_pq)
    orig = pd.read_parquet(orig_pq)
    n = len(df)
    joints = np.stack(orig["action"].values[:n]).astype(np.float32)   # (n,7)
    assert joints.shape[1] == 7, joints.shape
    df["action"] = list(joints)
    rel = os.path.relpath(ee_pq, args.ee_root)
    os.makedirs(os.path.dirname(os.path.join(args.dst, rel)), exist_ok=True)
    df.to_parquet(os.path.join(args.dst, rel), index=False)

    # meta: copy episodes + tasks, update info.json action feature, recompute stats
    shutil.copytree(os.path.join(args.ee_root, "meta", "episodes"),
                    os.path.join(args.dst, "meta", "episodes"), dirs_exist_ok=True)
    for f in ("tasks.parquet",):
        s = os.path.join(args.ee_root, "meta", f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.dst, "meta", f))

    info = json.load(open(os.path.join(args.ee_root, "meta", "info.json")))
    info["features"]["action"] = {"dtype": "float32", "shape": [7],
                                  "names": JOINT_NAMES, "fps": info["fps"]}
    json.dump(info, open(os.path.join(args.dst, "meta", "info.json"), "w"), indent=1)

    stats = json.load(open(os.path.join(args.ee_root, "meta", "stats.json")))
    stats["action"] = {"min": joints.min(0).tolist(), "max": joints.max(0).tolist(),
                       "mean": joints.mean(0).tolist(), "std": (joints.std(0) + 1e-8).tolist(),
                       "count": [int(n)]}
    json.dump(stats, open(os.path.join(args.dst, "meta", "stats.json"), "w"), indent=1)

    # videos: share the downscaled ones
    dv = os.path.join(args.dst, "videos")
    if not os.path.exists(dv):
        os.symlink(os.path.abspath(os.path.join(args.ee_root, "videos")), dv)

    print(f"[make_joint_dataset] {n} frames -> {args.dst}  (action=joints[7], videos shared)")


if __name__ == "__main__":
    main()
