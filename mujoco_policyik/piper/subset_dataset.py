"""Subset a (transformed) LeRobot v3 dataset to the first N episodes.

Takes a prefix of episodes so frame indices stay contiguous (no reindexing), and
recomputes action/state stats over the subset. Used to match the partial video
download (both cams, <=5 GB) for 4050-friendly ACT training.

    python -m mujoco_policyik.piper.subset_dataset --src piper_eepose --dst piper_train --episodes 311
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd


def _recompute_stats(stats, arr, key):
    stats[key] = {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
                  "mean": arr.mean(0).tolist(), "std": (arr.std(0) + 1e-8).tolist(),
                  "count": [int(len(arr))]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="piper_eepose")
    p.add_argument("--dst", default="piper_train")
    p.add_argument("--episodes", type=int, default=311)
    p.add_argument("--videos-from", default="piper_data", help="dir holding the downloaded videos/")
    args = p.parse_args()

    n = args.episodes
    os.makedirs(os.path.join(args.dst, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "meta"), exist_ok=True)

    # --- data parquet: keep episode_index < n (a contiguous prefix) ---
    sp = sorted(glob.glob(os.path.join(args.src, "data", "**", "*.parquet"), recursive=True))[0]
    df = pd.read_parquet(sp)
    sub = df[df["episode_index"] < n].reset_index(drop=True)
    rel = os.path.relpath(sp, args.src)
    os.makedirs(os.path.dirname(os.path.join(args.dst, rel)), exist_ok=True)
    sub.to_parquet(os.path.join(args.dst, rel), index=False)

    # --- episodes meta: first n rows ---
    ep_src = sorted(glob.glob(os.path.join(args.src, "meta", "episodes", "**", "*.parquet"), recursive=True))[0]
    ep = pd.read_parquet(ep_src)
    ep_sub = ep[ep["episode_index"] < n].reset_index(drop=True)
    ep_rel = os.path.relpath(ep_src, args.src)
    os.makedirs(os.path.dirname(os.path.join(args.dst, ep_rel)), exist_ok=True)
    ep_sub.to_parquet(os.path.join(args.dst, ep_rel), index=False)

    # --- info.json ---
    info = json.load(open(os.path.join(args.src, "meta", "info.json")))
    info["total_episodes"] = n
    info["total_frames"] = int(len(sub))
    info["splits"] = {"train": f"0:{n}"}
    json.dump(info, open(os.path.join(args.dst, "meta", "info.json"), "w"), indent=1)

    # --- stats.json: recompute action + state over the subset ---
    stats = json.load(open(os.path.join(args.src, "meta", "stats.json")))
    _recompute_stats(stats, np.stack(sub["action"].values), "action")
    _recompute_stats(stats, np.stack(sub["observation.state"].values), "observation.state")
    json.dump(stats, open(os.path.join(args.dst, "meta", "stats.json"), "w"), indent=1)

    # --- tasks ---
    for t in ("tasks.parquet",):
        s = os.path.join(args.src, "meta", t)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.dst, "meta", t))

    # --- videos: symlink the downloaded tree ---
    sv = os.path.abspath(os.path.join(args.videos_from, "videos"))
    dv = os.path.join(args.dst, "videos")
    if os.path.isdir(sv) and not os.path.exists(dv):
        try:
            os.symlink(sv, dv)
            link = "symlinked"
        except OSError:
            shutil.copytree(sv, dv)
            link = "copied"
    else:
        link = "already present" if os.path.exists(dv) else "MISSING (download not done yet)"

    print(f"[subset] {n} episodes, {len(sub)} frames -> {args.dst}  (videos: {link})")


if __name__ == "__main__":
    main()
