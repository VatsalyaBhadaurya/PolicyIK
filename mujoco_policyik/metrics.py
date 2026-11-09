"""Success metrics following the PolicyIK evaluation protocol.

A task succeeds when the *mean* position and orientation errors over the final
``H`` control steps (1 s at 30 Hz) fall below the tolerance:

* strict: 1 mm and 0.2 deg  (Equation 1)
* coarse: 1 cm and 2 deg    (the relaxed diagnostic)
"""

from __future__ import annotations

import numpy as np

from . import TOL_ORI_RAD, TOL_POS_M

FINAL_WINDOW = 30  # H = 30 steps = 1 s at 30 Hz


def terminal_errors(e_pos: np.ndarray, e_ori: np.ndarray, window: int = FINAL_WINDOW):
    """Mean position (m) and orientation (rad) error over the final ``window`` steps."""
    return float(np.mean(e_pos[-window:])), float(np.mean(e_ori[-window:]))


def success(e_pos, e_ori, window: int = FINAL_WINDOW, coarse: bool = False) -> bool:
    mp, mo = terminal_errors(e_pos, e_ori, window)
    pos_tol = 1e-2 if coarse else TOL_POS_M
    ori_tol = np.deg2rad(2.0) if coarse else TOL_ORI_RAD
    return bool(mp <= pos_tol and mo <= ori_tol)


def summarize(results: list[dict]) -> dict:
    """Aggregate a list of per-task {'e_pos','e_ori'} traces into headline metrics."""
    mp = np.array([terminal_errors(r["e_pos"], r["e_ori"])[0] for r in results])
    mo = np.array([terminal_errors(r["e_pos"], r["e_ori"])[1] for r in results])
    strict = np.array([success(r["e_pos"], r["e_ori"]) for r in results])
    coarse = np.array([success(r["e_pos"], r["e_ori"], coarse=True) for r in results])
    return {
        "n": len(results),
        "strict_pct": 100.0 * strict.mean(),
        "coarse_pct": 100.0 * coarse.mean(),
        "median_pos_mm": 1e3 * float(np.median(mp)),
        "median_ori_deg": float(np.degrees(np.median(mo))),
    }
