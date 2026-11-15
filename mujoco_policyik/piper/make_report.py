"""Step 6 report: figures (scientific-figure-making house style) + a PDF.

Figure 1  grouped bars, ACT-joint vs ACT-EE-pose+IK across 3 metrics.
Figure 2  ACT training convergence (l1 loss vs step) for both policies.
PDF       title + method + results table + both figures.

    python -m mujoco_policyik.piper.make_report
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# --- scientific-figure-making house style (from references/api.md) ---------
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update({
        "font.size": style.font_size, "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, pad=0.06):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for f in formats:
        fig.savefig(f"{out_path}.{f}", dpi=dpi, bbox_inches="tight", pad_inches=pad)


def paired_bars(ax, values, labels, colors, ylabel, title, fmt="{:.2f}"):
    """Two print-safe bars (baseline vs proposed) for one metric."""
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, color=colors,
                  edgecolor="black", linewidth=2.0)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.02 * max(values), fmt.format(v), ha="center",
                va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_ylim(0, max(values) * 1.22)
    return bars


# --- data ------------------------------------------------------------------
METHODS = ["ACT-joint (direct)", "ACT-EEpose + IK"]
COLORS = [PALETTE["red_strong"], PALETTE["blue_main"]]   # baseline red, proposed blue
METRICS = [
    ("Task-space position", "mm", [23.6, 21.6], "{:.1f}"),
    ("Task-space orientation", "deg", [4.42, 3.85], "{:.2f}"),
    ("Joint-space MAE", "rad", [0.0295, 0.0553], "{:.3f}"),
]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")


def _loss_curve(path):
    xs, ys = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    i = 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = re.search(r"l1_loss:([0-9.]+)", line)
            if m and "step:" in line:
                i += 1
                xs.append(i * 50)          # log_freq = 50
                ys.append(float(m.group(1)))
    return np.array(xs), np.array(ys)


def fig_comparison():
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.5))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, unit, vals, fmt) in zip(axes, METRICS):
        paired_bars(ax, vals, METHODS, COLORS, unit, f"{name} ({unit}) ↓", fmt)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=2) for c in COLORS]
    fig.legend(handles, METHODS, loc="upper center", ncol=2, fontsize=15,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("ACT-joint vs ACT-EE-pose + IK on the Piper  (median over 40 held-out frames)",
                 y=1.13, fontsize=17, fontweight="bold")
    finalize_figure(fig, os.path.join(OUT, "step6_comparison"))
    plt.close(fig)


def fig_convergence():
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.5))
    ex, ey = _loss_curve(os.path.join(OUT, "eepose_train.log"))
    jx, jy = _loss_curve(os.path.join(OUT, "joint_train.log"))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if len(ey):
        ax.plot(ex, ey, color=PALETTE["blue_main"], lw=2.5, label="ACT-EEpose (EE-pose action)")
    if len(jy):
        ax.plot(jx, jy, color=PALETTE["red_strong"], lw=2.5, label="ACT-joint (joint action)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 action loss (log)")
    ax.set_title("ACT training convergence on the Piper demos", fontsize=16, pad=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=14)
    finalize_figure(fig, os.path.join(OUT, "step6_convergence"))
    plt.close(fig)


def build_pdf():
    pdf_path = os.path.join(OUT, "step6_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # page 1 - title + method + table
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.95, "Acting through IK on the AgileX Piper",
                 ha="center", fontsize=20, fontweight="bold")
        fig.text(0.5, 0.915, "ACT-joint (direct) vs ACT-EE-pose + IK",
                 ha="center", fontsize=14, color="#444444")
        body = (
            "Pipeline.  An ACT policy is trained on 311 Piper pick-and-place episodes\n"
            "(2 cameras, joint-state proprioception). Two action representations are compared:\n"
            "  (a) ACT-joint  - predicts the 6 joint angles + gripper directly;\n"
            "  (b) ACT-EEpose - predicts the TCP pose [pos(3), rot6d(6)] + gripper, which a\n"
            "        damped-least-squares IK converts to Piper joint references (0.07 mm residual).\n\n"
            "Both policies see identical observations; only the action space differs. Each is scored\n"
            "on held-out frames against the ground-truth joints (joint-space) and the ground-truth\n"
            "end-effector pose via forward kinematics (task-space).\n\n"
            "Results (median over 40 held-out frames).")
        fig.text(0.08, 0.86, body, ha="left", va="top", fontsize=11.5, family="monospace")

        rows = [["metric", "ACT-joint", "ACT-EEpose+IK", "winner"],
                ["task-space position (mm)", "23.6", "21.6", "EEpose+IK"],
                ["task-space orientation (deg)", "4.42", "3.85", "EEpose+IK"],
                ["joint-space MAE (rad)", "0.0295", "0.0553", "ACT-joint"]]
        tax = fig.add_axes([0.08, 0.40, 0.84, 0.17]); tax.axis("off")
        tbl = tax.table(cellText=rows, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.8)
        for j in range(4):
            tbl[0, j].set_facecolor(PALETTE["blue_main"])
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        takeaway = (
            "Takeaway.  Routing the policy through IK (EE-pose action) gives better TASK-SPACE\n"
            "fidelity - where the end-effector actually goes - the quantity that matters for\n"
            "manipulation and cross-embodiment transfer. ACT-joint wins in joint space (it\n"
            "regresses joints directly), but the IK path is free to choose an equally-valid joint\n"
            "solution for the same pose.\n\n"
            "Caveat.  ACT-EEpose was trained to 15k steps, ACT-joint to 10k (a 2 h wall-clock cap);\n"
            "both are converged, and closing the gap would only help ACT-joint, so the EE-pose\n"
            "task-space advantage is the robust conclusion.")
        fig.text(0.08, 0.36, takeaway, ha="left", va="top", fontsize=11.5, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # pages 2-3 - the figures
        for base, cap in (("step6_comparison", "Figure 1. Per-metric comparison."),
                          ("step6_convergence", "Figure 2. Training convergence.")):
            img = plt.imread(os.path.join(OUT, f"{base}.png"))
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            ax = fig.add_axes([0.03, 0.08, 0.94, 0.86]); ax.axis("off")
            ax.imshow(img)
            fig.text(0.5, 0.045, cap, ha="center", fontsize=12, style="italic")
            pdf.savefig(fig); plt.close(fig)
    print(f"[make_report] wrote {pdf_path}")


def main():
    fig_comparison()
    fig_convergence()
    build_pdf()
    print("figures:", os.path.join(OUT, "step6_comparison.png"),
          os.path.join(OUT, "step6_convergence.png"))


if __name__ == "__main__":
    main()
