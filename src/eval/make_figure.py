"""The headline figure: 60 s drift on test, four predictors.

Deliberately four bars and nothing else. The comparison that matters is
whether reading the IMU beats not reading it, and how much of the remaining
gap is heading.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "results" / "final_scoring.csv"
OUT = REPO_ROOT / "results" / "figures" / "drift_60s_test.png"

# (row key in the CSV, label, colour)
BARS = [
    ("constant_velocity_dr", "Constant-velocity DR\n(reads no IMU)", "#7f7f7f"),
    ("ins_dr", "Classical INS DR", "#9e9e9e"),
    ("model_final", "Our model\n(CNN + raw-gyro heading)", "#1f77b4"),
    ("model_final_true_heading", "Our model\n+ oracle heading", "#2ca02c"),
]


def main() -> int:
    df = pd.read_csv(SRC)
    d = df[(df.role == "test") & (df.duration_s == 60.0)]

    labels, means, errs = [], [], []
    for key, label, _ in BARS:
        sub = d[d.predictor == key]
        if sub.empty:
            continue
        labels.append(label)
        means.append(float(sub.crse_m.mean()))
        errs.append(float(sub.crse_m.std(ddof=1) / np.sqrt(len(sub))))
    colours = [c for k, _, c in BARS if not d[d.predictor == k].empty]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=colours,
                  edgecolor="black", linewidth=0.6, width=0.62)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(errs) * 1.3,
                f"{m:.0f} m", ha="center", va="bottom", fontweight="bold")

    best_baseline = min(means[:2]) if len(means) >= 2 else None
    if best_baseline is not None:
        ax.axhline(best_baseline, color="#d62728", ls="--", lw=1.2,
                   label=f"best baseline ({best_baseline:.0f} m)")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean position drift after 60 s outage (m)")
    ax.set_title("GNSS outage drift, test split (4 sessions, "
                 f"{len(d[d.predictor == BARS[0][0]])} outages per predictor)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    for l, m, e in zip(labels, means, errs):
        print(f"  {l.splitlines()[0]:<28}{m:8.1f} +- {e:.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
