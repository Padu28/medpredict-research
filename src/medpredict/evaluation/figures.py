"""Report figures: reliability diagrams, subgroup forest plot, shift tests."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from medpredict.storage.database import Database  # noqa: E402

CAL = {"uncalibrated": "#94a3b8", "sigmoid": "#38bdf8", "isotonic": "#f472b6"}


def make_figures(db: Database, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    cc = db.query_df("SELECT * FROM calibration_curves")
    models = cc.model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4), sharey=True)
    for ax, m in zip(axes, models):
        for r in cc[cc.model == m].itertuples():
            ax.plot(json.loads(r.predicted), json.loads(r.observed), "o-", color=CAL[r.calibration], label=r.calibration, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="k", lw=0.8)
        ax.set(title=m, xlabel="mean predicted risk", ylabel="observed rate", xlim=(0, 0.8), ylim=(0, 0.8))
    axes[0].legend(frameon=False)
    fig.suptitle("Reliability diagrams (test period)")
    fig.tight_layout()
    fig.savefig(out / "reliability_diagrams.png")
    plt.close(fig)

    sub = db.query_df("SELECT * FROM subgroups WHERE suppressed = 0")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(sub) + 1.2))
    labels = [f"{g} · {v} (n={n})" for g, v, n in zip(sub.grouping, sub.group, sub.n)]
    ax.errorbar(sub.auroc, range(len(sub)), xerr=[sub.auroc - sub.auroc_lo, sub.auroc_hi - sub.auroc], fmt="o",
                color="#38bdf8", ecolor="#94a3b8", capsize=3)
    ax.set_yticks(range(len(sub)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="AUROC (95% bootstrap CI)", title="Subgroup discrimination (test)")
    fig.tight_layout()
    fig.savefig(out / "subgroup_forest.png")
    plt.close(fig)

    sh = db.query_df("SELECT * FROM shift_results")
    miss = sh[sh.test == "missingness"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(miss.level, miss.auroc, "o-", color="#38bdf8")
    ax[0].set(title="AUROC vs injected lab missingness", ylabel="AUROC")
    other = sh[sh.test != "missingness"]
    ax[1].barh([f"{t}: {lv}" for t, lv in zip(other.test, other.level)], other.ece, color="#f472b6")
    ax[1].set(title="Calibration error (ECE) under shift", xlabel="ECE")
    fig.tight_layout()
    fig.savefig(out / "shift_tests.png")
    plt.close(fig)
