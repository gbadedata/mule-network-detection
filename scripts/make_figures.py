"""Generate the figures for the writeup.

Trains on the time split and scores the held-out tail, then draws two figures into
docs/img/: the laundered value recovered as the review budget grows (the operating
result), and recall per typology at a fixed budget (where the model is strong and where
it is not). Runs on the real IBM-AML data if present, else on the mock.

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aml import aml_data, metrics, model  # noqa: E402
from aml.graph_features import prior_features  # noqa: E402

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"
INK = "#1a1a1a"
MODEL_C = "#1f5f8b"
RAND_C = "#b0b0b0"


def load():
    real = Path("data/aml/HI-Small_Trans.csv")
    if real.exists():
        return aml_data.load_aml(real), "real IBM-AML"
    return aml_data.load_aml_frame(aml_data.mock_aml_frames(seed=7)), "mock (illustrative)"


def value_vs_budget(test_df, scores, tag):
    amt = test_df["amount"].to_numpy(float)
    y = test_df["is_laundering"].to_numpy(int)
    order = np.argsort(-scores)
    recovered = np.cumsum(amt[order] * y[order])
    total = amt[y == 1].sum()
    x = np.arange(1, len(order) + 1)
    rand = total * x / len(order)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, recovered / total * 100, color=MODEL_C, lw=2, label="model, ranked by score")
    ax.plot(x, rand / total * 100, color=RAND_C, lw=1.5, ls="--", label="random review order")
    half = int(np.searchsorted(recovered, total * 0.5)) + 1
    ax.axvline(half, color=INK, lw=0.8, alpha=0.5)
    ax.annotate(f"half the laundered value\nin the top {half:,} reviewed",
                xy=(half, 50), xytext=(half + len(x) * 0.05, 32),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.set_xlabel("transfers reviewed (highest score first)")
    ax.set_ylabel("laundered value recovered (%)")
    ax.set_title(f"Laundered value recovered vs review budget ({tag})", color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = IMG / "value_vs_budget.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def typology_recall(test_df, scores, tag):
    budget = int(0.02 * len(test_df))
    typ = metrics.per_typology_recall(test_df, scores, budget)
    if typ.empty:
        return None
    typ = typ.sort_values("recall")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(typ["pattern"], typ["recall"] * 100, color=MODEL_C)
    for yi, v in enumerate(typ["recall"] * 100):
        ax.text(min(v + 1.5, 98), yi, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("recall within a 2% alert budget")
    ax.set_title(f"Detection recall by laundering typology ({tag})", color=INK)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    out = IMG / "typology_recall.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    df, tag = load()
    feat, cols = prior_features(df)
    train_df, _, test_df = model.time_split(feat)
    clf = model.train_model(train_df, cols)
    scores = model.score(clf, test_df, cols)

    a = value_vs_budget(test_df, scores, tag)
    b = typology_recall(test_df, scores, tag)
    print(f"wrote {a}")
    if b:
        print(f"wrote {b}")


if __name__ == "__main__":
    main()
