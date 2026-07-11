"""Train the mule-scoring model and report it honestly.

Runs on the real IBM-AML data if present at data/aml/HI-Small_Trans.csv, else on the
schema-faithful mock. Reports transaction PR-AUC, precision/recall and value recovered
under an investigation budget, recall per typology, and the top of the account queue.

    python run_model.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml import aml_data, metrics, model, scoring
from aml.graph_features import prior_features


def load() -> pd.DataFrame:
    real = Path("data/aml/HI-Small_Trans.csv")
    if real.exists():
        print("Loading real IBM-AML from data/aml/ ...")
        return aml_data.load_aml(real)
    print("Real IBM-AML not found; using the schema-faithful mock (numbers illustrative).")
    return aml_data.load_aml_frame(aml_data.mock_aml_frames(seed=7))


def main() -> None:
    df = load()
    feat, cols = prior_features(df)
    train_df, valid_df, test_df = model.time_split(feat)
    print(f"  train {len(train_df):,} | valid {len(valid_df):,} | test {len(test_df):,} "
          f"transfers; test laundering rate {test_df['is_laundering'].mean():.3%}\n")

    clf = model.train_model(train_df, cols)
    test_scores = model.score(clf, test_df, cols)

    ap = metrics.pr_auc(test_df["is_laundering"].to_numpy(), test_scores)
    base = test_df["is_laundering"].mean()
    print(f"Transaction PR-AUC on the held-out tail: {ap:.3f}  (base rate {base:.3%})")

    budget = int(0.02 * len(test_df))
    b = metrics.at_budget(test_df, test_scores, budget)
    print(f"At a {budget:,}-alert budget: precision {b['precision']:.3f}, "
          f"recall {b['recall']:.3f}, value recovered {b['value_recall']:.1%}\n")

    typ = metrics.per_typology_recall(test_df, test_scores, budget)
    if not typ.empty:
        print("Recall within budget, per typology:")
        for r in typ.itertuples(index=False):
            print(f"  {r.pattern:16s} {int(r.sum):4d}/{int(r.count):<4d} = {r.recall:.0%}")
        print()

    # account queue is built over the test window the investigator would be working
    queue = scoring.account_queue(test_df, test_scores, top=10)
    hit = int(queue["is_laundering_acct"].sum())
    print(f"Top of the account queue ({hit}/10 are laundering accounts):")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(queue[["priority", "value_moved", "is_laundering_acct", "reasons"]]
              .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
