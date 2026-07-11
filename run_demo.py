"""End-to-end mule-network demo: graph, leakage-safe features, network surfacing.

Runs on the real IBM-AML data if present at data/aml/HI-Small_Trans.csv, else on a
schema-faithful mock so it works with no download.

    python run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml import aml_data, graph_features


def load() -> pd.DataFrame:
    real = Path("data/aml/HI-Small_Trans.csv")
    if real.exists():
        print("Loading real IBM-AML from data/aml/ ...")
        return aml_data.load_aml(real)
    print("Real IBM-AML not found; using the schema-faithful mock.")
    return aml_data.load_aml_frame(aml_data.mock_aml_frames(seed=7))


def main() -> None:
    df = load()
    n_acct = len({*df["from_account"], *df["to_account"]})
    print(f"  {len(df):,} transfers | {n_acct:,} accounts | "
          f"laundering rate {df['is_laundering'].mean():.3%} | "
          f"laundering value ${df.loc[df.is_laundering == 1, 'amount'].sum():,.0f}\n")

    feat, cols = graph_features.prior_features(df)
    print(f"Leakage-safe streaming features: {len(cols)} (no future information).")
    gp = feat.groupby("is_laundering")
    for name, col in [("in-burst (fan-in)", "dst_in_burst"),
                      ("out-burst (fan-out)", "src_out_burst"),
                      ("amount vs usual (spike)", "amt_to_dst_mean_prior")]:
        m = gp[col].mean()
        print(f"  {name:24s} legit {m.get(0, float('nan')):.2f} vs "
              f"laundering {m.get(1, float('nan')):.2f}")
    print()

    if "pattern" in df.columns:
        s = graph_features.structural_flags(df)
        flag = dict(zip(s["account"], s["flagged"], strict=False))
        collectors = set(df.loc[df["pattern"] == "fan_in", "to_account"])
        distributors = set(df.loc[df["pattern"] == "fan_out", "from_account"])
        c_hit = sum(flag.get(a, 0) for a in collectors)
        d_hit = sum(flag.get(a, 0) for a in distributors)
        print("Burst features detect the hub of every star ring:")
        print(f"  fan-in collectors caught    {c_hit}/{len(collectors)}")
        print(f"  fan-out distributors caught {d_hit}/{len(distributors)}")
        print("  the one-shot senders around a hub have no signature alone; they are "
              "recovered\n  by expanding the hub's ego network.\n")

    nets = graph_features.candidate_networks(df)
    tot = int(graph_features.account_summary(df)["is_laundering_acct"].sum())
    caught = int(nets["laundering_accounts"].sum())
    rings = nets[nets["accounts"] < 100]
    clean = rings[rings["laundering_share"] >= 0.8]
    print(f"Candidate networks surfaced: {len(nets)} "
          f"({caught}/{tot} laundering accounts inside them).")
    print(f"  {len(clean)} are small rings that are >=80% laundering.")
    print("  Single structural thresholds are high-recall but low-precision here: no one "
          "rule\n  separates every typology. Combining these features in a supervised, "
          "typology-aware\n  model is the next step, and where precision comes from.")


if __name__ == "__main__":
    main()
