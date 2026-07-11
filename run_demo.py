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
    indeg = gp["dst_in_deg_prior"].mean()
    spike = gp["amt_to_dst_mean_prior"].mean()
    print("  mean receiver in-degree-so-far   legit vs laundering: "
          f"{indeg.get(0, float('nan')):.2f} vs {indeg.get(1, float('nan')):.2f}")
    print("  mean amount vs receiver's usual   legit vs laundering: "
          f"{spike.get(0, float('nan')):.2f} vs {spike.get(1, float('nan')):.2f}\n")

    if "pattern" in df.columns:
        s = graph_features.structural_flags(df)
        cols3 = ["pattern", "is_laundering"]
        laund = pd.concat([
            df[["from_account", *cols3]].rename(columns={"from_account": "account"}),
            df[["to_account", *cols3]].rename(columns={"to_account": "account"}),
        ])
        laund = laund[laund["is_laundering"] == 1]
        dom = laund.groupby("account")["pattern"].agg(lambda x: x.value_counts().index[0])
        flag = dict(zip(s["account"], s["flagged"], strict=False))
        print("Structural flags catch some typologies and miss others (the honest finding):")
        by_pat: dict[str, list[int]] = {}
        for acct, pat in dom.items():
            by_pat.setdefault(pat, [0, 0])
            by_pat[pat][0] += int(flag.get(acct, 0))
            by_pat[pat][1] += 1
        for pat, (c, t) in sorted(by_pat.items()):
            print(f"  {pat:16s} {c:4d}/{t:<4d} = {c / t:.0%}")
        print("  fan-in and fan-out endpoints look like ordinary accounts by degree "
              "alone; they need burst-rate features (next).\n")

    nets = graph_features.candidate_networks(df)
    tight = nets[nets["laundering_share"] >= 0.9]
    print(f"Candidate rings surfaced (weakly connected components of flagged accounts): "
          f"{len(nets)}")
    print(f"  {len(tight)} are >=90% laundering; the model's job is to raise recall on "
          "the confounded typologies.")


if __name__ == "__main__":
    main()
