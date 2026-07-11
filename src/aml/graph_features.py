"""Graph construction and features for money-mule network detection.

Two kinds of feature, kept deliberately separate, because they have different rules
and different deployment stories:

1. Streaming features (`prior_features`) describe an account's history *strictly
   before* the current transfer: how many distinct counterparties it has had, how much
   it has moved, and its pass-through ratio (money forwarded out over money received
   in). These use no future information, so they are what a real-time monitor could
   actually score on. This is the graph version of the strictly-before discipline.

2. Retrospective features (`account_summary`) describe an account over a whole window,
   for an investigator triaging surfaced networks after the fact. They are not
   leakage-safe for real-time scoring and are not used as model inputs; they support
   the investigation view and the honest evaluation.

`candidate_networks` surfaces groups to investigate. A real transaction graph is one
giant weakly connected component, so components of the raw graph are useless. We first
filter to pass-through-heavy accounts (the mule signature) and take components of that
induced subgraph, which isolates candidate rings.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "amount_log",
    "src_out_cnt_prior", "src_out_deg_prior", "src_amt_out_prior",
    "dst_in_cnt_prior", "dst_in_deg_prior", "dst_amt_in_prior",
    "dst_amt_out_prior", "dst_passthrough_prior", "amt_to_dst_mean_prior",
    "hour",
]


def build_account_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Directed multigraph: accounts are nodes, transfers are timestamped edges."""
    g = nx.MultiDiGraph()
    for r in df.itertuples(index=False):
        g.add_edge(r.from_account, r.to_account, ts=r.ts, amount=r.amount,
                   is_laundering=int(r.is_laundering))
    return g


def _prior_count(key: pd.Series) -> np.ndarray:
    return key.groupby(key).cumcount().to_numpy()


def _prior_distinct(key: pd.Series, value: pd.Series) -> np.ndarray:
    """Distinct `value` seen within `key` strictly before the current row."""
    frame = pd.DataFrame({"k": key.to_numpy(), "v": value.to_numpy()})
    first = ~frame.duplicated(["k", "v"])
    incl = first.groupby(frame["k"]).cumsum().to_numpy()
    return incl - first.to_numpy().astype(int)


def _prior_sum(key: pd.Series, value: np.ndarray) -> np.ndarray:
    s = pd.Series(value)
    return (s.groupby(key.to_numpy()).cumsum().to_numpy() - value)


def _prior_inflow_outflow(df: pd.DataFrame):
    """For each transfer, the receiver's prior total inflow and outflow.

    Built from a per-account event log so it captures both roles: an account both
    receives (inflow) and sends (outflow). Strictly-before via a stable sort.
    """
    n = len(df)
    amt = df["amount"].to_numpy(float)
    ev = pd.DataFrame({
        "acct": np.concatenate([df["to_account"].to_numpy(), df["from_account"].to_numpy()]),
        "ts": np.concatenate([df["ts"].to_numpy(), df["ts"].to_numpy()]),
        "inflow": np.concatenate([amt, np.zeros(n)]),
        "outflow": np.concatenate([np.zeros(n), amt]),
        "is_in": np.concatenate([np.ones(n, bool), np.zeros(n, bool)]),
        "orig": np.concatenate([np.arange(n), np.arange(n)]),
    }).sort_values(["acct", "ts"], kind="stable")
    g = ev.groupby("acct", sort=False)
    ev["cum_in_prior"] = g["inflow"].cumsum() - ev["inflow"]
    ev["cum_out_prior"] = g["outflow"].cumsum() - ev["outflow"]
    rec = ev[ev["is_in"]].set_index("orig")
    prior_in = rec["cum_in_prior"].reindex(range(n)).to_numpy()
    prior_out = rec["cum_out_prior"].reindex(range(n)).to_numpy()
    return prior_in, prior_out


def prior_features(df: pd.DataFrame):
    """Return (df_with_features, feature_cols): leakage-safe streaming features."""
    df = df.sort_values("ts").reset_index(drop=True)
    src, dst = df["from_account"], df["to_account"]
    amt = df["amount"].to_numpy(float)

    dst_in_cnt = _prior_count(dst)
    dst_amt_in = _prior_sum(dst, amt)
    dst_in_prior, dst_out_prior = _prior_inflow_outflow(df)

    feats = {
        "amount_log": np.log1p(amt),
        "src_out_cnt_prior": _prior_count(src),
        "src_out_deg_prior": _prior_distinct(src, dst),
        "src_amt_out_prior": _prior_sum(src, amt),
        "dst_in_cnt_prior": dst_in_cnt,
        "dst_in_deg_prior": _prior_distinct(dst, src),
        "dst_amt_in_prior": dst_amt_in,
        "dst_amt_out_prior": dst_out_prior,
        # pass-through: money the receiver has already forwarded per unit received.
        "dst_passthrough_prior": dst_out_prior / (dst_in_prior + 1.0),
        # is this amount a spike versus what the receiver usually takes in?
        "amt_to_dst_mean_prior": amt / (dst_amt_in / np.maximum(dst_in_cnt, 1) + 1.0),
        "hour": df["ts"].dt.hour.to_numpy(),
    }
    feat_df = pd.DataFrame(feats, index=df.index).fillna(0.0)
    out = pd.concat([df, feat_df], axis=1)
    return out, list(feat_df.columns)


def account_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retrospective per-account structure over the window (investigation + eval)."""
    out_g = df.groupby("from_account")
    in_g = df.groupby("to_account")
    accts = pd.Index(sorted(set(df["from_account"]) | set(df["to_account"])), name="account")

    s = pd.DataFrame(index=accts)
    s["out_cnt"] = out_g.size().reindex(accts).fillna(0)
    s["in_cnt"] = in_g.size().reindex(accts).fillna(0)
    s["distinct_out"] = out_g["to_account"].nunique().reindex(accts).fillna(0)
    s["distinct_in"] = in_g["from_account"].nunique().reindex(accts).fillna(0)
    s["amt_out"] = out_g["amount"].sum().reindex(accts).fillna(0.0)
    s["amt_in"] = in_g["amount"].sum().reindex(accts).fillna(0.0)
    hi = np.maximum(s["amt_in"], s["amt_out"])
    s["passthrough"] = np.minimum(s["amt_in"], s["amt_out"]) / hi.replace(0, np.nan)
    s["passthrough"] = s["passthrough"].fillna(0.0)
    s["net_flow"] = s["amt_in"] - s["amt_out"]

    laund = pd.concat([
        df[["from_account", "is_laundering"]].rename(columns={"from_account": "account"}),
        df[["to_account", "is_laundering"]].rename(columns={"to_account": "account"}),
    ])
    s["laundering_txns"] = laund.groupby("account")["is_laundering"].sum().reindex(accts).fillna(0)
    s["is_laundering_acct"] = (s["laundering_txns"] > 0).astype(int)
    return s.reset_index()


def structural_flags(df: pd.DataFrame, fan_pct: float = 98.0,
                     min_passthrough: float = 0.7) -> pd.DataFrame:
    """Flag accounts by typology-specific structure, since no single rule fits all.

    A collector shows fan-in (many distinct senders), a distributor shows fan-out
    (many distinct receivers), a layering or scatter-gather intermediary shows
    balanced pass-through. We take a union of these, using high percentiles for the
    hub-like counts. This is first-pass triage; legit hubs still slip through, which
    is what the supervised model is for.
    """
    s = account_summary(df)
    din, dout = s["distinct_in"].to_numpy(float), s["distinct_out"].to_numpy(float)
    in_hi = np.percentile(din[din > 0], fan_pct) if (din > 0).any() else np.inf
    out_hi = np.percentile(dout[dout > 0], fan_pct) if (dout > 0).any() else np.inf
    s["fan_in_flag"] = (din >= in_hi).astype(int)
    s["fan_out_flag"] = (dout >= out_hi).astype(int)
    s["passthrough_flag"] = ((s["passthrough"] >= min_passthrough)
                             & (din >= 1) & (dout >= 1)).astype(int)
    s["flagged"] = ((s["fan_in_flag"] + s["fan_out_flag"] + s["passthrough_flag"]) > 0).astype(int)
    return s


def candidate_networks(df: pd.DataFrame, fan_pct: float = 98.0,
                       min_passthrough: float = 0.7) -> pd.DataFrame:
    """Surface candidate mule networks as components of the flagged subgraph.

    The raw graph is one giant component, so we keep transfers where either endpoint
    is structurally flagged and take weakly connected components of that subgraph.
    Returns one row per component with size and, for evaluation, its share of
    laundering.
    """
    s = structural_flags(df, fan_pct=fan_pct, min_passthrough=min_passthrough)
    flagged = set(s.loc[s["flagged"] == 1, "account"])

    # Both endpoints flagged keeps surfaced rings tight and high-precision on the
    # wired typologies (chains, cycles, scatter-gather). Fan-in and fan-out endpoints
    # need the burst-rate features that come next; they are not reliably caught here.
    sub = df[df["from_account"].isin(flagged) & df["to_account"].isin(flagged)]
    g = nx.from_pandas_edgelist(sub, "from_account", "to_account",
                                create_using=nx.DiGraph())
    comps = list(nx.weakly_connected_components(g))

    laund_acct = set(s.loc[s["is_laundering_acct"] == 1, "account"])
    recs = []
    for i, c in enumerate(sorted(comps, key=len, reverse=True)):
        recs.append({
            "network_id": i,
            "accounts": len(c),
            "flagged_accounts": len(c & flagged),
            "laundering_accounts": len(c & laund_acct),
            "laundering_share": len(c & laund_acct) / len(c),
        })
    return pd.DataFrame(recs)
