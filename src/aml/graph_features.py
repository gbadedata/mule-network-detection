"""Graph construction and features for money-mule network detection.

Two kinds of feature, kept deliberately separate, because they have different rules
and different deployment stories:

1. Streaming features (`prior_features`) describe an account's history *strictly
   before* the current transfer: distinct counterparties and amounts so far, its
   pass-through ratio, and crucially the burst counts (counterparties arriving in a
   short trailing window) that separate a mule hub from a legit account of the same
   total degree. These use no future information, so they are what a real-time monitor
   could score on. This is the graph version of the strictly-before discipline.

2. Retrospective features (`account_summary`) describe an account over a whole window
   for an investigator: peak burst, and a u-turn measure (does an outflow closely match
   money that just arrived). They are not leakage-safe for real-time scoring and are not
   model inputs; they support the investigation view and the evaluation.

`candidate_networks` surfaces groups to investigate. A real transaction graph is one
giant weakly connected component, so components of the raw graph are useless. We flag
accounts by typology-specific structure, then take components of the transfers among
them, plus the ego network of each flagged hub to recover star rings (fan-in, fan-out)
whose one-shot members have no signature on their own.
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
    "dst_in_burst", "src_out_burst", "src_recent_in_ratio",
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


def _prior_flows(df: pd.DataFrame, window: str = "3D") -> dict[str, np.ndarray]:
    """Per-transfer flow features for both endpoints, strictly before the transfer.

    Built from a per-account event log: each transfer is one inflow event at the
    receiver and one outflow event at the sender, so both roles of an account are
    captured. Returns arrays aligned to df rows:

      dst_cum_in, dst_cum_out   receiver's cumulative inflow / outflow so far
      dst_in_burst              receiver's inflow COUNT in the trailing window (fan-in)
      src_out_burst             sender's outflow COUNT in the trailing window (fan-out)
      src_recent_in             sender's inflow AMOUNT in the trailing window, i.e. money
                                that just arrived and is now being forwarded (u-turn)

    Windowed values use a per-account searchsorted on sorted timestamps, so they are
    strictly-before and cheap.
    """
    n = len(df)
    w = pd.Timedelta(window)
    amt = df["amount"].to_numpy(float)
    acct = np.concatenate([df["to_account"].to_numpy(), df["from_account"].to_numpy()])
    ts = np.concatenate([df["ts"].to_numpy("datetime64[ns]"),
                         df["ts"].to_numpy("datetime64[ns]")])
    inflow = np.concatenate([amt, np.zeros(n)])
    in_cnt = np.concatenate([np.ones(n), np.zeros(n)])
    out_cnt = np.concatenate([np.zeros(n), np.ones(n)])
    outflow = np.concatenate([np.zeros(n), amt])
    is_in = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    orig = np.concatenate([np.arange(n), np.arange(n)])

    order = np.lexsort((ts, acct))  # by account, then time
    acct_s, ts_s, is_in_s, orig_s = acct[order], ts[order], is_in[order], orig[order]
    inflow_s, outflow_s = inflow[order], outflow[order]
    in_cnt_s, out_cnt_s = in_cnt[order], out_cnt[order]

    cum_in = np.zeros(2 * n)
    cum_out = np.zeros(2 * n)
    win_in_amt = np.zeros(2 * n)
    win_in_cnt = np.zeros(2 * n)
    win_out_cnt = np.zeros(2 * n)

    bounds = np.flatnonzero(np.r_[True, acct_s[1:] != acct_s[:-1], True])
    for gi in range(len(bounds) - 1):
        a, b = bounds[gi], bounds[gi + 1]
        gts = ts_s[a:b]
        pos = np.arange(b - a)
        left = np.searchsorted(gts, gts - w, side="left")
        pin = np.r_[0.0, np.cumsum(inflow_s[a:b])]
        pout = np.r_[0.0, np.cumsum(outflow_s[a:b])]
        pic = np.r_[0.0, np.cumsum(in_cnt_s[a:b])]
        poc = np.r_[0.0, np.cumsum(out_cnt_s[a:b])]
        cum_in[a:b] = pin[pos]        # events strictly before this one
        cum_out[a:b] = pout[pos]
        win_in_amt[a:b] = pin[pos] - pin[left]
        win_in_cnt[a:b] = pic[pos] - pic[left]
        win_out_cnt[a:b] = poc[pos] - poc[left]

    def gather(arr: np.ndarray, want_in: bool) -> np.ndarray:
        out = np.zeros(n)
        mask = is_in_s if want_in else ~is_in_s
        out[orig_s[mask]] = arr[mask]
        return out

    return {
        "dst_cum_in": gather(cum_in, True),
        "dst_cum_out": gather(cum_out, True),
        "dst_in_burst": gather(win_in_cnt, True),
        "src_out_burst": gather(win_out_cnt, False),
        "src_recent_in": gather(win_in_amt, False),
    }


def _max_burst(df: pd.DataFrame, key_col: str, window: str) -> pd.Series:
    """Per-account max number of events in any trailing window (inclusive).

    Retrospective, for the investigation view: the peak inflow or outflow burst an
    account ever showed, which is what separates a fan-in collector from a legit
    account of the same total degree.
    """
    w = pd.Timedelta(window)
    out: dict[str, int] = {}
    for key, sub in df[[key_col, "ts"]].sort_values([key_col, "ts"]).groupby(key_col, sort=False):
        gts = sub["ts"].to_numpy("datetime64[ns]")
        pos = np.arange(len(gts))
        left = np.searchsorted(gts, gts - w, side="left")
        out[key] = int((pos - left + 1).max())
    return pd.Series(out)


def prior_features(df: pd.DataFrame, window: str = "3D"):
    """Return (df_with_features, feature_cols): leakage-safe streaming features."""
    df = df.sort_values("ts").reset_index(drop=True)
    src, dst = df["from_account"], df["to_account"]
    amt = df["amount"].to_numpy(float)

    dst_in_cnt = _prior_count(dst)
    dst_amt_in = _prior_sum(dst, amt)
    f = _prior_flows(df, window=window)

    feats = {
        "amount_log": np.log1p(amt),
        "src_out_cnt_prior": _prior_count(src),
        "src_out_deg_prior": _prior_distinct(src, dst),
        "src_amt_out_prior": _prior_sum(src, amt),
        "dst_in_cnt_prior": dst_in_cnt,
        "dst_in_deg_prior": _prior_distinct(dst, src),
        "dst_amt_in_prior": dst_amt_in,
        "dst_amt_out_prior": f["dst_cum_out"],
        # pass-through: money the receiver has already forwarded per unit received.
        "dst_passthrough_prior": f["dst_cum_out"] / (f["dst_cum_in"] + 1.0),
        # is this amount a spike versus what the receiver usually takes in?
        "amt_to_dst_mean_prior": amt / (dst_amt_in / np.maximum(dst_in_cnt, 1) + 1.0),
        # burst: counterparties arriving in a short window separate mules from hubs.
        "dst_in_burst": f["dst_in_burst"],
        "src_out_burst": f["src_out_burst"],
        # u-turn: money that just arrived and is being forwarded on now.
        "src_recent_in_ratio": f["src_recent_in"] / (amt + 1.0),
        "hour": df["ts"].dt.hour.to_numpy(),
    }
    feat_df = pd.DataFrame(feats, index=df.index).fillna(0.0)
    out = pd.concat([df, feat_df], axis=1)
    return out, list(feat_df.columns)


def account_summary(df: pd.DataFrame, window: str = "3D") -> pd.DataFrame:
    """Retrospective per-account structure over the window (investigation + eval)."""
    out_g = df.groupby("from_account")
    in_g = df.groupby("to_account")
    accts = pd.Index(sorted(set(df["from_account"]) | set(df["to_account"])), name="account")

    s = pd.DataFrame(index=accts)
    s["out_cnt"] = out_g.size().reindex(accts).fillna(0)
    s["in_cnt"] = in_g.size().reindex(accts).fillna(0)
    s["distinct_out"] = out_g["to_account"].nunique().reindex(accts).fillna(0)
    s["distinct_in"] = in_g["from_account"].nunique().reindex(accts).fillna(0)
    s["max_in_burst"] = _max_burst(df, "to_account", window).reindex(accts).fillna(0)
    s["max_out_burst"] = _max_burst(df, "from_account", window).reindex(accts).fillna(0)
    s["amt_out"] = out_g["amount"].sum().reindex(accts).fillna(0.0)
    s["amt_in"] = in_g["amount"].sum().reindex(accts).fillna(0.0)
    hi = np.maximum(s["amt_in"], s["amt_out"])
    s["passthrough"] = np.minimum(s["amt_in"], s["amt_out"]) / hi.replace(0, np.nan)
    s["passthrough"] = s["passthrough"].fillna(0.0)
    s["net_flow"] = s["amt_in"] - s["amt_out"]

    # u-turn: does an outflow closely match money that just arrived? We score the
    # amount match between this outflow and the account's recent inflow, near 1 when
    # they are similar and low when they differ. High for chain and scatter
    # intermediaries that forward what they received, low for accounts whose in and out
    # happen at unrelated times and amounts, which static balance cannot tell apart.
    flows = _prior_flows(df, window=window)
    amt = df["amount"].to_numpy(float)
    recent_in = flows["src_recent_in"]
    match = np.minimum(recent_in, amt) / np.maximum(np.maximum(recent_in, amt), 1.0)
    rp = pd.Series(match, index=df["from_account"].to_numpy()).groupby(level=0).max()
    s["rapid_passthrough"] = rp.reindex(accts).fillna(0.0)

    laund = pd.concat([
        df[["from_account", "is_laundering"]].rename(columns={"from_account": "account"}),
        df[["to_account", "is_laundering"]].rename(columns={"to_account": "account"}),
    ])
    s["laundering_txns"] = laund.groupby("account")["is_laundering"].sum().reindex(accts).fillna(0)
    s["is_laundering_acct"] = (s["laundering_txns"] > 0).astype(int)
    return s.reset_index()


def structural_flags(df: pd.DataFrame, fan_pct: float = 98.0,
                     min_rapid: float = 0.85) -> pd.DataFrame:
    """Flag accounts by typology-specific structure, since no single rule fits all.

    A collector shows fan-in (many distinct senders), a distributor shows fan-out
    (many distinct receivers), a layering or scatter-gather intermediary shows
    balanced pass-through. We take a union of these, using high percentiles for the
    hub-like counts. This is first-pass triage; legit hubs still slip through, which
    is what the supervised model is for.
    """
    s = account_summary(df)
    bin_, bout = s["max_in_burst"].to_numpy(float), s["max_out_burst"].to_numpy(float)
    # burst peaks, not total degree: a collector and a legit account can have the same
    # number of counterparties, but the collector packs them into a short window.
    in_hi = np.percentile(bin_[bin_ > 0], fan_pct) if (bin_ > 0).any() else np.inf
    out_hi = np.percentile(bout[bout > 0], fan_pct) if (bout > 0).any() else np.inf
    s["fan_in_flag"] = ((bin_ >= in_hi) & (bin_ >= 4)).astype(int)
    s["fan_out_flag"] = ((bout >= out_hi) & (bout >= 4)).astype(int)
    s["passthrough_flag"] = (s["rapid_passthrough"] >= min_rapid).astype(int)
    s["flagged"] = ((s["fan_in_flag"] + s["fan_out_flag"] + s["passthrough_flag"]) > 0).astype(int)
    return s


def candidate_networks(df: pd.DataFrame, fan_pct: float = 98.0,
                       min_rapid: float = 0.85) -> pd.DataFrame:
    """Surface candidate mule networks as components of the flagged subgraph.

    The raw graph is one giant component, so we keep transfers where either endpoint
    is structurally flagged and take weakly connected components of that subgraph.
    Returns one row per component with size and, for evaluation, its share of
    laundering.
    """
    s = structural_flags(df, fan_pct=fan_pct, min_rapid=min_rapid)
    flagged = set(s.loc[s["flagged"] == 1, "account"])
    hubs = set(s.loc[(s["fan_in_flag"] == 1) | (s["fan_out_flag"] == 1), "account"])

    # Two kinds of ring. Wired typologies (chains, cycles, scatter-gather) show up as
    # transfers where both endpoints are flagged. Star typologies (fan-in, fan-out)
    # have a detectable hub and many one-shot counterparties that are invisible alone,
    # so we pull in the hub's ego network to recover the whole ring.
    both = df["from_account"].isin(flagged) & df["to_account"].isin(flagged)
    hub_edge = df["from_account"].isin(hubs) | df["to_account"].isin(hubs)
    sub = df[both | hub_edge]
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
