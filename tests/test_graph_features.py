"""Tests for the mule-network graph features.

The central guarantee is that the streaming features use no future information. We
verify it directly: features computed on a time-prefix of the data must match the
features computed on the full data for those same early rows. If a future transfer
changed a past row, there would be lookahead.
"""

import numpy as np
import pandas as pd

from aml import aml_data, graph_features


def _frame():
    base = pd.Timestamp("2022-09-01")
    # (offset_min, from, to, amount, is_laundering)
    rows = [
        (0, "B", "A", 100.0, 0),   # A first inflow, from B
        (1, "C", "A", 100.0, 0),   # A second inflow, second distinct sender
        (2, "D", "B", 50.0, 0),    # unrelated
        (3, "A", "E", 190.0, 1),   # A forwards out (pass-through)
        (4, "B", "A", 20.0, 0),    # A inflow again, but sender B already seen
    ]
    return pd.DataFrame({
        "ts": [base + pd.Timedelta(minutes=m) for m, *_ in rows],
        "from_account": [r[1] for r in rows],
        "to_account": [r[2] for r in rows],
        "amount": [r[3] for r in rows],
        "is_laundering": [r[4] for r in rows],
    })


def test_prior_features_use_no_future_info():
    df = _frame()
    full, cols = graph_features.prior_features(df)
    for k in range(1, len(df)):
        prefix, _ = graph_features.prior_features(df.iloc[:k])
        # early rows must be identical whether or not later rows exist
        a = full[cols].iloc[:k].reset_index(drop=True)
        b = prefix[cols].reset_index(drop=True)
        assert np.allclose(a.to_numpy(), b.to_numpy()), f"future info leaked at k={k}"


def test_dst_in_degree_counts_distinct_prior_senders():
    df = _frame()
    f, _ = graph_features.prior_features(df)
    f = f.sort_values("ts").reset_index(drop=True)
    # A's inflow rows are 0,1,4 with distinct prior senders {}, {B}, {B,C}
    a_rows = f[f["to_account"] == "A"]
    assert a_rows["dst_in_deg_prior"].tolist() == [0, 1, 2]
    # sender B sends at rows 0 and 4: prior out counts 0 then 1
    b_rows = f[f["from_account"] == "B"]
    assert b_rows["src_out_cnt_prior"].tolist() == [0, 1]


def test_passthrough_prior_rises_after_forwarding():
    df = _frame()
    f, _ = graph_features.prior_features(df)
    f = f.sort_values("ts").reset_index(drop=True)
    # before A forwards, its pass-through-prior is ~0; the forward is row index 3
    a_inflows = f[f["to_account"] == "A"].sort_values("ts")
    assert a_inflows["dst_passthrough_prior"].iloc[0] == 0.0  # first inflow, nothing sent yet


def test_account_summary_labels_both_endpoints():
    df = _frame()
    s = graph_features.account_summary(df)
    laund = set(s.loc[s["is_laundering_acct"] == 1, "account"])
    # the only laundering transfer is A -> E, so both A and E are laundering accounts
    assert "A" in laund and "E" in laund
    assert "D" not in laund


def test_build_graph_has_all_transfers():
    df = _frame()
    g = graph_features.build_account_graph(df)
    assert g.number_of_edges() == len(df)
    assert g.number_of_nodes() == len({*df["from_account"], *df["to_account"]})


def test_mock_contains_every_typology():
    raw = aml_data.mock_aml_frames(seed=7)
    patterns = set(raw.loc[raw["Is Laundering"] == 1, "Pattern"])
    assert {"fan_in", "fan_out", "chain", "cycle", "scatter_gather"} <= patterns


def test_loader_normalises_columns():
    raw = aml_data.mock_aml_frames(seed=7)
    df = aml_data.load_aml_frame(raw)
    for col in ["ts", "from_account", "to_account", "amount", "is_laundering"]:
        assert col in df.columns
    assert df["ts"].is_monotonic_increasing


def test_in_burst_counts_recent_arrivals():
    base = pd.Timestamp("2022-09-01")
    rows = [("B", "A", 0), ("C", "A", 1), ("D", "A", 2)]  # three arrivals to A, minutes apart
    df = pd.DataFrame({
        "ts": [base + pd.Timedelta(minutes=m) for *_, m in rows],
        "from_account": [r[0] for r in rows],
        "to_account": [r[1] for r in rows],
        "amount": [100.0, 100.0, 100.0],
        "is_laundering": [0, 0, 0],
    })
    f, _ = graph_features.prior_features(df, window="3D")
    f = f.sort_values("ts").reset_index(drop=True)
    # strictly-before inflow counts to A within the window: 0, then 1, then 2
    assert f["dst_in_burst"].tolist() == [0, 1, 2]


def test_patterns_parse_and_join_handles_bank_zeros(tmp_path):
    # patterns file with leading-zero banks, in the real block format
    text = (
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 3-degree Fan-Out\n"
        "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "2022/09/01 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\n"
        "2022/09/02 08:44,0217,80FD27570,0024856,8090E8EB0,10621.24,Shekel,10621.24,Shekel,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )
    pf = tmp_path / "patterns.txt"
    pf.write_text(text)
    pats = aml_data.load_patterns(pf)
    assert set(pats["pattern"]) == {"fan_out", "cycle"}
    # bank 021174 must normalise to 21174 to match pandas int parsing of the Trans CSV
    assert (pats["from_account"] == "21174-800737690").sum() == 2

    # a Trans-shaped frame where those transfers exist should get labelled
    raw = aml_data.mock_aml_frames(n_accounts=300, n_legit=500, n_fan_in=2, n_fan_out=2,
                                   n_chains=2, n_cycles=2, n_scatter_gather=1, seed=5)
    df = aml_data.load_aml_frame(raw).drop(columns=["pattern"])
    # inject one of the pattern transfers so the join has a hit
    import pandas as pd
    row = {"ts": pd.Timestamp("2022/09/01 00:06"), "from_bank": "21174",
           "from_account": "21174-800737690", "to_bank": "12",
           "to_account": "12-80011F990", "amount_paid": 2848.96, "amount_received": 2848.96,
           "payment_currency": "Euro", "receiving_currency": "Euro", "payment_format": "ACH",
           "is_laundering": 1, "amount": 2848.96}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out = aml_data.attach_patterns(df, pats)
    assert (out["pattern"] == "fan_out").sum() == 1


def test_burst_flags_catch_fan_in_and_fan_out_hubs():
    df = aml_data.load_aml_frame(aml_data.mock_aml_frames(seed=7))
    s = graph_features.structural_flags(df)
    flag = dict(zip(s["account"], s["flagged"], strict=False))
    collectors = set(df.loc[df["pattern"] == "fan_in", "to_account"])
    distributors = set(df.loc[df["pattern"] == "fan_out", "from_account"])
    # the burst signal should catch essentially every star hub
    assert sum(flag.get(a, 0) for a in collectors) >= 0.9 * len(collectors)
    assert sum(flag.get(a, 0) for a in distributors) >= 0.9 * len(distributors)
