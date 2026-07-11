"""Tests for the supervised layer: split integrity, scoring, and the queue.

Uses a small mock so training is fast. These check the pipeline is sound, not that any
particular score is achieved; the numbers that matter come from real data.
"""


from aml import aml_data, metrics, model, scoring
from aml.graph_features import prior_features


def _small():
    raw = aml_data.mock_aml_frames(n_accounts=1500, n_legit=8000, n_fan_in=15,
                                   n_fan_out=15, n_chains=15, n_cycles=10,
                                   n_scatter_gather=8, seed=1)
    return aml_data.load_aml_frame(raw)


def test_time_split_is_ordered_and_disjoint():
    df, cols = prior_features(_small())
    tr, va, te = model.time_split(df, 0.6, 0.2)
    assert len(tr) + len(va) + len(te) == len(df)
    # strictly increasing boundaries in time
    assert tr["ts"].max() <= va["ts"].min()
    assert va["ts"].max() <= te["ts"].min()


def test_scores_are_probabilities():
    df, cols = prior_features(_small())
    tr, _, te = model.time_split(df)
    clf = model.train_model(tr, cols)
    s = model.score(clf, te, cols)
    assert len(s) == len(te)
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_model_beats_base_rate_on_mock():
    df, cols = prior_features(_small())
    tr, _, te = model.time_split(df)
    clf = model.train_model(tr, cols)
    s = model.score(clf, te, cols)
    ap = metrics.pr_auc(te["is_laundering"].to_numpy(), s)
    # the injected structure is separable, so the model should clear the base rate widely
    assert ap > 5 * te["is_laundering"].mean()


def test_account_queue_surfaces_laundering():
    df, cols = prior_features(_small())
    tr, _, te = model.time_split(df)
    clf = model.train_model(tr, cols)
    s = model.score(clf, te, cols)
    q = scoring.account_queue(te, s, top=10)
    assert len(q) == 10
    assert "reasons" in q.columns
    # the highest-priority accounts should be mostly laundering
    assert q["is_laundering_acct"].mean() >= 0.5


def test_at_budget_recovers_value():
    df, cols = prior_features(_small())
    tr, _, te = model.time_split(df)
    clf = model.train_model(tr, cols)
    s = model.score(clf, te, cols)
    b = metrics.at_budget(te, s, budget=int(0.03 * len(te)))
    assert 0.0 <= b["value_recall"] <= 1.0
    assert b["recall"] > te["is_laundering"].mean()
