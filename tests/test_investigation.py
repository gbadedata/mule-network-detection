"""Smoke test for the investigation SQL: it parses into the named queries and each one
executes against a small registered mock without error.
"""

import importlib.util
from pathlib import Path

import duckdb

from aml import aml_data

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_investigation", ROOT / "scripts" / "run_investigation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sql_parses_into_expected_queries():
    runner = _load_runner()
    queries = runner.parse_queries((ROOT / "sql" / "investigation.sql").read_text())
    expected = {"fan_in_collectors", "fan_out_distributors", "rapid_passthrough",
                "layering_two_hop", "cross_bank_flow", "value_concentration"}
    assert expected <= set(queries)


def test_each_query_executes():
    runner = _load_runner()
    queries = runner.parse_queries((ROOT / "sql" / "investigation.sql").read_text())
    txns = aml_data.load_aml_frame(aml_data.mock_aml_frames(
        n_accounts=800, n_legit=4000, n_fan_in=8, n_fan_out=8, n_chains=8,
        n_cycles=6, n_scatter_gather=5, seed=3))
    con = duckdb.connect()
    con.register("txns", txns)
    for _name, sql in queries.items():
        con.execute(sql).df()  # must not raise
