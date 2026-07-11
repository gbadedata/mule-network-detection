"""IBM-AML data: a real CSV loader and a schema-faithful mock.

The mock lets the whole pipeline run with no download. It injects the laundering
typologies the real generator uses (fan-in, fan-out, layering chains, cycles, and
scatter-gather), so structure is real work rather than a giveaway, and it mixes
legitimate traffic onto the same accounts so structure alone does not perfectly
separate mules.

The real loader reads the Kaggle files (e.g. HI-Small_Trans.csv) and normalises the
columns to the same names the rest of the code uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2022-09-01 00:00:00")
CURRENCIES = np.array(["US Dollar", "Euro", "Yuan", "Yen", "UK Pound"])
FORMATS = np.array(["ACH", "Cheque", "Wire", "Credit Card", "Cash"])

# Columns as they appear in the raw Kaggle CSV, in order. pandas renames the second
# "Account" header to "Account.1" on read, so we use that name here.
RAW_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid",
            "Payment Currency", "Payment Format", "Is Laundering"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw IBM-AML columns to the names the pipeline uses.

    Account ids are only unique together with their bank, so the account key is
    bank-prefixed. Amounts can differ across currencies; `amount` uses Amount Paid.
    """
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["Timestamp"]),
        "from_bank": df["From Bank"].astype(str),
        "from_account": df["From Bank"].astype(str) + "-" + df["Account"].astype(str),
        "to_bank": df["To Bank"].astype(str),
        "to_account": df["To Bank"].astype(str) + "-" + df["Account.1"].astype(str),
        "amount_paid": df["Amount Paid"].astype(float),
        "amount_received": df["Amount Received"].astype(float),
        "payment_currency": df["Payment Currency"].astype(str),
        "receiving_currency": df["Receiving Currency"].astype(str),
        "payment_format": df["Payment Format"].astype(str),
        "is_laundering": df["Is Laundering"].astype(int),
    })
    out["amount"] = out["amount_paid"]
    if "Pattern" in df.columns:
        out["pattern"] = df["Pattern"].astype(str)
    return out.sort_values("ts").reset_index(drop=True)


def load_aml(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a real IBM-AML transactions CSV (e.g. HI-Small_Trans.csv)."""
    return _normalise(pd.read_csv(path, nrows=nrows))


def load_aml_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an in-memory raw frame (used by the mock and tests)."""
    return _normalise(df)


# --------------------------------------------------------------------------- mock

def _accounts(rng, bank_pool, n):
    banks = rng.choice(bank_pool, n)
    ids = rng.integers(0x10000000, 0x9FFFFFFF, n)
    return np.array([f"{b:03d}-{i:08X}" for b, i in zip(banks, ids, strict=False)])


def mock_aml_frames(n_accounts: int = 6000, n_legit: int = 60_000, n_fan_in: int = 40,
                    n_fan_out: int = 40, n_chains: int = 40, n_cycles: int = 30,
                    n_scatter_gather: int = 25, seed: int = 7) -> pd.DataFrame:
    """Build a schema-faithful mock with injected laundering typologies.

    Returns a raw frame with the same columns as the Kaggle CSV plus a `Pattern`
    column (empty for legitimate rows) that the real files carry separately and that
    the evaluation uses for per-typology breakdowns.
    """
    rng = np.random.default_rng(seed)
    window_s = 45 * 86_400
    bank_pool = np.arange(1, 60)
    legit_accounts = _accounts(rng, bank_pool, n_accounts)
    mule_pool = _accounts(rng, bank_pool, 6000)
    rows = []

    def add(src, dst, t, amt, laundering, pattern=""):
        cur = rng.choice(CURRENCIES)
        sb, sa = src.split("-")
        db, da = dst.split("-")
        rows.append((REF_DATE + pd.Timedelta(seconds=float(t)), sb, sa, db, da,
                     round(float(amt), 2), cur, round(float(amt), 2), cur,
                     rng.choice(FORMATS), int(laundering), pattern))

    # --- legitimate background traffic (also lands on mule accounts by chance) ---
    src_idx = rng.integers(0, n_accounts, n_legit)
    dst_idx = rng.integers(0, n_accounts, n_legit)
    t_legit = rng.uniform(0, window_s, n_legit)
    amt_legit = np.maximum(1.0, rng.lognormal(6.0, 1.1, n_legit))
    for s, d, t, a in zip(src_idx, dst_idx, t_legit, amt_legit, strict=False):
        if s != d:
            add(legit_accounts[s], legit_accounts[d], t, a, 0)

    mp = 0

    def take(k):
        nonlocal mp
        got = mule_pool[mp:mp + k]
        mp += k
        return got

    # give mule accounts a little legitimate traffic too, so structure is not a tell
    def legit_noise(members, per=1):
        for m in members:
            for _ in range(per):
                other = legit_accounts[rng.integers(0, n_accounts)]
                add(m, other, rng.uniform(0, window_s),
                    max(1.0, rng.lognormal(5.5, 1.0)), 0)

    # --- fan-in (gather): many sources -> one collector, short window ---
    for _ in range(n_fan_in):
        m = take(int(rng.integers(6, 16)) + 1)
        collector = m[0]
        t0, base = rng.uniform(0, window_s - 3 * 86_400), rng.uniform(2000, 20000)
        for s in m[1:]:
            add(s, collector, t0 + rng.uniform(0, 2 * 86_400),
                base * rng.uniform(0.8, 1.2), 1, "fan_in")
        legit_noise(m)

    # --- fan-out (scatter): one distributor -> many ---
    for _ in range(n_fan_out):
        m = take(int(rng.integers(6, 16)) + 1)
        distributor = m[0]
        t0, base = rng.uniform(0, window_s - 3 * 86_400), rng.uniform(2000, 20000)
        for d in m[1:]:
            add(distributor, d, t0 + rng.uniform(0, 2 * 86_400),
                base * rng.uniform(0.8, 1.2), 1, "fan_out")
        legit_noise(m)

    # --- layering chain: A -> B -> C -> ... pass-through, increasing time ---
    for _ in range(n_chains):
        m = take(int(rng.integers(4, 8)))
        t, amt = rng.uniform(0, window_s - 5 * 86_400), rng.uniform(5000, 40000)
        for i in range(len(m) - 1):
            t += rng.uniform(3600, 2 * 86_400)
            add(m[i], m[i + 1], t, amt * rng.uniform(0.9, 0.99), 1, "chain")
        legit_noise(m)

    # --- cycle: A -> B -> C -> A ---
    for _ in range(n_cycles):
        m = take(int(rng.integers(3, 6)))
        t, amt = rng.uniform(0, window_s - 5 * 86_400), rng.uniform(5000, 30000)
        for i in range(len(m)):
            t += rng.uniform(3600, 2 * 86_400)
            add(m[i], m[(i + 1) % len(m)], t, amt * rng.uniform(0.9, 0.99), 1, "cycle")
        legit_noise(m)

    # --- scatter-gather: source -> many intermediaries -> collector ---
    for _ in range(n_scatter_gather):
        m = take(int(rng.integers(5, 10)) + 2)
        source, collector, mids = m[0], m[1], m[2:]
        t0 = rng.uniform(0, window_s - 4 * 86_400)
        amt, k = rng.uniform(8000, 50000), len(mids)
        for mid in mids:
            add(source, mid, t0 + rng.uniform(0, 86_400),
                amt / k * rng.uniform(0.9, 1.1), 1, "scatter_gather")
            add(mid, collector, t0 + rng.uniform(86_400, 2 * 86_400),
                amt / k * rng.uniform(0.85, 1.0), 1, "scatter_gather")
        legit_noise(m)

    raw = pd.DataFrame(rows, columns=RAW_COLS + ["Pattern"])
    return raw.sort_values("Timestamp").reset_index(drop=True)


def write_mock_aml(out_dir: str | Path, **kwargs) -> Path:
    """Write a mock CSV shaped like HI-Small_Trans.csv (drops the Pattern column)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mock_Trans.csv"
    mock_aml_frames(**kwargs).drop(columns=["Pattern"]).to_csv(path, index=False)
    return path
