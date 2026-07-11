"""Run the investigation queries over the transaction data with DuckDB.

Loads the real IBM-AML CSV if present at data/aml/HI-Small_Trans.csv, else generates the
mock, normalises it, registers it as `txns`, and runs each named query in
sql/investigation.sql.

    python scripts/run_investigation.py            # run all queries
    python scripts/run_investigation.py fan_in_collectors
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aml import aml_data  # noqa: E402

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "investigation.sql"


def load_txns():
    real = Path("data/aml/HI-Small_Trans.csv")
    if real.exists():
        print("Loading real IBM-AML from data/aml/ ...\n")
        return aml_data.load_aml(real)
    print("Real IBM-AML not found; using the schema-faithful mock.\n")
    return aml_data.load_aml_frame(aml_data.mock_aml_frames(seed=7))


def parse_queries(text: str) -> dict[str, str]:
    """Split the SQL file into named blocks using the `-- name: <id>` markers."""
    queries: dict[str, str] = {}
    name = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            if name and buf:
                queries[name] = "\n".join(buf).strip()
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name and buf:
        queries[name] = "\n".join(buf).strip()
    return queries


def main() -> None:
    txns = load_txns()  # noqa: F841  (registered into duckdb below)
    con = duckdb.connect()
    # keep peak memory down on the large real file: allow disk spill and fewer threads.
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=4")
    con.register("txns", txns)
    queries = parse_queries(SQL_PATH.read_text())

    wanted = sys.argv[1:] or list(queries)
    for name in wanted:
        if name not in queries:
            print(f"[skip] no query named {name}")
            continue
        print(f"=== {name} ===")
        print(con.execute(queries[name]).df().to_string(index=False))
        print()


if __name__ == "__main__":
    main()
