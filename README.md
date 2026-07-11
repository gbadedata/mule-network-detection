# Money-Mule Network Detection

Detecting money laundering by the shape of money flow, not one transaction at a time.

At a realistic base rate (laundering is well under 1% of transactions), scoring
transfers in isolation is close to hopeless. The signal lives in structure: many
accounts feeding one (fan-in), one feeding many (fan-out), money passed along a chain,
rapid pass-through, and cycles. This project builds the account graph, computes
features over it, and surfaces candidate mule networks for investigation. It runs on
the IBM "Transactions for Anti-Money Laundering" dataset, and on a schema-faithful mock
with no download.

## Two kinds of feature, kept separate on purpose

**Streaming features** (`graph_features.prior_features`) describe an account's history
*strictly before* the current transfer: distinct counterparties so far, amount moved so
far, pass-through ratio (money forwarded out per unit received in), and whether the
current amount is a spike versus the account's usual inflow. They use no future
information, so they are what a real-time monitor could actually score on. This is the
graph version of the strictly-before discipline: a feature that peeks at an account's
future transfers is lookahead you would not have at decision time.

The leakage guard is tested directly: features computed on a time-prefix of the data
must match the features computed on the full data for those same early rows
(`tests/test_graph_features.py`).

**Retrospective features** (`graph_features.account_summary`) describe an account over a
whole window for an investigator triaging surfaced networks. They are not leakage-safe
for real-time scoring and are not model inputs; they support the investigation view and
honest evaluation.

## Surfacing candidate networks

A real transaction graph is one giant weakly connected component, so components of the
raw graph tell you nothing. `candidate_networks` flags accounts by a union of
typology-specific signals (fan-in, fan-out, balanced pass-through) and takes the
weakly connected components of the transfers among them, which isolates candidate rings.

## An honest finding, already

Different laundering typologies have different structural signatures, and no single rule
fits all of them. On the mock, balanced pass-through cleanly catches cycles, chains, and
scatter-gather intermediaries, but fan-in and fan-out endpoints score near zero: a
collector receiving from ten accounts looks like an ordinary account by degree alone,
because plenty of legitimate accounts do too. The separating signal is the burst (many
counterparties in a short window) and the amount pattern, not raw degree. That is the
case for a supervised model over multiple weak signals rather than one hand-tuned
threshold, and it is what comes next.

## Scoring and the investigation queue

A gradient-boosted model scores each transfer using the leakage-safe streaming features,
trained on a time split so it never sees the future. Because those features carry each
account's network context (burst, pass-through, degree), this is network-aware
transaction scoring rather than isolated-row scoring, which is the whole point at a base
rate this low. Transaction scores aggregate to account risk, and `account_queue` ranks
accounts by expected laundered value (risk times money moved) with plain-language reason
codes, so each alert explains itself ("fans out: 14 outflows in a short window").

Evaluation is deliberately honest: transaction PR-AUC, precision and recall and value
recovered under a fixed investigation budget, and recall broken out per typology, since
the model is strong on some and weak on others. On the mock the scores are high because
the injected typologies are cleanly separable; that is a property of synthetic data, not
a claim about production. The numbers that matter come from running on the real IBM data.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test           # includes the no-lookahead guard
python run_demo.py  # graph, features, and network surfacing
python run_model.py # train, evaluate honestly, and print the account queue
python scripts/run_investigation.py  # analyst SQL over the transactions (DuckDB)
python scripts/make_figures.py       # value-vs-budget and per-typology recall
```

Data download instructions are in [`data/aml/README.md`](data/aml/README.md). Raw CSVs
are git-ignored.
