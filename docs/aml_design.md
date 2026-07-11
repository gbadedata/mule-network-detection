# Money-mule network detection: design and results

## The problem

Money mules move illicit funds through networks of accounts. The signal is not in any
single transfer but in how accounts are wired together over time: many accounts feeding
one (fan-in), one feeding many (fan-out), money walked along a chain, cycles, and rapid
pass-through where funds arrive and leave almost at once. At a realistic base rate this
matters a great deal, because laundering is a tiny fraction of all transfers, so scoring
transactions in isolation recovers little, and the structure is where the signal
concentrates.

This project builds the account graph, computes features over it, scores transfers, and
prioritises accounts for investigation. It uses the IBM "Transactions for Anti-Money
Laundering" dataset (synthetic, with per-transaction laundering labels and a separate
file naming the typology of each laundering attempt), and a schema-faithful mock so the
whole pipeline runs with no download.

## Two kinds of feature, kept separate

The distinction is the core discipline of the project.

Streaming features describe an account's history strictly before the current transfer:
distinct counterparties and amounts so far, pass-through ratio, whether the current
amount is a spike against the account's usual inflow, and the burst counts
(counterparties arriving in a short trailing window) that separate a mule hub from a
legit account of the same total degree. They use no future information, so they are what
a real-time monitor could score on. A feature computed on the whole graph, including an
account's later transfers, is the graph version of lookahead: information you would not
have at decision time. The guard is tested directly. Features computed on a time-prefix
of the data must match the features computed on the full data for those same early rows;
if a later transfer changes an earlier row, that is leakage.

Retrospective features describe an account over a whole window for an investigator: peak
burst, a u-turn measure, and burst concentration. They are not leakage-safe for real-time
scoring and are never model inputs; they support the investigation view.

## Surfacing networks

A real transaction graph is one giant weakly connected component, so components of the
raw graph tell you nothing. We flag accounts by typology-specific structure and take
components of the transfers among them, then pull in the ego network of each flagged hub,
which recovers star rings (fan-in, fan-out) whose one-shot members have no signature on
their own.

## What the mock showed, and why it flattered the model

On the mock the model reached a transaction PR-AUC near 0.99, with per-typology recall of
93 to 100 percent. That number is a property of synthetic data, not a result. The mock
injects clean, separable typologies and, more to the point, has no legitimate
high-throughput accounts. Real data has them everywhere, and they are the whole
difficulty.

## Results on real data (HI-Small)

Roughly 5.08 million transfers, a 0.177 percent laundering rate on the held-out tail,
scored with the leakage-safe streaming features on a time split.

Transaction scoring. PR-AUC is 0.037, about twenty times the base rate but low in
absolute terms, and that is the reality of per-transaction laundering detection.
The operating result is the number that matters to a review team: reviewing the top two
percent of scored transfers recovers 54.5 percent of laundered value and 38 percent of
laundering transfers.

Per typology. Recall within a two percent budget, over the labelled laundering transfers,
is moderate and uneven everywhere: fan-in 51, stack 47, random 48, scatter-gather 43,
gather-scatter 39, bipartite 39, cycle 33, fan-out 27. Nothing is cleanly solved and
nothing is missed. The time-concentrated typologies fare a little better than the ones
that blend into ordinary high-volume behaviour.

The account queue and the whale confound. Ranking accounts by expected laundered value
alone surfaces legitimate payment processors and exchanges, accounts moving hundreds of
billions through thousands of counterparties, because raw dollar value dominates. The
separating idea is burst concentration: the share of an account's activity that falls in
its single busiest window. A mule's activity is concentrated near one; a legit hub is
busy all year, near zero. Weighting the queue by risk, log-dollars, and concentration
raised the share of laundering accounts at the top of the queue from two in ten to six in
ten, and the accounts it now surfaces include genuine high-throughput laundering hubs
rather than legitimate ones. The remaining false positives are concentration-one accounts
that the model scored highly, which include both mules and quiet legitimate accounts.

## Limitations and what deployment would need

The dataset is synthetic; no bank releases labelled mule data, so this is the standard
public stand-in and the ceiling on any claim made here. Telling mules apart from
legitimate high-throughput accounts is the central unsolved problem, and burst
concentration is a first step rather than a solution. A production system would add
peer-group baselines (throughput relative to accounts of the same type) and
account-history features (how new the behaviour is), and would score entities and
networks rather than isolated transfers. The value of this project is a pipeline that is
disciplined about time, clear about what synthetic data can and cannot show, and specific
about where the real difficulty lies.
