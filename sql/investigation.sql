-- Investigation queries for money-mule analysis.
--
-- These run against a normalised transaction table `txns` with columns:
--   ts, from_account, to_account, from_bank, to_bank, amount, is_laundering
-- (scripts/run_investigation.py registers it from the raw CSV or the mock).
--
-- On real data the hard confound is legitimate high-throughput accounts (processors,
-- exchanges) that move huge volume through many counterparties. Raw burst surfaces
-- them, so the fan queries also report burst_concentration: the share of an account's
-- activity that falls in its single busiest window. A mule's burst is concentrated
-- (near 1); a legit hub is busy all the time (near 0). The is_laundering column is
-- shown only to see how the structural view lines up with truth; ranking never uses it.

-- 1. Fan-in collectors: accounts that gather inflows, with how concentrated the peak is.
--    Window function, not a self-join, so it scales to millions of rows.
-- name: fan_in_collectors
WITH burst AS (
    SELECT to_account AS account, ts,
           COUNT(*) OVER (PARTITION BY to_account ORDER BY ts
                          RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND CURRENT ROW) AS w_in
    FROM txns
),
agg AS (
    SELECT to_account AS account, COUNT(*) AS total_in,
           ROUND(SUM(amount), 2) AS total_in_value,
           AVG(is_laundering) AS laundering_share
    FROM txns GROUP BY to_account
)
SELECT b.account,
       MAX(b.w_in)                                AS peak_inflows_3d,
       a.total_in,
       ROUND(MAX(b.w_in) * 1.0 / a.total_in, 2)   AS burst_concentration,
       a.total_in_value,
       ROUND(a.laundering_share, 3)               AS laundering_share
FROM burst b JOIN agg a ON a.account = b.account
GROUP BY b.account, a.total_in, a.total_in_value, a.laundering_share
HAVING MAX(b.w_in) >= 5
ORDER BY peak_inflows_3d DESC, burst_concentration DESC
LIMIT 25;

-- 2. Fan-out distributors: the sending-side mirror, with the same concentration read.
-- name: fan_out_distributors
WITH burst AS (
    SELECT from_account AS account, ts,
           COUNT(*) OVER (PARTITION BY from_account ORDER BY ts
                          RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND CURRENT ROW) AS w_out
    FROM txns
),
agg AS (
    SELECT from_account AS account, COUNT(*) AS total_out,
           ROUND(SUM(amount), 2) AS total_out_value,
           AVG(is_laundering) AS laundering_share
    FROM txns GROUP BY from_account
)
SELECT b.account,
       MAX(b.w_out)                               AS peak_outflows_3d,
       a.total_out,
       ROUND(MAX(b.w_out) * 1.0 / a.total_out, 2) AS burst_concentration,
       a.total_out_value,
       ROUND(a.laundering_share, 3)               AS laundering_share
FROM burst b JOIN agg a ON a.account = b.account
GROUP BY b.account, a.total_out, a.total_out_value, a.laundering_share
HAVING MAX(b.w_out) >= 5
ORDER BY peak_outflows_3d DESC, burst_concentration DESC
LIMIT 25;

-- 3. Rapid pass-through: money forwarded within a day of arriving, at a similar amount.
--    Restricted to moderate-degree accounts so the self-join stays bounded.
-- name: rapid_passthrough
WITH deg AS (
    SELECT account, COUNT(*) AS cnt FROM (
        SELECT from_account AS account FROM txns
        UNION ALL SELECT to_account AS account FROM txns
    ) GROUP BY account
),
active AS (SELECT account FROM deg WHERE cnt BETWEEN 2 AND 200)
SELECT i.to_account                       AS account,
       COUNT(*)                           AS uturn_events,
       ROUND(SUM(o.amount), 2)            AS forwarded_value,
       ROUND(AVG(EXTRACT(EPOCH FROM (o.ts - i.ts)) / 3600.0), 1) AS avg_hours_held
FROM txns i
JOIN txns o
    ON o.from_account = i.to_account
   AND o.ts > i.ts
   AND o.ts <= i.ts + INTERVAL 24 HOUR
   AND o.amount BETWEEN 0.8 * i.amount AND 1.2 * i.amount
WHERE i.to_account IN (SELECT account FROM active)
GROUP BY i.to_account
HAVING COUNT(*) >= 2
ORDER BY forwarded_value DESC
LIMIT 25;

-- 4. Two-hop layering: A -> B -> C where B forwards soon after receiving.
--    Middle account restricted to moderate degree, same reason.
-- name: layering_two_hop
WITH deg AS (
    SELECT account, COUNT(*) AS cnt FROM (
        SELECT from_account AS account FROM txns
        UNION ALL SELECT to_account AS account FROM txns
    ) GROUP BY account
),
active AS (SELECT account FROM deg WHERE cnt BETWEEN 2 AND 200)
SELECT a.from_account                     AS source,
       b.from_account                     AS middle,
       b.to_account                       AS destination,
       ROUND(a.amount, 2)                 AS amount_in,
       ROUND(b.amount, 2)                 AS amount_out,
       ROUND(EXTRACT(EPOCH FROM (b.ts - a.ts)) / 3600.0, 1) AS hours_between
FROM txns a
JOIN txns b
    ON b.from_account = a.to_account
   AND b.ts > a.ts
   AND b.ts <= a.ts + INTERVAL 48 HOUR
   AND b.amount BETWEEN 0.85 * a.amount AND 1.0 * a.amount
WHERE a.from_account <> b.to_account
  AND b.from_account IN (SELECT account FROM active)
ORDER BY amount_in DESC
LIMIT 25;

-- 5. Cross-bank movement: value leaving each bank to other banks.
-- name: cross_bank_flow
SELECT from_bank, to_bank,
       COUNT(*)                           AS transfers,
       ROUND(SUM(amount), 2)              AS total_moved,
       ROUND(AVG(is_laundering), 3)       AS laundering_share
FROM txns
WHERE from_bank <> to_bank
GROUP BY from_bank, to_bank
ORDER BY total_moved DESC
LIMIT 25;

-- 6. Value concentration: accounts by the laundered value they touched.
-- name: value_concentration
SELECT account,
       ROUND(SUM(amount), 2)              AS laundered_value,
       COUNT(*)                           AS laundering_transfers
FROM (
    SELECT from_account AS account, amount FROM txns WHERE is_laundering = 1
    UNION ALL
    SELECT to_account AS account, amount FROM txns WHERE is_laundering = 1
)
GROUP BY account
ORDER BY laundered_value DESC
LIMIT 25;
