-- Investigation queries for money-mule analysis.
--
-- These run against a normalised transaction table `txns` with columns:
--   ts, from_account, to_account, from_bank, to_bank, amount, is_laundering
-- (scripts/run_investigation.py registers it from the raw CSV or the mock).
--
-- They are the questions an analyst asks when triaging a network: who gathers, who
-- scatters, who passes money straight through, and where the value concentrates. The
-- is_laundering column is used only to show how the structural view lines up with the
-- ground truth; the ranking never depends on it.

-- 1. Fan-in collectors: accounts that gather from many senders in a short window.
--    Total degree finds legit hubs; the burst (distinct senders within any 3-day
--    window) finds the collector. Ranked by peak burst.
-- name: fan_in_collectors
WITH windowed AS (
    SELECT i.to_account AS account, i.ts,
           COUNT(DISTINCT j.from_account) AS window_senders,
           SUM(j.amount)                  AS window_in
    FROM txns i
    JOIN txns j
        ON j.to_account = i.to_account
       AND j.ts <= i.ts
       AND j.ts > i.ts - INTERVAL 3 DAY
    GROUP BY i.to_account, i.ts
),
lshare AS (
    SELECT to_account AS account, AVG(is_laundering) AS laundering_share
    FROM txns GROUP BY to_account
)
SELECT w.account,
       MAX(w.window_senders)          AS peak_senders_3d,
       ROUND(MAX(w.window_in), 2)     AS peak_in_3d,
       ROUND(l.laundering_share, 3)   AS laundering_share
FROM windowed w
JOIN lshare l ON l.account = w.account
GROUP BY w.account, l.laundering_share
HAVING MAX(w.window_senders) >= 5
ORDER BY peak_senders_3d DESC, peak_in_3d DESC
LIMIT 25;

-- 2. Fan-out distributors: accounts that scatter to many receivers in a short window.
--    Same idea as fan-in, on the sending side. Ranked by peak burst.
-- name: fan_out_distributors
WITH windowed AS (
    SELECT i.from_account AS account, i.ts,
           COUNT(DISTINCT j.to_account) AS window_receivers,
           SUM(j.amount)                AS window_out
    FROM txns i
    JOIN txns j
        ON j.from_account = i.from_account
       AND j.ts <= i.ts
       AND j.ts > i.ts - INTERVAL 3 DAY
    GROUP BY i.from_account, i.ts
),
lshare AS (
    SELECT from_account AS account, AVG(is_laundering) AS laundering_share
    FROM txns GROUP BY from_account
)
SELECT w.account,
       MAX(w.window_receivers)        AS peak_receivers_3d,
       ROUND(MAX(w.window_out), 2)    AS peak_out_3d,
       ROUND(l.laundering_share, 3)   AS laundering_share
FROM windowed w
JOIN lshare l ON l.account = w.account
GROUP BY w.account, l.laundering_share
HAVING MAX(w.window_receivers) >= 5
ORDER BY peak_receivers_3d DESC, peak_out_3d DESC
LIMIT 25;

-- 3. Rapid pass-through: money forwarded within a day of arriving, at a similar amount.
--    The classic u-turn. Each row is an inflow matched to a quick, similar outflow.
-- name: rapid_passthrough
SELECT
    i.to_account                       AS account,
    COUNT(*)                           AS uturn_events,
    ROUND(SUM(o.amount), 2)            AS forwarded_value,
    ROUND(AVG(EXTRACT(EPOCH FROM (o.ts - i.ts)) / 3600.0), 1) AS avg_hours_held
FROM txns i
JOIN txns o
    ON o.from_account = i.to_account
   AND o.ts > i.ts
   AND o.ts <= i.ts + INTERVAL 24 HOUR
   AND o.amount BETWEEN 0.8 * i.amount AND 1.2 * i.amount
GROUP BY i.to_account
HAVING COUNT(*) >= 2
ORDER BY forwarded_value DESC
LIMIT 25;

-- 4. Two-hop layering: A -> B -> C where B forwards soon after receiving.
--    Surfaces the middle of a layering chain and the value passing through it.
-- name: layering_two_hop
SELECT
    a.from_account                     AS source,
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
ORDER BY amount_in DESC
LIMIT 25;

-- 5. Cross-bank movement: value leaving each bank to other banks.
--    Layering often hops between institutions to break the trail.
-- name: cross_bank_flow
SELECT
    from_bank,
    to_bank,
    COUNT(*)                           AS transfers,
    ROUND(SUM(amount), 2)              AS total_moved,
    ROUND(AVG(is_laundering), 3)       AS laundering_share
FROM txns
WHERE from_bank <> to_bank
GROUP BY from_bank, to_bank
ORDER BY total_moved DESC
LIMIT 25;

-- 6. Value concentration: accounts by the laundered value they touched.
--    A check on where the real money is, for sizing the investigation.
-- name: value_concentration
SELECT
    account,
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
