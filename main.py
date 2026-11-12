#!/usr/bin/env python3
"""Probabilistic forecaster for the make.fun pumpfun-top5-forecast-01 bounty.

Reads the house's 10-minute snapshots of every pump.fun coin launched since
the opening (/input/history.ndjson + /input/task.json) and writes one
forecast (/output/forecast.json): for each of the top 5 places, a probability
for each named coin holding that place at an answer read, a best-guess top 5,
and a probability distribution over market-cap size bins for each place.

Method (self-calibrating: every parameter below is estimated from the history
it is given, so the same program fits the short practice window and the long
final one):

* Each coin's log10 market cap (in SOL) is projected from the history cut to
  the answer reads as a normal distribution. Its centre is the coin's last
  observed cap (a coin missing from the last mark is capped at that mark's
  chart floor: it fell out of the top 200) plus a per-hour drift built from
  the coin's own shrunk trajectory slope, a venue prior (graduated pool coins
  are stickier than bonding-curve coins) and the damped growth of the chart's
  top cap. Drift saturates with horizon so young coins' explosive first-hour
  slopes are not extrapolated linearly for a day. The spread grows with the
  square root of the horizon and is wider for curve coins than pool coins.
* Coins the history cannot name at a place (launched after the cut, or never
  ranked) are modelled as a field of pseudo-coins whose caps are drawn from
  the distribution of peak caps that young coins reached inside the history,
  each minus a horizon-decaying shortfall (a newborn at a near read has
  usually not yet reached, or has retraced from, its lifetime peak).
  In the Monte Carlo they compete for places like any coin; the probability
  mass they win is exactly the mass the forecast leaves unnamed at each
  place, which the scoring spreads over the answer chart's unnamed coins.
* A fixed-seed Monte Carlo over the read window (each sample first draws a
  read time uniformly from the seven answer reads) tallies, per place, how
  often each coin holds it and the distribution of the cap that holds it;
  those frequencies, floored to 6 decimals, are the forecast. Two runs on
  the same history are byte for byte identical.

Standard library only, no network, deterministic.
"""

import bisect
import json
import math
import os
import random
from datetime import datetime, timezone

SEED = 987654321
SAMPLES = 6000
MAX_CANDIDATES = 350
EXTRA_RISERS = 60
NEW_DRAWS_BASE = 24  # plus 5 per hour of read window, capped at 110


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json")) as fh:
        task = json.load(fh)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                marks.append(json.loads(line))
    marks.sort(key=lambda m: parse_ts(m["at"]))
    return task, marks


def coin_series(marks):
    """mint -> dict with obs [(t, log10 cap SOL, venue, complete)], created."""
    coins = {}
    floors = []
    top1 = []
    for mark in marks:
        t = parse_ts(mark["at"])
        rows = mark.get("rows") or []
        if rows:
            floors.append((t, math.log10(int(rows[-1]["market_cap_lamports"]) / 1e9)))
            top1.append((t, math.log10(int(rows[0]["market_cap_lamports"]) / 1e9)))
        for row in rows:
            mint = row["mint"]
            entry = coins.get(mint)
            if entry is None:
                entry = coins[mint] = {
                    "obs": [],
                    "created": parse_ts(row["created_at"]),
                    "mayhem": bool(row.get("mayhem")),
                }
            entry["obs"].append(
                (t, math.log10(int(row["market_cap_lamports"]) / 1e9),
                 row.get("venue"), bool(row.get("complete")))
            )
    return coins, floors, top1


def slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    return sxy / sxx if sxx > 0 else 0.0


def build_candidates(coins, floors, t_cut):
    floor_last = floors[-1][1] if floors else 1.4
    cands = []
    for mint, data in coins.items():
        obs = data["obs"]
        t_last, x_last, venue_last, complete_last = obs[-1]
        in_last = t_last == floors[-1][0] if floors else False
        x_eff = x_last if in_last else min(x_last, floor_last - 0.02)
        span_h = (obs[-1][0] - obs[0][0]) / 3600.0
        if span_h > 0.02 and len(obs) >= 2:
            raw_slope = (obs[-1][1] - obs[0][1]) / span_h
        else:
            raw_slope = 0.0
        shrink = min(1.0, len(obs) / 3.0) * min(1.0, span_h / 0.75)
        graduated = venue_last == "pool" or complete_last
        cands.append({
            "mint": mint,
            "x": x_eff,
            "x_last": x_last,
            "slope": raw_slope * shrink,
            "graduated": graduated,
            "complete": complete_last,
            "stale": not in_last,
            "age_h": max(0.0, (t_cut - data["created"]) / 3600.0),
            "peak": max(o[1] for o in obs),
        })
    by_x = sorted(cands, key=lambda c: -c["x"])
    chosen = {c["mint"] for c in by_x[:MAX_CANDIDATES]}
    risers = sorted(cands, key=lambda c: -(c["x"] + 1.5 * max(0.0, c["slope"])))
    for c in risers:
        if len(chosen) >= MAX_CANDIDATES + EXTRA_RISERS:
            break
        chosen.add(c["mint"])
    return [c for c in cands if c["mint"] in chosen], floor_last


def newcomer_stats(coins, t0):
    """Distribution of peak log10 caps reached by coins born after t0."""
    peaks = []
    for data in coins.values():
        if data["created"] >= t0 - 1.0:
            peaks.append(max(o[1] for o in data["obs"]))
    if len(peaks) < 30:
        return None
    peaks.sort()
    cut = peaks[int(0.97 * len(peaks)):]
    if len(cut) < 8:
        cut = peaks[-max(8, len(peaks) // 20):]
    # Raw mean of the top peaks. The caller discounts this towards what a
    # young coin is worth at an answer read (see the new_mu adjustment in
    # main): a lifetime peak overstates a newborn's cap at a near read.
    mu = sum(cut) / len(cut)
    var = sum((p - mu) ** 2 for p in cut) / len(cut)
    return mu, max(0.30, math.sqrt(var))


def top_growth(top1):
    if len(top1) < 3:
        return 0.0
    half = top1[len(top1) // 2:]
    g = slope([t / 3600.0 for t, _ in half], [x for _, x in half])
    return max(-0.05, min(0.20, g)) * 0.35


def bin_of(value, edges):
    if value < edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges)
    return bisect.bisect_right(edges, value)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = load_input(input_dir)
    t0 = parse_ts(task["t0"])
    t_cut = parse_ts(task["history_until"])
    reads = [parse_ts(r) for r in task["reads"]]
    edges = [float(e) for e in task["size"]["edges"]]
    places = int(task.get("places", 5))

    coins, floors, top1 = coin_series(marks)
    cands, floor_last = build_candidates(coins, floors, t_cut)
    g_top = top_growth(top1)
    new_stats = newcomer_stats(coins, t0)

    h_min = max(0.0, (reads[0] - t_cut) / 3600.0)
    h_max = max(h_min, (reads[-1] - t_cut) / 3600.0)
    h_mid = 0.5 * (h_min + h_max)

    # Per-candidate projection parameters (drift saturates with horizon).
    proj = []
    for c in cands:
        if c["graduated"]:
            venue_drift, sdh = -0.010, 0.20
        elif c["complete"]:
            venue_drift, sdh = 0.020, 0.28
        else:
            venue_drift, sdh = -0.030, 0.33
        drift = venue_drift + 0.45 * c["slope"] + g_top
        drift = max(-0.22, min(0.18, drift))
        sdh *= 1.0 - 0.15 * min(1.0, c["age_h"] / 24.0)
        # A coin present in the last mark was measured minutes ago, so its
        # projection starts tighter than 0.22: on the practice backtest the
        # wider base spread donated probability mass to the pseudo-coin
        # field and to far-tail outcomes (0.15 scored best of 0.12..0.26,
        # monotonically). Stale coins keep the wider 0.30.
        base_sd = 0.30 if c["stale"] else 0.15
        proj.append((c["mint"], c["x"], drift, sdh, base_sd))

    if new_stats is None:
        top_x = max((c["x"] for c in cands), default=3.0)
        new_mu, new_sd = top_x - 0.3, 0.6
    else:
        new_mu, new_sd = new_stats
        # Peak-to-read shortfall, decaying with horizon: a pseudo-coin's
        # cap at a read is its lifetime peak minus a shortfall (it has not
        # run up yet, or has retraced), modelled as an independent draw of
        # mean and spread `disc`, so the field's centre drops by disc and
        # its spread widens to sqrt(sd^2 + disc^2): most newborns sit far
        # below their peaks while a few still rocket past the incumbents.
        # Raw peaks with neither shift won ~61% of every place on the
        # practice backtest, starving the named incumbents that in fact
        # held most places. As the horizon grows a newborn has time to run
        # to its peak, so disc fades (0.62 at 15 min, ~0.1 at 12 h).
        disc = 0.7 / (1.0 + h_mid / 2.0)
        new_mu += 0.15 * min(1.0, h_mid / 6.0) - disc
        new_sd = math.sqrt(new_sd * new_sd + disc * disc)

    window_h = h_max  # reads end this many hours after the cut
    n_new = int(max(24, min(110, NEW_DRAWS_BASE + 5 * window_h)))

    rng = random.Random(SEED)
    place_counts = [dict() for _ in range(places)]
    size_counts = [[0] * (len(edges) + 1) for _ in range(places)]
    top5_counts = {}

    for _ in range(SAMPLES):
        horizon = rng.uniform(h_min, h_max)
        sat = horizon / (1.0 + horizon / 6.0)
        draws = []
        for mint, x, drift, sdh, base_sd in proj:
            mu = x + drift * sat
            sd = math.sqrt(base_sd * base_sd + sdh * sdh * horizon)
            draws.append((rng.gauss(mu, sd), mint))
        new_mu_h = new_mu + g_top * sat * 0.5
        for _ in range(n_new):
            draws.append((rng.gauss(new_mu_h, new_sd), None))
        draws.sort(key=lambda d: -d[0])
        for k in range(places):
            value, mint = draws[k]
            size_counts[k][bin_of(value, edges)] += 1
            if mint is not None:
                place_counts[k][mint] = place_counts[k].get(mint, 0) + 1
                top5_counts[mint] = top5_counts.get(mint, 0) + 1

    probabilities = []
    for k in range(places):
        rows = sorted(place_counts[k].items(), key=lambda kv: (-kv[1], kv[0]))
        for mint, count in rows[:100]:
            p = count / SAMPLES
            if p < 0.004:
                continue
            probabilities.append({
                "mint": mint,
                "place": k + 1,
                "p": int(p * 1_000_000) / 1_000_000,
            })

    def top5_score(mint):
        return sum(place_counts[k].get(mint, 0) * (places + 1 - k)
                   for k in range(places))

    ranked = sorted(
        top5_counts.items(),
        key=lambda kv: (-top5_score(kv[0]), -place_counts[0].get(kv[0], 0), kv[0]),
    )
    top5 = [mint for mint, _ in ranked[:places]]
    if len(top5) < places:  # degenerate history: fill by last observed cap
        fill = sorted(coins, key=lambda m: -max(o[1] for o in coins[m]["obs"]))
        for mint in fill:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    sizes = []
    for k in range(places):
        for b, count in enumerate(size_counts[k]):
            p = count / SAMPLES
            if p < 0.003:
                continue
            sizes.append({
                "place": k + 1,
                "bin": b,
                "p": int(p * 1_000_000) / 1_000_000,
            })

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w") as fh:
        json.dump(forecast, fh, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
