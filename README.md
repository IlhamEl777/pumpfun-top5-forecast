# pumpfun-top5-forecast

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/IlhamEl777/pumpfun-top5-forecast/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/IlhamEl777/pumpfun-top5-forecast/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

A self-calibrating probabilistic forecaster for new pump.fun coins. It reads the house's 10-minute snapshots of every coin launched since the opening (`/input/history.ndjson`, `/input/task.json`) and writes one forecast (`/output/forecast.json`): for each of the top 5 places by market cap, a probability for each named coin holding that place at an answer read, a best-guess top 5, and a probability distribution over market-cap size bins per place.

How it works: every coin's log10 market cap (SOL) is projected from the history cut to the answer reads as a normal distribution — its centre is the last observed cap (a coin missing from the last mark is capped at that mark's chart floor) plus a horizon-saturating drift built from the coin's own shrunk trajectory slope, a venue prior (graduated pool coins are stickier than bonding-curve coins) and the damped growth of the chart's top cap; its spread grows with the square root of the horizon. Coins the history cannot name (launched after the cut, or never ranked) are a field of pseudo-coins drawn from the peak-cap distribution that young coins reached inside the history; in the fixed-seed Monte Carlo they compete for places like any coin, and the mass they win is exactly the mass the forecast leaves unnamed at each place. The Monte Carlo tallies how often each coin holds each place and which size bin the cap at each place falls in; those frequencies are the forecast. All parameters are estimated from the history given, so the same program fits the 2-hour practice window and the 24-hour final. Standard library only, no network, deterministic: two runs on the same history are byte for byte identical.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with history.ndjson and task.json> MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
