# HYSTERESIS

**A path-dependent market response model.**

HYSTERESIS is an open research project exploring a simple idea: two markets can look identical right now — same price, same volatility, same volume — and still behave differently in the near future, because they arrived at that state through different histories. A market recovering from a recent shock is not the same as a market that got there gradually, even if the numbers match.

## Core Hypothesis

> The predictive state of a market depends not only on its current features, but on the path — the recent shocks and recoveries — by which it got there.

HYSTERESIS tests this directly by giving the model a dedicated **memory state** that tracks unresolved shock/recovery history, separate from its model of the market's current condition, and by learning how shocks propagate across assets as a **state-dependent response operator** rather than a fixed correlation matrix.

## What the Model Predicts

Instead of a single number, HYSTERESIS forecasts a full distribution over:

- Expected return
- Expected volatility
- Tail-risk probability (chance of a large loss)
- Liquidity recovery time

## Why It's Different

| Typical latent-dynamics model | HYSTERESIS |
| --- | --- |
| One latent state, history buried in hidden activations | Separate, inspectable memory state with a learned recovery rate |
| Predicts return or volatility alone | Predicts return, volatility, tail risk, and recovery time together |
| Static or implicit cross-asset interactions | Explicit, directed, sparse shock-propagation graph that changes with market state |
| Interpretability added after the fact | Memory levels, propagation edges, and recovery rates are native outputs |

## Documentation

- [`MATH.md`](./MATH.md) — full mathematical and implementation guide, written to be understandable even for beginners in quant finance and math.

## Status

Research prototype / work in progress. Not a trading system — no execution logic, position sizing, or risk controls are implemented. Hypotheses are meant to be tested and potentially falsified through the ablations described in `MATH.md`, not assumed true.