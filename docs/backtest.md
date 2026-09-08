# How the backtest works

The engine replays candles one at a time and never lets the strategy see a price
it could not have seen live. Everything below is a deliberate choice; where a
choice could go either way, the pessimistic option was taken.

## Order of events inside a bar

For every bar `t`, in this exact order:

1. **Open.** The order queued at the close of bar `t-1` is filled at the open of
   bar `t`, plus slippage.
2. **Intrabar.** The stop and the profit targets are checked against the bar's
   high and low. If the bar touches both, **the stop is assumed to have come
   first** — a single OHLC bar cannot tell us the real sequence, so we take the
   unfavourable one.
3. **Funding.** Every 8-hour boundary (00:00, 08:00, 16:00 UTC) crossed while the
   position is open is charged. Longs pay, shorts receive.
4. **Close.** Only now does the strategy see bar `t`. It may tighten the stop
   (effective from bar `t+1`) and may queue one order for bar `t+1`.
5. **Mark.** Equity is marked to market and appended to the equity curve.

Because step 4 comes after steps 1–3, a signal generated on bar `t` cannot be
filled at bar `t` prices. `tests/unit/test_backtest_engine.py::TestNoLookAhead`
locks this in with a bar that gaps 50% overnight.

## Fills

- **Slippage** always works against the trade: buys fill higher, sells lower.
  Three models are configurable — `fixed_bps`, `percent` and `atr`.
- **Gaps.** A bar that opens beyond the stop has already gapped through it, so the
  fill happens at the open, not at the stop level. The same rule applies to
  targets, where it happens to work in our favour.
- **Fees** are charged on the filled notional on both sides of the trade.
- **Sizing** comes from the risk manager, never from the strategy. The stop
  distance travels with the signal and is re-anchored to the actual fill price,
  so the amount risked stays equal to what was approved.

## Partial exits

A position may leave in several pieces (for example half at the first target,
the rest on the trailing stop). The journal keeps **one row per position**, with
the size-weighted average exit price and the reason of the final exit, so trade
counts and win rates are not inflated by scale-outs.

## Artefacts

Each run writes `runs/<run_id>/`:

| File | Contents |
| --- | --- |
| `meta.json` | Run id, code version, git sha, full config, data hash, duration |
| `config.yaml` | The effective configuration, including resolved strategy params |
| `trades.parquet` | One row per closed position |
| `equity.parquet` | Balance, equity, open position count and drawdown per bar |
| `signals.parquet` | Every signal, including rejected ones and why |
| `metrics.json` | The full metric set of section 9.1 |
| `report.html` | Standalone interactive report (built by `tradingbot backtest report`) |
| `charts/*.png` | Key figures rendered with kaleido, optional (`--png`) |

`tradingbot backtest report --run-id …` writes `report.html`. It inlines Plotly, so
the file opens in a browser with no server. Trade markers on the price chart are
taken from `trades.parquet`: a long entry is a triangle-up at `entry_ts`, a short
entry a triangle-down, an exit a cross. Walk-forward, Monte Carlo and the
parameter heatmap are honest placeholders until those stages produce artefacts.

Slippage is the one cost that the trade schema has no column for, so the engine
tallies what it actually cost across all fills and reports it at run level. On a
liquid pair with the default ATR model it is routinely larger than commissions,
which is why it is shown next to them rather than folded into the PnL.

The identifier is `YYYYMMDD-HHMMSS-<config hash>`. The prefix keeps runs apart,
the suffix identifies the configuration. Trade identifiers are derived from the
trade itself rather than randomly generated, so two runs over the same input
produce byte-identical `trades.parquet` files.
