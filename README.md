# qse — self-evolving cross-sectional crypto trading research system

An autonomous alpha-factor discovery loop over the top 30 Binance USDT
perpetuals: a deterministic C++20 engine aligns the universe on a strict
1-hour epoch grid, trades a dollar-neutral rank long/short portfolio, and
scores factors by per-epoch Spearman **rank IC** and net PnL after costs.
Factors are proposed by a three-agent LLM pipeline (Proposer → Implementer
→ Critic) and gated by a trial-penalized out-of-sample score that punishes
specification-gaming.

## Layout

```
CMakeLists.txt             top-level build (or g++ directly, see below)
engine/
  include/qse/
    events.hpp             Kline + nanosecond timestamp conventions
    xsec/
      grid.hpp             MarketGrid — row-major T×N state grid — and
                           GridView, a factor's only window onto it: every
                           read is clamped to the locked row, so reading
                           the future is not expressible
      aligner.hpp          EpochAligner — the Time-Alignment Buffer.
                           Epoch barrier: a row locks only when every
                           symbol's watermark passes it; missing bars
                           forward-fill (close carried, volume 0, staleness
                           flagged); appends into locked rows throw
      ops.hpp              the factor vocabulary: xs_rank/xs_zscore/
                           xs_demean/clip + ts_mean/std/sum/delta/min/max/
                           ret/ret_std/corr, all NaN-aware
      factor.hpp           IXsecFactor: compute(GridView, out_scores) once
                           per epoch; QSE_REGISTER_XFACTOR registry
      portfolio.hpp        dollar-neutral book: long top 20% / short bottom
                           20% with a hysteresis band (q_in/q_out), fills at
                           next open, costs on traded weight deltas only,
                           funding PnL accrued from actual funding prints
      ic.hpp               Spearman rank IC (midrank ties) against the same
                           tradable open-to-open forward return the book earns
  src/xsec/ + src/xsec_main.cpp   implementations + qse_xsec CLI
data/
  raw/um/                  Binance zips, checksummed, never modified
  curated/um/              extracted kline + funding CSVs (read-only) and
                           universe.txt — the pinned symbol list
  pipeline/                download_um.sh + symbols.txt (candidate universe)
factors/xsec/              hand-built baselines (funding carry, reversal,
                           volume shock, low-vol)
factors/xsec_generated/    LLM-written factors, compiled into the engine
agents/                    Proposer / Implementer / Critic (Anthropic SDK;
                           DeepSeek via its Anthropic-compatible endpoint)
evaluation/fitness.py      deflated rank IC + the holdout trial ledger
reports/                   per-iteration hypothesis + backtest + critique
```

## Build & run

```sh
./data/pipeline/download_um.sh 2025-07 2026-06     # ~30 MB, checksummed;
                                                   # writes universe.txt
                                                   # (incremental runs never
                                                   # drop existing members)
./data/pipeline/download_um.sh 2026-08-01 2026-08-20  # fill the current month
                                                   # from daily dumps, merged
                                                   # into the month's CSV

g++ -std=c++20 -O3 -Iengine/include engine/src/xsec/*.cpp \
    engine/src/xsec_main.cpp factors/xsec/*.cpp \
    factors/xsec_generated/*.cpp -o build_bin/qse_xsec 2>/dev/null \
 || g++ -std=c++20 -O3 -Iengine/include engine/src/xsec/*.cpp \
    engine/src/xsec_main.cpp factors/xsec/*.cpp -o build_bin/qse_xsec

./build_bin/qse_xsec --selftest

./build_bin/qse_xsec --data-dir data/curated/um --start 2025-07 --end 2026-06 \
    --rebalance-every 24 --q-out 0.35 \
    --report reports/xsec_baselines.json --log-dir reports/logs/xsec
```

(With cmake: `cmake -S . -B build && cmake --build build && ctest --test-dir build`.)

Defaults: 5bps taker fee + 3bps slippage per side, charged on traded weight
deltas only; top/bottom 20% books with exits at the 35% rank band
(hysteresis); `--rebalance-every 24` re-forms the book daily while the rank
IC is still measured every epoch. `--fee-bps 0 --slip-bps 0` for gross-alpha
analysis.

The binary runs every factor twice and compares FNV digests over the full
equity/IC stream — identical inputs must produce identical results or the
batch fails.

## Anti-look-ahead design

- **Epoch barrier** (`EpochAligner`): factor evaluation at T_n is blocked
  until every symbol's close state for T_n is locked; a bar or funding print
  arriving for an already-locked epoch throws.
- **Forward-fill**: a symbol that drops a k-line carries its prior close
  with volume 0, `valid = 0`, and a staleness counter; the portfolio will
  not trade a name whose execution bar is forward-filled.
- **GridView clamp**: factors receive a view pinned to the locked row;
  every accessor clamps to it, so T_{n+1} data is unreachable by
  construction — and the selftest proves it by corrupting future epochs and
  asserting bit-identical past results.
- **Execution**: the book formed on T_n's scores is established at
  T_{n+1}'s open and earns the open-to-open return to T_{n+2}; the rank IC
  is measured against that same tradable return, not the untradable
  close-to-close gap.

## Anti-overfitting design

- `evaluation/fitness.py` keeps a persistent global trial counter N. Every
  holdout-window backtest increments it, and factors are scored by
  `IC_adj = mean_ic − λ·ic_std·√(2·ln N / T)` — the extreme-value bound on
  the best score N zero-skill strategies would reach by chance. Grinding
  the backtest raises the bar for every future factor.
- Hard promotion gate, all on the held-out quarter (2026-04..06): mean rank
  IC > 0.03 AND deflated IC > 0 AND positive net Sharpe.
- The Critic receives the baselines, the fitness breakdown, and the 15
  worst per-epoch ICs, and must check them against the hypothesis's own
  predicted failure modes.

## What the baselines established

- **Costs, not signal, were the killer.** At hourly rebalancing every
  baseline has real gross edge (funding carry ≈ +25%/yr gross, reversal IC
  t-stat ≈ 9) and every one loses net.
- **Funding-rate carry survives costs** because its edge is a cash flow
  (the perp funding transfer), not decaying price alpha: smoothing funding
  over 168h and rebalancing daily cuts turnover to ~0.8%/epoch and nets
  ≈ +44%/yr, Sharpe ≈ 2.7, maxDD ≈ 10% over the 12-month window — holding
  up on the 2026-Q2 quarter it never trained on (Sharpe 5.0 there).
- **Rank IC alone does not promote**: low-vol keeps its IC out-of-sample
  but loses money there — which is exactly why the gate also demands
  positive net Sharpe.

## Agent loop

```sh
pip install anthropic pydantic

# DeepSeek via its Anthropic-compatible endpoint (JSON-prompted fallback)
export LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=...
python -m agents.pipeline_xsec [iters]

# Anthropic: native structured outputs
export ANTHROPIC_API_KEY=...
python -m agents.pipeline_xsec [iters]

# Override the model for either provider
LLM_MODEL=claude-sonnet-5 python -m agents.pipeline_xsec
```

Each iteration: Proposer hypothesizes a cross-sectional ranking formula in
the ops vocabulary → Implementer emits a .cpp into `factors/xsec_generated/`
(compile-error retry loop) → the factor is built into the same binary as the
baselines and backtested over the train window (2025-07..2026-03) → the
Critic sees baselines, deflated-IC fitness, and the worst per-epoch ICs,
then promotes/revises/rejects. A train promote buys exactly ONE look at the
held-out quarter, and that look is charged to the trial ledger whether it
passes or not. Rejected factors are deleted but their trial still counts.

## Live paper trading (`qse_live`)

The research engine's streaming append path, wired to the real world:

- **Signals on real data**: klines + funding polls `fapi.binance.com`
  (public REST, no key) feed the same `EpochAligner` — the epoch barrier,
  forward-fill, and tradability mask all work exactly as in backtest.
- **Orders on the Futures Testnet**: signed REST to
  `testnet.binancefuture.com` — real order flow, simulated funds. Market
  orders sized from the model book's weight deltas (`clientOrderId`
  idempotency, 1x isolated margin, minQty/step filtering).
- **Model PnL stays pure**: the equity curve is computed from production
  grid opens with `run_xsec`'s exact accounting; the venue (testnet) curve
  is tracked separately as telemetry, with per-fill slippage vs the epoch
  open.

Build (CMake only; the agent pipeline's direct-g++ build is unaffected):

```sh
cmake -S . -B build && cmake --build build --target qse_live
```

Setup: create testnet API keys at testnet.binancefuture.com (API
Management → HMAC SHA256), then:

```sh
export BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=...

# 1. Offline replay over curated CSVs — the model digest must equal the
#    backtest digest (it is the same FNV stream run_xsec folds):
./build/engine/qse_live --data-dir data/curated/um --replay \
    --start 2025-07 --end 2026-06 --log-dir reports/live

# 2. Dry run: production data, model book only, no venue calls (no keys):
./build/engine/qse_live --data-dir data/curated/um --no-orders

# 3. Paper trade:
./build/engine/qse_live --data-dir data/curated/um --capital 10000
```

Flags: `--factor` (default `funding_carry_168h`), `--capital`, `--poll-sec`
(12), `--horizon-days` (365), `--fee-bps/--slip-bps/--q-in/--q-out/
--min-names/--rebalance-every` (live defaults `--q-out 0.35
--rebalance-every 24` — the promoted research config), `--log-dir`
(`reports/live`: per-epoch `epochs.jsonl` + exit `summary.json`),
`--selftest` (unit checks + a synthetic-grid replay-equivalence proof that
the live code path reproduces `run_xsec` bit-for-bit).

Live output: a `[status]` line every 60s (locked rows, model equity, last
finalized epoch's PnL/IC, venue balance + unrealized, model-book position
count) and one `t=…` line per finalized epoch (hourly). Since
`funding_carry_168h` re-forms its book daily, hours with zero orders are
normal — the heartbeat keeps the session's pulse visible between
rebalances.

Live limitations, by design: 1h epochs only; the aligner span is fixed at
startup (`--horizon-days`) and the process exits cleanly near its end —
restart re-warms from curated CSVs (re-run `download_um.sh` when the
curated→now gap approaches ~60 days); REST poll latency means fills land
seconds after the epoch open (the backtest assumes exactly at the open —
the gap shows up as slippage telemetry); funding settles 00/08/16 UTC and
its print is ingested before the kline that closes the epoch, per the
aligner's ordering rule; a symbol that drops a kline is excluded from the
book by the same tradability mask as in backtest. Decision-time semantics:
the book for epoch t is placed when row t locks (≈ open[t+1]) using only
row t's validity — whether the execution bar will be real is unknowable in
advance, a distinction the backtest can make and live cannot.

Next steps: multi-factor combination of promoted factors, universe
expansion past 30 names, and venue-side auto-reconciliation (the adapter
currently logs position drift and re-expresses the model book each
rebalance).
