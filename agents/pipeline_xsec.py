"""Cross-sectional ReAct discovery loop: Proposer → Implementer → backtest → Critic.

The xsec variant of agents/pipeline.py. The LLM proposes symbolic ranking
factors over the aligned 30-asset Binance UM-perp grid (1h epochs, 12
months); the engine scores them with the per-epoch Spearman rank IC and a
dollar-neutral top/bottom-20% long/short book with hysteresis, daily
rebalancing, and taker costs on traded deltas.

Fitness is the deflated mean rank IC (evaluation/fitness.py::evaluate_xsec)
with the same holdout-ledger discipline as the single-asset loop: train looks
are free, every holdout look increments trials.json and raises the bar.

Hard promotion gate (all on the holdout window):
    mean rank IC > 0.03  AND  deflated IC > 0  AND  net Sharpe > 0

python -m agents.pipeline_xsec [max_iterations]
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agents.config import load_provider, make_client
from agents.schemas import Critique, XsecFactorHypothesis, XsecFactorImplementation
from evaluation import fitness

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "factors" / "xsec_generated"
REPORTS = ROOT / "reports"
LOG_DIR = REPORTS / "logs"
DATA_DIR = ROOT / "data" / "curated" / "um"

# Train/holdout split by month: the last quarter is held out.
TRAIN = ("2025-07", "2026-03")
HOLDOUT = ("2026-04", "2026-06")

# Portfolio config passed to every run — matches reports/xsec_baselines.json.
PORTFOLIO_ARGS = ["--rebalance-every", "24", "--q-out", "0.35"]

MIN_HOLDOUT_IC = 0.03

CFG = load_provider()
client = make_client(CFG)

PROPOSER_XSEC_SYSTEM = """\
You are the Proposer in a quantitative research loop over the top 30 Binance
USDT perpetuals on a strict 1-hour epoch grid (12 months of data). The
portfolio is DOLLAR-NEUTRAL: every epoch your factor scores all 30 names,
the engine longs the top 20% and shorts the bottom 20% (with a hysteresis
band and daily rebalancing). Market direction is therefore irrelevant —
only the cross-sectional ORDERING of next-hour returns is scored.

Scoring: per-epoch Spearman rank IC between your scores and forward
open-to-open returns, deflated by ln(total holdout trials). The hard gate is
out-of-sample mean rank IC > 0.03 with positive net Sharpe after ~8bps per
side on traded weight deltas. High-turnover signals die on costs even with
good IC: prefer lookbacks of 12-168 bars over bar-to-bar noise.

Per symbol per closed 1h bar you have: open, high, low, close, volume,
quote_volume, taker_buy_ratio (taker buys / volume — order-flow imbalance),
funding (last funding rate, a crowding/positioning signal). Forward-filled
gap bars carry the prior close with volume 0.

Your formula must be expressible with ONLY this vocabulary:
  cross-sectional: xs_rank, xs_zscore, xs_demean, clip
  time-series:     ts_mean, ts_std, ts_sum, ts_delta, ts_min, ts_max,
                   ts_ret, ts_ret_std, ts_corr
Higher score = stronger long candidate.

Baselines already running (do not re-propose epsilon-variants of them):
funding-rate carry (16/72/168h means), short-horizon reversal (24/48/72h),
volume shock (24h vs 168h volume), low-vol tilt (168h return std)."""

IMPLEMENTER_XSEC_SYSTEM = """\
You are the Implementer. Translate the hypothesis into C++20 for the qse
cross-sectional engine. Output a single complete .cpp file that:
- includes "qse/xsec/factor.hpp" and "qse/xsec/ops.hpp"
- defines the class in an anonymous namespace, inheriting
  qse::xsec::IXsecFactor
- implements `void compute(const GridView& g, std::span<double> out)`.
  The ENGINE calls it once per epoch with history ending at the current
  row — NEVER loop over time yourself; use only GridView accessors
  (g.get/close/volume/funding/ret/is_valid, r = rows back) and the ops::
  helpers for any window statistic. Loop over symbols n only.
- writes one score per symbol into out[n]; write qse::xsec::kNaN when the
  window is not available or !g.is_valid(n) — NaN means "no opinion" and
  the engine excludes the name
- overrides `std::size_t warmup() const` returning the largest lookback + 1
- no I/O, no clocks, no randomness, no static mutable state (the engine
  runs every factor twice and rejects digest mismatches)
- ends with QSE_REGISTER_XFACTOR(<factor_name>, <ClassName>);
The factor_name must match the hypothesis name and be a valid C++
identifier, and must not collide with a baseline factor name."""

CRITIC_XSEC_SYSTEM = """\
You are the Critic. You receive the hypothesis, the train-window backtest of
the new factor AND all baselines (same grid, costs, portfolio construction),
the deflated-IC fitness dict, and the factor's 15 worst per-epoch ICs.
Find reasons the result is NOT real:
- IC t-stat < 2 in-sample, or IC driven by a handful of epochs (compare the
  worst-IC list against the mean — a real ranking edge is broad and shallow).
- IC positive but net Sharpe negative: turnover is eating the edge; check
  realized avg_turnover against the hypothesis's expected_turnover claim.
- The formula is an epsilon-perturbation of a baseline (trial grinding).
- PnL inconsistent with the stated rationale or failure modes.
The hard promotion gate on holdout is mean rank IC > 0.03, deflated IC > 0,
net Sharpe > 0. Promote on train only if the factor plausibly clears it.
A false promote is worse than a false reject."""


# --------------------------------------------------------------------------
# Structured output (same two-provider strategy as agents/pipeline.py)
# --------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON in model output")


def ask[T: BaseModel](schema: type[T], system: str, user: str, max_tokens: int = 8000) -> T:
    if CFG.native_structured_outputs:
        resp = client.messages.parse(
            model=CFG.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return resp.parsed_output

    json_system = (
        f"{system}\n\nRespond with a single JSON object matching this schema "
        f"(no prose, no markdown fences):\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )
    messages = [{"role": "user", "content": user}]
    last_err: Exception | None = None
    for _ in range(3):
        resp = client.messages.create(
            model=CFG.model, max_tokens=max_tokens, system=json_system,
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return schema.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError) as e:
            last_err = e
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Invalid: {e}. Emit only the corrected JSON object."},
            ]
    raise RuntimeError(f"provider '{CFG.name}' failed to produce valid {schema.__name__}: {last_err}")


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

def propose(feedback: str | None) -> XsecFactorHypothesis:
    user = "Propose a new cross-sectional ranking factor."
    if feedback:
        user += f"\n\nCritic feedback on your previous attempt:\n{feedback}"
    return ask(XsecFactorHypothesis, PROPOSER_XSEC_SYSTEM, user)


def implement(hyp: XsecFactorHypothesis,
              compile_errors: str | None = None) -> XsecFactorImplementation:
    factor_hpp = (ROOT / "engine/include/qse/xsec/factor.hpp").read_text()
    grid_hpp = (ROOT / "engine/include/qse/xsec/grid.hpp").read_text()
    ops_hpp = (ROOT / "engine/include/qse/xsec/ops.hpp").read_text()
    reference = (ROOT / "factors/xsec/funding_carry.cpp").read_text()
    user = (
        f"Hypothesis:\n{hyp.model_dump_json(indent=2)}\n\n"
        f"Engine interfaces:\n```cpp\n{factor_hpp}\n```\n"
        f"```cpp\n{grid_hpp}\n```\n```cpp\n{ops_hpp}\n```\n\n"
        f"Reference implementation (style guide):\n```cpp\n{reference}\n```"
    )
    if compile_errors:
        user += f"\n\nYour previous attempt failed to compile:\n{compile_errors}\nFix it."
    return ask(XsecFactorImplementation, IMPLEMENTER_XSEC_SYSTEM, user, max_tokens=8000)


def critique(hyp: XsecFactorHypothesis, batch: list[dict], fit: dict,
             worst_ics: list[dict]) -> Critique:
    user = (
        f"Hypothesis:\n{hyp.model_dump_json(indent=2)}\n\n"
        f"Train backtest (new factor + baselines, same grid/costs):\n"
        f"{json.dumps(batch, indent=2)}\n\n"
        f"Fitness (deflated rank IC):\n{json.dumps(fit, indent=2)}\n\n"
        f"Worst per-epoch ICs of the new factor:\n{json.dumps(worst_ics, indent=2)}"
    )
    return ask(Critique, CRITIC_XSEC_SYSTEM, user)


# --------------------------------------------------------------------------
# Build + backtest
# --------------------------------------------------------------------------

def build_engine_xsec() -> tuple[bool, str]:
    """Compile the xsec engine + baseline factors + generated factors with
    g++ directly (same reasoning as pipeline.build_engine: no cmake dep)."""
    out = ROOT / "build_bin" / "qse_xsec"
    out.parent.mkdir(exist_ok=True)
    srcs = [*(ROOT / "engine/src/xsec").glob("*.cpp"),
            ROOT / "engine/src/xsec_main.cpp",
            *(ROOT / "factors/xsec").glob("*.cpp"),
            *GENERATED.glob("*.cpp")]
    cmd = ["g++", "-std=c++20", "-O3",
           f"-I{ROOT}/engine/include", *map(str, srcs), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return r.returncode == 0, r.stderr[-4000:]


def run_xsec_batch(window: tuple[str, str], tag: str) -> tuple[list[dict], Path]:
    """One qse_xsec run of every registered factor over the month window."""
    log_dir = LOG_DIR / f"xsec_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / f"xsec_batch_{tag}.json"
    cmd = [str(ROOT / "build_bin/qse_xsec"),
           "--data-dir", str(DATA_DIR),
           "--start", window[0], "--end", window[1],
           *PORTFOLIO_ARGS,
           "--report", str(report_path), "--log-dir", str(log_dir)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(report_path.read_text()), log_dir


def worst_epoch_ics(log_dir: Path, name: str, k: int = 15) -> list[dict]:
    def iso(ns: int) -> str:
        return dt.datetime.fromtimestamp(ns / 1e9, dt.UTC).strftime("%Y-%m-%d %H:%M")
    rows = []
    lines = (log_dir / f"{name}_ic.csv").read_text().splitlines()[1:]
    for ln in lines:
        ts, ic, to, pnl = ln.split(",")
        if ic != "nan":
            rows.append({"epoch": iso(int(ts)), "ic": float(ic),
                         "turnover": float(to), "pnl_net": float(pnl)})
    return sorted(rows, key=lambda r: r["ic"])[:k]


def score(name: str, window: tuple[str, str], tag: str,
          in_sample: bool) -> tuple[list[dict], dict, list[dict]]:
    batch, log_dir = run_xsec_batch(window, tag)
    row = next((r for r in batch if r["factor"] == name), None)
    if row is None:
        raise RuntimeError(f"factor '{name}' missing from report — "
                           "name collision or registration failure")
    fit = fitness.evaluate_xsec(row, in_sample=in_sample)
    return batch, fit, worst_epoch_ics(log_dir, name)


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------

def run_xsec_loop(max_iterations: int = 3) -> None:
    print(f"provider={CFG.name} model={CFG.model} mode=xsec "
          f"train={TRAIN} holdout={HOLDOUT}", file=sys.stderr)

    ok, err = build_engine_xsec()
    if not ok:
        raise SystemExit(f"engine build failed before loop start:\n{err}")
    baseline_names = {r["factor"] for r in run_xsec_batch(TRAIN, "train")[0]}

    feedback: str | None = None
    for i in range(max_iterations):
        hyp = propose(feedback)
        print(f"[{i}] proposed: {hyp.name}", file=sys.stderr)
        if hyp.name in baseline_names:
            feedback = f"'{hyp.name}' collides with an existing factor name; pick a new name."
            continue

        src = GENERATED / f"{hyp.name}.cpp"
        errors: str | None = None
        for attempt in range(3):
            impl = implement(hyp, errors)
            src.write_text(impl.cpp_source)
            ok, errors = build_engine_xsec()
            if ok:
                break
            print(f"[{i}] compile attempt {attempt + 1} failed", file=sys.stderr)
            print(errors, file=sys.stderr)
        else:
            src.unlink(missing_ok=True)
            build_engine_xsec()
            feedback = "Implementation failed to compile 3 times; simplify the formula."
            continue

        # Stage 1 — iterate freely against the train window.
        batch, fit, worst = score(hyp.name, TRAIN, "train", in_sample=True)
        crit = critique(hyp, batch, fit, worst)
        print(f"[{i}] train: IC {fit['mean_ic']:.4f} (t {fit['ic_tstat']:.2f}), "
              f"net Sharpe {fit['ann_sharpe_net']:.2f}, turnover "
              f"{fit['avg_turnover']:.4f}; verdict {crit.verdict}", file=sys.stderr)

        record = {
            "provider": CFG.name, "model": CFG.model, "mode": "xsec",
            "hypothesis": hyp.model_dump(),
            "train_batch": batch, "train_fitness": fit,
            "critique": crit.model_dump(),
        }

        # Stage 2 — a train-promote buys ONE holdout look (ledger charges it).
        if crit.verdict == "promote":
            h_batch, h_fit, _ = score(hyp.name, HOLDOUT, "holdout", in_sample=False)
            record["holdout_batch"] = h_batch
            record["holdout_fitness"] = h_fit
            promoted = (h_fit["mean_ic"] > MIN_HOLDOUT_IC
                        and h_fit["adjusted_ic"] > 0.0
                        and h_fit["ann_sharpe_net"] > 0.0)
            print(f"[{i}] holdout: IC {h_fit['mean_ic']:.4f} "
                  f"(deflated {h_fit['adjusted_ic']:.4f}, trial "
                  f"#{h_fit['n_trials_global']}), net Sharpe "
                  f"{h_fit['ann_sharpe_net']:.2f} -> "
                  f"{'PROMOTED' if promoted else 'failed holdout'}", file=sys.stderr)
            if promoted:
                (REPORTS / f"xsec_{hyp.name}_iter{i}.json").write_text(
                    json.dumps(record, indent=2, default=str))
                return
            crit = Critique(
                verdict="revise",
                statistical_flaws=[
                    f"holdout mean IC {h_fit['mean_ic']:.4f} / deflated "
                    f"{h_fit['adjusted_ic']:.4f} / net Sharpe "
                    f"{h_fit['ann_sharpe_net']:.2f} missed the gate "
                    f"(need IC > {MIN_HOLDOUT_IC}, deflated > 0, Sharpe > 0)"],
                logical_flaws=[],
                revision_directive="The ranking edge did not survive the held-out "
                                   "quarter. Slow the signal down or find a "
                                   "structurally different cross-sectional spread.",
            )

        (REPORTS / f"xsec_{hyp.name}_iter{i}.json").write_text(
            json.dumps(record, indent=2, default=str))

        if crit.verdict == "reject":
            src.unlink(missing_ok=True)
            build_engine_xsec()
        feedback = crit.revision_directive
    print("Loop exhausted without promotion — negative result recorded.", file=sys.stderr)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_xsec_loop(iters)
