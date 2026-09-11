"""Fitness evaluation for LLM-generated cross-sectional factors.

The core defence against specification-gaming: every backtest the agent runs
against the evaluation window increments a global trial counter N, and the
factor is scored by its *deflated* Sharpe ratio — the Sharpe that must beat
what N zero-skill trials would have reached by chance (extreme-value bound):

    SR_0 = sqrt(V[{SR_n}]) * [(1-g)*Z^-1(1 - 1/N) + g*Z^-1(1 - 1/(N*e))]
    DSR  = Z( (SR - SR_0) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2) )

with g = Euler-Mascheroni, SR the per-observation Sharpe, T the number of
return observations behind it, and g3/g4 the skewness and (non-excess)
kurtosis of those returns — Bailey & Lopez de Prado's deflated Sharpe ratio.
Random trial-and-error raises N without raising SR, so grinding the backtest
monotonically lowers the bar-adjusted score.

Why the Sharpe and not the IC. A factor can make money either by predicting
returns or by collecting a cash flow, and the second kind has no IC at all:
funding carry earns the perp funding transfer, so its rank IC is ~0 by
construction — funding_carry_168h over 2025-08..2026-07 scores IC t-stat 0.06
while netting Sharpe 2.7. An IC threshold rejects exactly the factors that
earn a cash flow, which is the project's own promoted baseline. The rank IC
is still computed and reported (the Critic reasons about it, and the
worst-epoch ICs are the diagnostic that catches IC-driven blowups), but the
gate is on the deflated Sharpe.

T is the number of RETURN observations (report field `n_returns`), not the
number of times the book moved. A book that does not trade still draws a
fresh return every epoch, so held positions add draws rather than repeat one;
`n_bets` — the rebalances that actually moved the book — is recorded in the
report as evidence about the strategy, and would be the right count only for
a per-decision statistic. Hourly and daily samplings agree on this series:
the net P&L is serially uncorrelated (rho_1 ~ 0.005, variance inflation
~1.0), so sqrt(8760) on hourly returns and sqrt(365) on daily ones land
within a few percent of each other.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

TRIALS_LEDGER = Path(__file__).parent / "trials.json"

# The penalty counter tracks HOLDOUT consultations only. In-sample (train)
# backtests are unlimited — they can only overfit the train window, and the
# Critic knows they are in-sample. What must stay scarce is information about
# the held-out window: every look at it is recorded and raises the bar.

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 365.0 * 24.0          # 1h epochs; the ledger stores annualized


# --- trial ledger -----------------------------------------------------------

def _load_ledger() -> dict:
    if TRIALS_LEDGER.exists():
        raw = json.loads(TRIALS_LEDGER.read_text())
        if isinstance(raw, dict) and "n_trials" in raw:
            return {"n_trials": int(raw["n_trials"]),
                    "sharpes": [float(x) for x in raw.get("sharpes", [])]}
    return {"n_trials": 0, "sharpes": []}


def record_trial(sharpe_ann: float | None = None) -> int:
    """Increment and persist the global trial counter, remembering the
    trial's annualized Sharpe. Called once per evaluation-window backtest,
    before the score is computed. The ledger is append-only from the agents'
    perspective — they cannot reset it. The recorded Sharpes are what gives
    the deflation its V[{SR_n}]; a ledger written before this field existed
    (or a trial whose Sharpe was not supplied) falls back to the null
    variance of a zero-true-Sharpe estimate, 1/(T-1) per observation."""
    led = _load_ledger()
    led["n_trials"] += 1
    if sharpe_ann is not None and math.isfinite(sharpe_ann):
        led["sharpes"].append(float(sharpe_ann))
    TRIALS_LEDGER.write_text(json.dumps(led))
    return led["n_trials"]


def trial_sharpe_variance() -> float | None:
    """Sample variance of the annualized trial Sharpes on record, or None
    when fewer than two have been recorded."""
    s = _load_ledger()["sharpes"]
    if len(s) < 2:
        return None
    mean = sum(s) / len(s)
    return sum((x - mean) ** 2 for x in s) / (len(s) - 1)


# --- normal distribution helpers (stdlib only, no SciPy here) ---------------

def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF — Acklam's rational approximation,
    |error| < 1.2e-9 over the whole open interval."""
    if not 0.0 < p < 1.0:
        raise ValueError("_norm_ppf needs 0 < p < 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


# --- the score --------------------------------------------------------------

def deflated_sharpe(sr: float, t: int | float, skew: float = 0.0,
                    kurtosis: float = 3.0, n_trials: int = 1,
                    var_sr_ann: float | None = None,
                    periods_per_year: float = PERIODS_PER_YEAR) -> dict:
    """Bailey & Lopez de Prado's deflated Sharpe.

    sr          per-observation Sharpe (NOT annualized)
    t           number of return observations it was estimated from
    skew        skewness of those returns
    kurtosis    non-excess kurtosis of those returns (normal == 3)
    n_trials    charged looks at the evaluation window (N)
    var_sr_ann  variance of the annualized trial Sharpes, V[{SR_n}]; None
                falls back to the null variance of a zero-true-Sharpe
                estimate, 1/(T-1) per observation, so a first trial still
                faces a non-zero bar as soon as N > 1.

    Returns the bar (sr_threshold), sr - bar, and DSR — the probability the
    true Sharpe exceeds zero — per observation, with annualized forms
    alongside for reporting.
    """
    t = max(float(t), 2.0)
    if n_trials < 2:
        sr0 = 0.0                     # one draw: its own expectation is the bar
    else:
        v = var_sr_ann / periods_per_year if var_sr_ann and var_sr_ann > 0 \
            else 1.0 / (t - 1.0)
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr
                          + (kurtosis - 1.0) / 4.0 * sr * sr))
    dsr = _norm_cdf((sr - sr0) * math.sqrt(t - 1.0) / denom)
    ann = math.sqrt(periods_per_year)
    return {
        "sr_epoch": sr,
        "sr_threshold_epoch": sr0,
        "sr_excess_epoch": sr - sr0,
        "ann_sharpe": sr * ann,
        "ann_sharpe_threshold": sr0 * ann,
        "ann_sharpe_excess": (sr - sr0) * ann,
        "dsr": dsr,
        "n_returns": int(t),
        "n_trials": n_trials,
        "var_sr_ann": var_sr_ann,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def evaluate_xsec(report_row: dict, lambda_: float = 1.0,
                  in_sample: bool = False) -> dict:
    """Score one cross-sectional backtest (a row of qse_xsec's report JSON).

    in_sample=True  — train-window run: raw numbers and the bar it would have
                      faced, but the trial ledger is read, NOT charged.
                      Marked so the Critic treats it as an in-sample fit.
    in_sample=False — holdout run: increments the global trial counter and
                      applies the deflation. The only score that promotes.

    The gate is on the Sharpe; the IC-side numbers (mean_ic, its t-stat, and
    the old IC deflation) are diagnostics for the Critic, not criteria —
    a cash-flow factor has no IC to show.
    """
    t_ic = max(int(report_row.get("n_epochs", 0)), 2)
    mean_ic = float(report_row["mean_ic"])
    ic_std = float(report_row["ic_std"])

    # Return-series side. `sharpe_epoch`, `n_returns`, `skew` and `kurtosis`
    # are written by qse_xsec; older rows only carry the annualized Sharpe.
    ann = float(report_row.get("ann_sharpe_net", 0.0) or 0.0)
    sr_epoch = report_row.get("sharpe_epoch")
    if sr_epoch is None:
        sr_epoch = ann / math.sqrt(PERIODS_PER_YEAR)
    t_ret = int(report_row.get("n_returns") or t_ic)
    skew = float(report_row.get("skew", 0.0) or 0.0)
    kurt = float(report_row.get("kurtosis", 3.0) or 3.0)

    var_sr = trial_sharpe_variance()
    if in_sample:
        n = None
        sharpe = deflated_sharpe(sr_epoch, t_ret, skew, kurt,
                                 max(_load_ledger()["n_trials"], 1), var_sr)
        adj_ic = mean_ic
    else:
        n = record_trial(ann)
        sharpe = deflated_sharpe(sr_epoch, t_ret, skew, kurt, n, var_sr)
        # IC deflation, diagnostics only: T here is the measured epoch count,
        # because each epoch's IC noise comes from that epoch's own return
        # draw (measured IC-series VIF 0.63..1.37, not 24).
        adj_ic = mean_ic - lambda_ * ic_std * math.sqrt(
            2.0 * math.log(max(n, 1)) / t_ic)
    return {
        "window": "train (in-sample)" if in_sample else "holdout (out-of-sample)",
        "n_trials_global": n,
        # --- the score ---
        "ann_sharpe_net": ann,
        "sharpe_epoch": sharpe["sr_epoch"],
        "sharpe_threshold_ann": sharpe["ann_sharpe_threshold"],
        "sharpe_excess_ann": sharpe["ann_sharpe_excess"],
        "dsr": sharpe["dsr"],
        "n_returns": t_ret,
        "n_bets": report_row.get("n_bets"),
        "rebalance_every": report_row.get("rebalance_every"),
        "skew": skew,
        "kurtosis": kurt,
        "total_return_net": report_row.get("total_return_net"),
        "max_drawdown": report_row.get("max_drawdown"),
        "avg_turnover": report_row.get("avg_turnover"),
        "n_epochs": report_row.get("n_epochs"),
        # --- diagnostics, not gates ---
        "mean_ic": mean_ic,
        "ic_tstat": report_row.get("ic_tstat"),
        "adjusted_ic": adj_ic,
        "penalty_ic": mean_ic - adj_ic,
    }
