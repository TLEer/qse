"""Fitness evaluation for LLM-generated cross-sectional factors.

The core defence against specification-gaming: every backtest the agent runs
against the evaluation window increments a global trial counter N, and the
score it optimises is *deflated* by the expected maximum a zero-skill
strategy would reach after N tries (extreme-value bound):

    IC_adj = mean_ic - lambda_ * ic_std * sqrt(2 * ln(N) / T)

Random trial-and-error raises N without raising expected IC, so grinding the
backtest monotonically lowers the bar-adjusted score.
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


def _load_trial_count() -> int:
    if TRIALS_LEDGER.exists():
        return json.loads(TRIALS_LEDGER.read_text())["n_trials"]
    return 0


def record_trial() -> int:
    """Increment and persist the global trial counter. Called once per
    evaluation-window backtest, before the score is computed. The ledger is
    append-only from the agents' perspective — they cannot reset it."""
    n = _load_trial_count() + 1
    TRIALS_LEDGER.write_text(json.dumps({"n_trials": n}))
    return n


def evaluate_xsec(report_row: dict, lambda_: float = 1.0,
                  in_sample: bool = False) -> dict:
    """Score one cross-sectional backtest (a row of qse_xsec's report JSON).

    in_sample=True  — train-window run: raw mean IC only, no penalty, does
                      NOT touch the trial ledger. Marked so the Critic treats
                      the number as an in-sample fit, not evidence.
    in_sample=False — holdout run: increments the global trial counter and
                      applies the deflated-IC penalty. This is the only score
                      that can promote a factor.
    """
    t = max(int(report_row.get("n_epochs", 0)), 2)
    mean_ic = float(report_row["mean_ic"])
    ic_std = float(report_row["ic_std"])
    if in_sample:
        n, adj = None, mean_ic
    else:
        n = record_trial()
        adj = mean_ic - lambda_ * ic_std * math.sqrt(2.0 * math.log(max(n, 1)) / t)
    return {
        "window": "train (in-sample)" if in_sample else "holdout (out-of-sample)",
        "n_trials_global": n,
        "mean_ic": mean_ic,
        "adjusted_ic": adj,
        "penalty": mean_ic - adj,
        "ic_tstat": report_row.get("ic_tstat"),
        "ann_sharpe_net": report_row.get("ann_sharpe_net"),
        "total_return_net": report_row.get("total_return_net"),
        "max_drawdown": report_row.get("max_drawdown"),
        "avg_turnover": report_row.get("avg_turnover"),
        "n_epochs": report_row.get("n_epochs"),
    }
