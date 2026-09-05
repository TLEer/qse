"""Structured-output schemas for the Proposer → Implementer → Critic loop.

Each agent's reply is parsed with client.messages.parse(), so malformed
output is a hard error, not a silent drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class XsecFactorHypothesis(BaseModel):
    """Proposer output: a cross-sectional ranking hypothesis."""
    name: str = Field(description="snake_case identifier, e.g. funding_momentum_24h")
    economic_rationale: str = Field(
        description="Why a cross-sectional SPREAD in this quantity predicts "
        "the spread in next-hour returns across the 30-asset universe. "
        "Hypotheses justified only by 'it fit the data' are rejected."
    )
    formula: str = Field(
        description="Pseudocode using ONLY: xs_rank, xs_zscore, xs_demean, "
        "clip, ts_mean, ts_std, ts_sum, ts_delta, ts_min, ts_max, ts_ret, "
        "ts_ret_std, ts_corr over the fields open/high/low/close/volume/"
        "quote_volume/taker_buy_ratio/funding. Higher score = long candidate."
    )
    lookback_bars: int = Field(description="Largest window used, in 1h bars")
    expected_turnover: str = Field(
        description="one of: low | medium | high. The Critic checks this "
        "against realized per-epoch turnover."
    )
    predicted_failure_modes: list[str] = Field(
        description="Market conditions where the ranking should invert or "
        "decay. The Critic verifies these against the worst per-epoch ICs."
    )


class XsecFactorImplementation(BaseModel):
    """Implementer output: engine-ready IXsecFactor C++."""
    factor_name: str
    cpp_source: str = Field(
        description="Complete .cpp implementing qse::xsec::IXsecFactor, "
        "compiled into factors/xsec_generated/, ending with "
        "QSE_REGISTER_XFACTOR(<factor_name>, <ClassName>). No I/O, no "
        "clocks, no randomness — the engine rejects nondeterminism via "
        "digest comparison."
    )
    warmup_bars: int
    notes: str


class Critique(BaseModel):
    """Critic output: verdict on a backtest report."""
    verdict: str = Field(description="one of: promote | revise | reject")
    statistical_flaws: list[str] = Field(
        description="e.g. IC t-stat driven by a few epochs, penalty exceeds "
        "raw IC, turnover contradicts the stated expectation."
    )
    logical_flaws: list[str] = Field(
        description="e.g. hypothesis predicts mean-reversion but PnL comes "
        "from trend segments; worst epochs contradict stated failure modes."
    )
    revision_directive: str = Field(
        description="Concrete instruction sent back to the Proposer. Must "
        "reference specific evidence, not just aggregate stats."
    )
