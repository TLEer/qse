#pragma once

// Dollar-neutral rank long/short portfolio over the aligned grid.
//
// Per epoch t (last locked row): score → rank → book membership with
// hysteresis → per-side equal weights. Execution happens at the NEXT bar's
// open; holding-period return is open[t+1] → open[t+2] (the same tradable
// return the rank IC is measured against). Costs are charged on traded
// weight deltas only — an unchanged position costs nothing, which is what
// the hysteresis band buys.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "qse/xsec/factor.hpp"
#include "qse/xsec/grid.hpp"

namespace qse::xsec {

struct PortfolioConfig {
    double fee_bps = 5.0;        // taker fee per side, on traded delta
    double slip_bps = 3.0;       // slippage per side, on traded delta
    double gross = 1.0;          // |longs| + |shorts| as fraction of NAV
    double q_in = 0.20;          // entry quantile (top/bottom 20%)
    double q_out = 0.30;         // exit quantile (hysteresis band)
    std::size_t min_names = 10;  // below this many valid names: go flat
    // Re-form the book only every K epochs (weights carried in between; the
    // rank IC is still measured every epoch). The main turnover lever.
    std::size_t rebalance_every = 1;
};

struct EpochRecord {
    TsNs ts{};              // close of the signal epoch
    double ic{};            // Spearman(score, fwd open-to-open return)
    double pnl_gross{};
    double cost{};
    double funding_pnl{};
    double pnl_net{};
    double turnover{};      // sum |w_t - w_{t-1}|
    std::size_t n_valid{};
};

struct XsecResult {
    std::vector<EpochRecord> epochs;
    std::vector<double> equity;   // NAV after each evaluated epoch, starts 1.0

    std::size_t n_epochs{};       // epochs with a finite IC
    double mean_ic{};
    double ic_std{};
    double ic_tstat{};            // mean / std * sqrt(n_epochs)
    double ic_ir{};               // mean / std (per epoch)
    double ann_sharpe_net{};
    double total_return_net{};
    double max_drawdown{};
    double avg_turnover{};
    std::uint64_t digest{};       // FNV-1a over equity + IC bit patterns
};

// FNV-1a fold of one double's bit pattern into the determinism digest.
// Exposed so the live adapter can fold its model-equity stream with the
// exact same routine run_xsec uses (replay equivalence relies on this).
void digest_add(std::uint64_t& h, double x);

// The per-epoch book construction shared by the backtest loop and the live
// adapter. `scores` must already carry the tradability mask (NaN = excluded)
// and be the raw (pre-rank) scores the IC is measured on. `epoch` is the
// absolute grid row index, `warmup` the factor's warmup rows. When the epoch
// is not a rebalance epoch, w/side are carried from prev_* (nothing trades).
// Ties in rank order break by symbol index — deterministic.
struct Book {
    std::vector<double> w;    // size N, dollar-neutral per side, 0 = flat
    std::vector<int> side;    // +1 long, -1 short, 0 flat
};

Book build_book(std::span<const double> scores, const std::vector<double>& prev_w,
                const std::vector<int>& prev_side, std::size_t epoch,
                std::size_t warmup, const PortfolioConfig& cfg);

XsecResult run_xsec(const MarketGrid& g, IXsecFactor& factor,
                    const PortfolioConfig& cfg);

}  // namespace qse::xsec
