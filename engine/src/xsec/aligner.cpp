#include "qse/xsec/aligner.hpp"

#include <algorithm>
#include <stdexcept>

namespace qse::xsec {

EpochAligner::EpochAligner(std::vector<std::string> symbols, TsNs epoch_ns,
                           TsNs start_open, TsNs end_open)
    : start_open_(start_open) {
    if (epoch_ns <= 0) throw std::invalid_argument("epoch_ns must be positive");
    if (end_open < start_open || (end_open - start_open) % epoch_ns != 0)
        throw std::invalid_argument("span must be a whole number of epochs");

    const auto T = static_cast<std::size_t>((end_open - start_open) / epoch_ns) + 1;
    const auto N = symbols.size();

    grid_.symbols = std::move(symbols);
    grid_.epoch_ns = epoch_ns;
    grid_.epoch_close.resize(T);
    for (std::size_t t = 0; t < T; ++t)
        grid_.epoch_close[t] = start_open + static_cast<TsNs>(t + 1) * epoch_ns;

    for (auto* v : {&grid_.open, &grid_.high, &grid_.low, &grid_.close,
                    &grid_.volume, &grid_.quote_volume, &grid_.taker_buy_ratio,
                    &grid_.funding_level, &grid_.funding_paid})
        v->assign(T * N, kNaN);
    grid_.valid.assign(T * N, 0);
    grid_.staleness.assign(T * N, 0);

    watermark_.assign(N, -1);
    bar_written_.assign(T * N, 0);
    last_close_.assign(N, kNaN);
    stale_count_.assign(N, 0);
    funding_last_.assign(N, kNaN);
    funding_pending_.resize(N);
}

void EpochAligner::append(std::size_t sym, const Kline& k, double taker_buy_volume) {
    const auto N = grid_.N();
    const auto T = grid_.T();
    if (sym >= N) throw std::out_of_range("append: bad symbol index");

    const TsNs off = k.open_time - start_open_;
    if (off % grid_.epoch_ns != 0)
        throw std::runtime_error("append: bar open_time off the epoch grid");
    const std::ptrdiff_t e = off / grid_.epoch_ns;

    // Before the span: ignore, as documented — must precede the watermark
    // check, or a fresh aligner (watermark -1) would reject pre-span bars
    // as "regressions" (the live warmup feeds curated months before the
    // session's span start).
    if (e < 0) return;

    if (e <= watermark_[sym])
        throw std::runtime_error("append: bar regresses the symbol watermark (" +
                                 grid_.symbols[sym] + " @" +
                                 std::to_string(k.open_time) + ")");
    if (e < static_cast<std::ptrdiff_t>(locked_))
        throw std::runtime_error("append: bar lands in a locked epoch (" +
                                 grid_.symbols[sym] + " @" +
                                 std::to_string(k.open_time) + ")");

    if (e >= static_cast<std::ptrdiff_t>(T)) {
        // Past the span: proves every in-span epoch complete for this symbol.
        watermark_[sym] = static_cast<std::ptrdiff_t>(T) - 1;
        try_lock_rows_();
        return;
    }

    const auto idx = static_cast<std::size_t>(e) * N + sym;
    grid_.open[idx] = k.open;
    grid_.high[idx] = k.high;
    grid_.low[idx] = k.low;
    grid_.close[idx] = k.close;
    grid_.volume[idx] = k.volume;
    grid_.quote_volume[idx] = k.quote_volume;
    grid_.taker_buy_ratio[idx] =
        (k.volume > 0.0 && std::isfinite(taker_buy_volume))
            ? taker_buy_volume / k.volume
            : 0.5;
    bar_written_[idx] = 1;
    watermark_[sym] = e;
    try_lock_rows_();
}

void EpochAligner::append_funding(std::size_t sym, TsNs calc_ts, double rate) {
    if (sym >= grid_.N()) throw std::out_of_range("append_funding: bad symbol index");
    auto& q = funding_pending_[sym];
    if (!q.empty() && calc_ts < q.back().ts)
        throw std::runtime_error("append_funding: prints out of order");
    if (locked_ > 0 && calc_ts <= grid_.epoch_close[locked_ - 1])
        throw std::runtime_error("append_funding: print lands in a locked epoch");
    q.push_back({calc_ts, rate});
}

void EpochAligner::seal() {
    std::fill(watermark_.begin(), watermark_.end(),
              static_cast<std::ptrdiff_t>(grid_.T()) - 1);
    try_lock_rows_();
}

void EpochAligner::try_lock_rows_() {
    if (grid_.N() == 0) return;
    const auto barrier = *std::min_element(watermark_.begin(), watermark_.end());
    while (static_cast<std::ptrdiff_t>(locked_) <= barrier) {
        lock_row_(locked_);
        ++locked_;
    }
}

void EpochAligner::lock_row_(std::size_t t) {
    const auto N = grid_.N();
    const TsNs row_open = start_open_ + static_cast<TsNs>(t) * grid_.epoch_ns;
    const TsNs row_close = grid_.epoch_close[t];

    for (std::size_t n = 0; n < N; ++n) {
        const auto idx = t * N + n;

        if (bar_written_[idx]) {
            grid_.valid[idx] = 1;
            grid_.staleness[idx] = 0;
            last_close_[n] = grid_.close[idx];
            stale_count_[n] = 0;
        } else if (std::isfinite(last_close_[n])) {
            // Forward-fill: carry the last real close, flag zero activity.
            grid_.open[idx] = grid_.high[idx] = grid_.low[idx] = grid_.close[idx] =
                last_close_[n];
            grid_.volume[idx] = 0.0;
            grid_.quote_volume[idx] = 0.0;
            grid_.taker_buy_ratio[idx] = 0.5;
            grid_.valid[idx] = 0;
            grid_.staleness[idx] = ++stale_count_[n];
        }
        // else: pre-listing — stays NaN, valid 0, staleness 0.

        // Funding: consume prints knowable by this row's close; a print inside
        // (row_open, row_close] accrues as paid this epoch.
        auto& q = funding_pending_[n];
        double paid = 0.0;
        while (!q.empty() && q.front().ts <= row_close) {
            funding_last_[n] = q.front().rate;
            if (q.front().ts > row_open) paid += q.front().rate;
            q.pop_front();
        }
        grid_.funding_level[idx] = funding_last_[n];
        grid_.funding_paid[idx] = std::isfinite(funding_last_[n]) ? paid : kNaN;
    }
}

}  // namespace qse::xsec
