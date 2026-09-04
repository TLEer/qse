#pragma once

// EpochAligner — the Time-Alignment Buffer.
//
// Ingests per-symbol klines (each symbol time-ordered; interleaving across
// symbols is arbitrary) and produces the aligned MarketGrid on a strict epoch
// grid. Row e is emitted only once EVERY symbol's watermark has passed e —
// either a real bar for epoch e arrived, or a later bar / seal() proved the
// bar absent, in which case the row is forward-filled (close carried,
// volume 0, valid 0, staleness incremented).
//
// Locked rows are physically immutable: appending a bar at or before the
// locked watermark throws. This is the hard guarantee that T_{n+1} data can
// never bleed into a T_n computation.
//
// Historical mode: append all funding, then all klines, then seal(). The
// same append() surface accepts streaming bars later (funding prints arrive
// before the kline that closes their epoch, which live feeds satisfy).

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "qse/xsec/grid.hpp"

namespace qse::xsec {

class EpochAligner {
public:
    // Grid spans epochs whose open times are start_open, start_open+epoch_ns,
    // ..., up to and including end_open. Bars outside the span are ignored
    // (a bar after end_open still advances the symbol's watermark to the end:
    // it proves every in-span epoch is complete for that symbol).
    EpochAligner(std::vector<std::string> symbols, TsNs epoch_ns,
                 TsNs start_open, TsNs end_open);

    // Bar open_time must land exactly on the epoch grid and be strictly newer
    // than the symbol's previous bar; throws otherwise. taker_buy_volume is
    // the UM kline column absent from qse::Kline; NaN means unknown.
    void append(std::size_t sym, const Kline& k, double taker_buy_volume = kNaN);

    // Funding prints must be per-symbol time-ordered. A print for an
    // already-locked epoch throws.
    void append_funding(std::size_t sym, TsNs calc_ts, double rate);

    // Declare end-of-data for all symbols: every remaining row locks (missing
    // bars forward-filled through the end of the span).
    void seal();

    std::size_t locked_rows() const { return locked_; }   // barrier position
    const MarketGrid& grid() const { return grid_; }      // rows [0, locked_)

private:
    void try_lock_rows_();
    void lock_row_(std::size_t t);

    MarketGrid grid_;
    TsNs start_open_{};
    std::size_t locked_ = 0;

    // Per symbol: highest epoch index proven complete (-1 = none).
    std::vector<std::ptrdiff_t> watermark_;
    std::vector<std::uint8_t> bar_written_;          // T*N, real bar present
    std::vector<double> last_close_;                 // ffill source, per symbol
    std::vector<std::uint16_t> stale_count_;         // per symbol
    std::vector<double> funding_last_;               // last rate seen, per symbol
    struct FundingPrint { TsNs ts; double rate; };
    std::vector<std::deque<FundingPrint>> funding_pending_;   // per symbol
};

}  // namespace qse::xsec
