#pragma once

// Cross-sectional state grid: the aligned T x N block every xsec factor
// computes on. Rows are 1h epochs, columns are symbols; storage is row-major
// contiguous (index = t * N + n) so cross-sections are cache-linear.
//
// Look-ahead defence mirrors the single-asset engine's philosophy: factors
// never touch MarketGrid directly — they receive a GridView pinned to the
// last locked row, and every accessor clamps to that row. Reading the future
// is not expressible through the type.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "qse/events.hpp"

namespace qse::xsec {

enum class Field : int {
    Open = 0, High, Low, Close, Volume, QuoteVolume,
    TakerBuyRatio,   // taker_buy_volume / volume (0.5 on ffilled rows)
    FundingLevel,    // last printed funding rate, forward-filled (signal)
    FundingPaid,     // rate paid in this epoch's accrual window, else 0 (PnL)
};

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Immutable once the aligner seals it. All field vectors are T*N, row-major.
struct MarketGrid {
    std::vector<std::string> symbols;   // size N
    std::vector<TsNs> epoch_close;      // size T, strictly +epoch_ns steps
    TsNs epoch_ns{};

    std::vector<double> open, high, low, close, volume, quote_volume,
                        taker_buy_ratio, funding_level, funding_paid;
    std::vector<std::uint8_t> valid;        // 1 = real bar, 0 = ffilled/absent
    std::vector<std::uint16_t> staleness;   // consecutive ffilled rows, 0 if real

    std::size_t T() const { return epoch_close.size(); }
    std::size_t N() const { return symbols.size(); }

    const std::vector<double>& field(Field f) const {
        switch (f) {
            case Field::Open: return open;
            case Field::High: return high;
            case Field::Low: return low;
            case Field::Close: return close;
            case Field::Volume: return volume;
            case Field::QuoteVolume: return quote_volume;
            case Field::TakerBuyRatio: return taker_buy_ratio;
            case Field::FundingLevel: return funding_level;
            case Field::FundingPaid: return funding_paid;
        }
        return close;  // unreachable
    }
};

// A factor's only window onto the data. `t` is the last locked row; reads are
// addressed as `r` rows back from it (r = 0 is the just-closed epoch). Any
// read past available history returns NaN — never throws, never wraps.
class GridView {
public:
    GridView(const MarketGrid& g, std::size_t t) : g_(g), t_(t) {}

    std::size_t n_symbols() const { return g_.N(); }
    std::size_t history() const { return t_ + 1; }   // rows available
    TsNs now() const { return g_.epoch_close[t_]; }

    double get(Field f, std::size_t n, std::size_t r = 0) const {
        if (r > t_ || n >= g_.N()) return kNaN;
        return g_.field(f)[(t_ - r) * g_.N() + n];
    }
    double open(std::size_t n, std::size_t r = 0) const  { return get(Field::Open, n, r); }
    double high(std::size_t n, std::size_t r = 0) const  { return get(Field::High, n, r); }
    double low(std::size_t n, std::size_t r = 0) const   { return get(Field::Low, n, r); }
    double close(std::size_t n, std::size_t r = 0) const { return get(Field::Close, n, r); }
    double volume(std::size_t n, std::size_t r = 0) const { return get(Field::Volume, n, r); }
    double quote_volume(std::size_t n, std::size_t r = 0) const { return get(Field::QuoteVolume, n, r); }
    double taker_buy_ratio(std::size_t n, std::size_t r = 0) const { return get(Field::TakerBuyRatio, n, r); }
    double funding(std::size_t n, std::size_t r = 0) const { return get(Field::FundingLevel, n, r); }

    bool is_valid(std::size_t n, std::size_t r = 0) const {
        if (r > t_ || n >= g_.N()) return false;
        return g_.valid[(t_ - r) * g_.N() + n] != 0;
    }

    // close(n, 0) / close(n, lookback) - 1; NaN if either leg is missing.
    double ret(std::size_t n, std::size_t lookback) const {
        const double a = close(n, 0), b = close(n, lookback);
        if (!std::isfinite(a) || !std::isfinite(b) || b == 0.0) return kNaN;
        return a / b - 1.0;
    }

private:
    const MarketGrid& g_;
    std::size_t t_;
};

}  // namespace qse::xsec
