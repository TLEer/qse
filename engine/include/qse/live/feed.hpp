#pragma once

// Live market-data feed for the paper-trading engine.
//
// BinanceFeed pulls REAL USDT-M futures production data (public REST, no
// key): 1h klines and funding-rate prints per symbol. The engine appends
// whatever the feed returns straight into the EpochAligner, so the feed is
// the first line of defence against look-ahead: it drops the forming kline
// (close_time >= now), drops off-grid bars, and only ever emits ascending
// bars per symbol. Network hiccups surface as per-symbol errors — the
// engine logs them and lets the aligner's forward-fill take over; a later
// poll re-fetches the missed span (limit > 1) and self-heals.
//
// Replay mode (--replay) substitutes a CsvFeed driven off a prebuilt grid,
// so the entire pipeline below the feed is testable offline.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "qse/events.hpp"

namespace qse::live {

inline constexpr double kLiveNaN = std::numeric_limits<double>::quiet_NaN();

struct FundingPrint {
    TsNs calc_ts{};   // Binance fundingTime, ns since epoch UTC
    double rate{};    // raw funding rate (0.0001 = 1bp), as printed
};

// A finalized kline plus the taker-buy volume the aligner wants (it is not
// part of qse::Kline).
struct FeedKline {
    qse::Kline k;
    double taker_buy = kLiveNaN;
};

// Everything one poll produced, per symbol (index == universe index).
struct FeedEvents {
    std::vector<std::vector<FeedKline>> klines;     // ascending, finalized
    std::vector<std::vector<FundingPrint>> funding; // ascending
    std::vector<std::string> errors;                // "" = ok, else msg
};

class IFeed {
public:
    virtual ~IFeed() = default;
    // Fetch any new events since the last poll/backfill. `now_ms` is the
    // caller's wall clock (UTC ms). On failure of some symbols, their
    // errors[] entries are set and the events for the others are still
    // returned; returns false only if every symbol failed.
    virtual bool poll(std::int64_t now_ms, FeedEvents& out, std::string& err) = 0;
    // Live feeds only: page-fill every symbol from its own start point (the
    // per-symbol warm watermarks) up to now. Events are delivered through
    // `out` exactly like a poll — the engine appends them via ingest_().
    // Returns false when unsupported (replay feeds) or when the backfill
    // was partial (out.errors[] names the failures).
    virtual bool backfill(const std::vector<TsNs>& after_ns, FeedEvents& out,
                          std::string& err) {
        (void)after_ns;
        out = FeedEvents{};
        err = "backfill not supported";
        return false;
    }
    // Live feeds only: server-clock offset (ms) for diagnostics.
    virtual bool server_time_offset(std::int64_t& offset_ms, std::string& err) {
        (void)offset_ms;
        err = "server time not supported";
        return false;
    }
};

// Production feed over fapi.binance.com. One instance per session; owns the
// per-symbol dedup cursors (last emitted kline open / fundingTime), so
// calling poll repeatedly never re-emits data and never goes backwards.
class BinanceFeed final : public IFeed {
public:
    // symbols: universe in pinned order (load_universe()). base_url is the
    // production REST root, e.g. "https://fapi.binance.com".
    BinanceFeed(std::vector<std::string> symbols, std::string base_url,
                int timeout_ms);
    ~BinanceFeed() override = default;

    // Page-fill every symbol from its warm watermark (the curated-CSV
    // coverage per symbol) up to the current time. Idempotent; advances the
    // cursors. Events are delivered through `out` like a poll — the engine
    // appends them via ingest_().
    bool backfill(const std::vector<TsNs>& after_ns, FeedEvents& out,
                  std::string& err) override;

    // Incremental poll: up to a few recent klines / funding prints per
    // symbol, filtered to finalized, in-span, ascending events.
    bool poll(std::int64_t now_ms, FeedEvents& out, std::string& err) override;

    // Server-clock offset (ms) for diagnostics: server - local.
    bool server_time_offset(std::int64_t& offset_ms, std::string& err);

private:
    std::vector<std::string> symbols_;
    std::string base_url_;
    int timeout_ms_;
    std::vector<TsNs> last_open_ns_;    // last emitted kline open_time
    std::vector<TsNs> last_funding_ns_; // last emitted fundingTime
};

// Body-only parsers, exposed for selftests (no network). `out` receives
// ascending events. Returns false on malformed JSON/fields (out cleared).
// parse_klines_body drops the forming bar (close_time >= now_ms) and bars
// whose open_time is off the 1h grid, and rejects non-finite prices.
bool parse_klines_body(const std::string& body, std::int64_t now_ms,
                       std::vector<FeedKline>& out);
bool parse_funding_body(const std::string& body,
                        std::vector<FundingPrint>& out);

}  // namespace qse::live
