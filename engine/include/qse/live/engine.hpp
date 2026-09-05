#pragma once

// Live paper-trading engine: drives the EpochAligner from a feed, evaluates
// the factor once per locked epoch, builds the book via the shared
// build_book(), places orders through a broker, and finalizes PnL/IC with
// run_xsec's exact accounting (same order of operations, same digest fold)
// so replay mode is digest-comparable with the backtest.
//
// Two modes:
//   - live:  warm from curated CSVs, backfill the gap via REST, then poll
//            the feed every poll_sec until SIGINT or the horizon ends.
//   - replay: CsvFeed replays a prebuilt grid row by row into a fresh
//            aligner with no wall clock and no network; fills are simulated
//            at the model price. The resulting model digest must equal
//            run_xsec's digest on the same grid — the equivalence selftest.
//
// Decision-time semantics: the book for epoch t is built when row t locks
// (which happens at close_t ≈ open[t+1] + poll latency) and the orders are
// placed immediately — the live analogue of the backtest's fill at
// open[t+1]. Only row t's validity is knowable at that instant, so the
// tradability mask is valid[t] alone; the backtest additionally excludes
// names whose execution bar (t+1) is forward-filled, a distinction live
// trading cannot make in advance. Model PnL is always charged on the grid's
// open[t+1] → open[t+2] return, exactly like the backtest.

#include <cstddef>
#include <cstdint>
#include <csignal>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "qse/live/broker.hpp"
#include "qse/live/feed.hpp"
#include "qse/xsec/aligner.hpp"
#include "qse/xsec/factor.hpp"
#include "qse/xsec/portfolio.hpp"

namespace qse::live {

struct LiveConfig {
    std::string data_dir;              // curated UM dir (live warmup)
    std::string universe_path;
    std::string factor_name = "funding_carry_168h";
    qse::xsec::PortfolioConfig pf;     // rebalance_every=24, q_out=0.35 live
    int horizon_days = 365;
    int poll_sec = 12;
    double capital = 0.0;              // 0 = venue balance (live), 100k (replay)
    bool no_orders = false;            // NullBroker dry run
    bool replay = false;
    std::string start_ym, end_ym;      // replay span, "YYYY-MM"
    std::string log_dir = "reports/live";
};

// Replay feed: replays a prebuilt grid row by row into the engine's fresh
// aligner. One poll = one row. Bars are emitted only where the source grid
// has a real bar (valid == 1) — gaps are left to the aligner's forward-fill,
// exactly like a live feed. Funding prints are reconstructed from the grid's
// funding_level / funding_paid columns: a print lands in the row whose
// window accrued it, at close_t (the row's last knowable instant).
class CsvFeed final : public IFeed {
public:
    explicit CsvFeed(const qse::xsec::MarketGrid& g);
    bool poll(std::int64_t now_ms, FeedEvents& out, std::string& err) override;

private:
    const qse::xsec::MarketGrid& g_;
    const std::size_t n_;
    const qse::TsNs start_open_;
    std::size_t next_row_ = 0;
    bool done_ = false;
};

// A placed book awaiting finalization (row t+2 locked).
struct Pending {
    std::size_t t{};
    std::vector<double> scores;   // masked scores (IC is measured on these)
    std::vector<double> w;        // book weights (PnL is charged on these)
    double turnover{};
    std::vector<FillReport> fills;
    std::vector<std::string> errors;
};

// One finalized epoch, serialized to the JSONL log.
struct LiveEpoch {
    std::size_t t{};
    qse::TsNs ts{};
    double ic = kLiveNaN;
    double pnl_gross{}, cost{}, funding_pnl{}, pnl_net{}, turnover{};
    std::size_t n_valid{};
    double model_equity{};
    double venue_balance = kLiveNaN;
    double venue_unrealized = kLiveNaN;
    std::vector<FillReport> fills;
    std::vector<std::string> errors;
};

class LiveEngine {
public:
    // feed/broker injected: live mode uses BinanceFeed + TestnetBroker (or
    // NullBroker with --no-orders); replay uses CsvFeed + SimBroker and
    // must pass the replay grid. `stop` is an optional signal flag the
    // caller's SIGINT handler sets; live mode polls it between ticks.
    LiveEngine(LiveConfig cfg, std::unique_ptr<IFeed> feed,
               std::unique_ptr<IBroker> broker,
               const qse::xsec::MarketGrid* replay_grid = nullptr,
               volatile std::sig_atomic_t* stop = nullptr);
    ~LiveEngine();

    // 0 = clean exit, 1 = fatal, 2 = usage error.
    int run();

    std::uint64_t model_digest() const { return digest_; }
    double model_equity() const { return equity_; }

private:
    int run_live_();
    int run_replay_();
    int fail_(const std::string& what);
    void load_rules_();
    void heartbeat_(std::int64_t now_ms);
    void tick_(std::int64_t now_ms);
    void ingest_(const FeedEvents& ev);
    void on_rows_locked_();
    void place_orders_(std::size_t t, const qse::xsec::Book& book,
                       std::vector<FillReport>& fills,
                       std::vector<std::string>& errors);
    void finalize_epoch_(const struct Pending& p);
    void log_epoch_(const LiveEpoch& rec);
    void write_summary_();
    void account_once_(std::string& err);

    LiveConfig cfg_;
    std::unique_ptr<IFeed> feed_;
    std::unique_ptr<IBroker> broker_;
    std::unique_ptr<qse::xsec::EpochAligner> aligner_;
    std::unique_ptr<qse::xsec::IXsecFactor> factor_;
    std::size_t warmup_{};
    std::size_t next_epoch_{};              // first not-yet-decided row
    std::vector<double> cur_w_, scores_;    // carried book state
    std::vector<int> cur_side_;
    std::vector<SymbolRules> rules_;
    std::vector<double> venue_qty_;         // signed qty from fills
    std::vector<int> venue_unknown_;        // 1 = reconcile from positions
    std::vector<int> fetch_fail_;           // consecutive feed errors/sym
    std::vector<int> late_funding_warned_;  // per-symbol warn-once flag
    int tick_fail_ = 0;                     // consecutive whole-tick outages
    std::int64_t last_status_ms_ = 0;       // heartbeat pacing (60s)
    std::deque<struct Pending> pending_;
    double nav_ = 0.0;
    std::uint64_t digest_ = 1469598103934665603ULL;   // FNV offset basis
    double equity_ = 1.0;
    std::vector<LiveEpoch> recs_;
    std::string exit_reason_ = "clean exit";
    std::string fatal_err_;
    volatile std::sig_atomic_t* stop_ = nullptr;
    FILE* jsonl_ = nullptr;
    std::string start_ym_, end_ym_;         // warm months used (live)
};

}  // namespace qse::live
