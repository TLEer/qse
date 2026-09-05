#include "qse/live/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <unistd.h>

#include "qse/live/http_client.hpp"
#include "qse/xsec/ic.hpp"
#include "qse/xsec/loader.hpp"
#include "qse/xsec/ops.hpp"

namespace qse::live {

namespace {

constexpr qse::TsNs kHourNs = 3'600'000'000'000LL;
constexpr std::int64_t kHourMs = 3'600'000LL;
constexpr double kWMin = 1e-4;             // skip deltas below this weight
constexpr int kMaxAnomalies = 10;          // consecutive per-symbol fails

std::string ym_of(std::int64_t now_ms, int month_shift) {
    std::time_t sec = now_ms / 1000;
    std::tm tm{};
    gmtime_r(&sec, &tm);
    std::tm shifted = tm;
    shifted.tm_mon += month_shift;
    std::time_t t2 = timegm(&shifted);
    std::tm out{};
    gmtime_r(&t2, &out);
    const int y = std::clamp(out.tm_year + 1900, 1970, 9999);
    const int m = std::clamp(out.tm_mon + 1, 1, 12);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d-%02d", y, m);
    return buf;
}

void json_num(FILE* f, double v) {
    if (std::isfinite(v)) std::fprintf(f, "%.10g", v);
    else std::fputs("null", f);
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    return o;
}

}  // namespace

CsvFeed::CsvFeed(const qse::xsec::MarketGrid& g)
    : g_(g), n_(g.N()), start_open_(g.epoch_close[0] - g.epoch_ns) {}

bool CsvFeed::poll(std::int64_t, FeedEvents& out, std::string& err) {
    out.klines.assign(n_, {});
    out.funding.assign(n_, {});
    out.errors.assign(n_, {});
    if (done_) {
        err = "replay exhausted";
        return false;
    }
    const std::size_t t = next_row_++;
    if (next_row_ >= g_.T()) done_ = true;

    const qse::TsNs open_t = start_open_ + static_cast<qse::TsNs>(t) * g_.epoch_ns;
    const qse::TsNs close_t = g_.epoch_close[t];

    for (std::size_t n = 0; n < n_; ++n) {
        const std::size_t i = t * n_ + n;

        // Funding prints knowable by this row's close.
        const double lev = g_.funding_level[i];
        const double prev =
            (t == 0) ? kLiveNaN : g_.funding_level[(t - 1) * n_ + n];
        const double paid = g_.funding_paid[i];
        const bool changed =
            std::isfinite(lev) && (t == 0 || lev != prev);
        if (changed) {
            if (std::isfinite(paid) && paid != 0.0 && paid != lev)
                out.funding[n].push_back({close_t - 1, paid - lev});
            out.funding[n].push_back({close_t, lev});
        } else if (std::isfinite(paid) && paid != 0.0) {
            out.funding[n].push_back({close_t - 1, paid});
        }

        if (!g_.valid[i]) continue;   // ffilled/pre-listing: no bar
        FeedKline fk;
        fk.k.open_time = open_t;
        fk.k.close_time = close_t - 1;
        fk.k.open = g_.open[i];
        fk.k.high = g_.high[i];
        fk.k.low = g_.low[i];
        fk.k.close = g_.close[i];
        fk.k.volume = g_.volume[i];
        fk.k.quote_volume = g_.quote_volume[i];
        if (std::isfinite(g_.taker_buy_ratio[i]) && g_.volume[i] > 0.0)
            fk.taker_buy = g_.taker_buy_ratio[i] * g_.volume[i];
        out.klines[n].push_back(std::move(fk));
    }
    return true;
}

// ---------------------------------------------------------------------------
// LiveEngine
// ---------------------------------------------------------------------------

LiveEngine::LiveEngine(LiveConfig cfg, std::unique_ptr<IFeed> feed,
                       std::unique_ptr<IBroker> broker,
                       const qse::xsec::MarketGrid* replay_grid,
                       volatile std::sig_atomic_t* stop)
    : cfg_(std::move(cfg)), feed_(std::move(feed)), broker_(std::move(broker)),
      stop_(stop) {
    auto& registry = qse::xsec::xfactor_registry();
    const auto it = registry.find(cfg_.factor_name);
    if (it == registry.end()) {
        std::fprintf(stderr, "qse_live: unknown factor '%s'\n", cfg_.factor_name.c_str());
        throw std::runtime_error("unknown factor");
    }
    factor_ = it->second();
    warmup_ = std::max<std::size_t>(factor_->warmup(), 1);

    if (cfg_.replay) {
        if (!replay_grid) throw std::runtime_error("replay mode needs a grid");
        aligner_ = std::make_unique<qse::xsec::EpochAligner>(
            replay_grid->symbols, replay_grid->epoch_ns,
            replay_grid->epoch_close[0] - replay_grid->epoch_ns,
            replay_grid->epoch_close.back());
    } else {
        const std::int64_t now_ms = unix_ms();
        const std::int64_t hour_floor = now_ms - (now_ms % kHourMs);
        const qse::TsNs start_open =
            static_cast<qse::TsNs>(hour_floor) * 1'000'000LL -
            static_cast<qse::TsNs>(warmup_ + 48) * kHourNs;
        const qse::TsNs end_open =
            start_open + static_cast<qse::TsNs>(cfg_.horizon_days) * 24 * kHourNs;
        auto symbols = qse::xsec::load_universe(cfg_.universe_path);
        aligner_ = std::make_unique<qse::xsec::EpochAligner>(
            std::move(symbols), kHourNs, start_open, end_open);
        start_ym_ = ym_of(now_ms, -1);
        end_ym_ = ym_of(now_ms, 0);
    }

    const std::size_t N = aligner_->grid().N();
    cur_w_.assign(N, 0.0);
    cur_side_.assign(N, 0);
    scores_.assign(N, kLiveNaN);
    venue_qty_.assign(N, 0.0);
    venue_unknown_.assign(N, 0);
    fetch_fail_.assign(N, 0);
    late_funding_warned_.assign(N, 0);
    rules_.assign(N, SymbolRules{});
    // Replay has no venue balance: size orders with the default capital.
    nav_ = cfg_.capital > 0.0 ? cfg_.capital : (cfg_.replay ? 100000.0 : 0.0);
}

LiveEngine::~LiveEngine() {
    if (jsonl_) std::fclose(jsonl_);
}

int LiveEngine::run() {
    const std::string log_path = cfg_.log_dir + "/epochs.jsonl";
    if (std::FILE* f = std::fopen(log_path.c_str(), "a")) jsonl_ = f;

    if (cfg_.replay) {
        if (jsonl_)
            std::fprintf(stderr, "qse_live: replaying %zu epochs\n", aligner_->grid().T());
        load_rules_();          // SimBroker: static defaults
        int rc = run_replay_();
        write_summary_();
        return rc;
    }
    int rc = run_live_();
    write_summary_();
    return rc;
}

// Fail fast with the reason visible both on stderr and in the summary.
int LiveEngine::fail_(const std::string& what) {
    fatal_err_ = what;
    exit_reason_ = "fatal: " + what;
    std::fprintf(stderr, "qse_live: FATAL: %s\n", what.c_str());
    return 1;
}

// Populate the per-symbol exchange rules. Must run AFTER broker->connect()
// (the testnet cache is filled there); rules_ drives order sizing, so a
// symbol with no rules can never trade.
void LiveEngine::load_rules_() {
    const std::size_t N = aligner_->grid().N();
    std::size_t ok = 0;
    for (std::size_t n = 0; n < N; ++n) {
        if (broker_->rules(aligner_->grid().symbols[n], rules_[n])) {
            ++ok;
        } else if (jsonl_) {
            std::fprintf(stderr, "qse_live: no rules for %s\n",
                         aligner_->grid().symbols[n].c_str());
        }
    }
    std::fprintf(stderr, "qse_live: %zu/%zu symbols have exchange rules\n",
                 ok, N);
}

int LiveEngine::run_live_() {
    std::string err;
    if (jsonl_)
        std::fprintf(stderr,
                     "qse_live: warm %s..%s, factor %s, horizon %d days\n",
                     start_ym_.c_str(), end_ym_.c_str(),
                     cfg_.factor_name.c_str(), cfg_.horizon_days);

    // 1) Warm from curated CSVs. Rows lock but nothing evaluates yet — the
    //    broker must be ready before the backfill below, because every warm
    //    epoch is evaluated the moment its events are ingested, and those
    //    epochs' orders need real exchange rules.
    std::vector<qse::TsNs> warm_watermarks;
    try {
        warm_watermarks = qse::xsec::warm_aligner_from_csv(
            *aligner_, cfg_.data_dir, start_ym_, end_ym_);
    } catch (const std::exception& e) {
        return fail_("warmup failed: " + std::string(e.what()));
    }

    // 2) Broker first: connect, exchange rules, one-time symbol setup,
    //    capital.
    if (!broker_->connect(err)) {
        return fail_("broker connect: " + err);
    }
    load_rules_();
    if (!broker_->setup_symbols(aligner_->grid().symbols, err))
        std::fprintf(stderr,
                     "qse_live: WARNING setup_symbols failed — those symbols "
                     "will NOT receive orders: %s\n",
                     err.c_str());
    if (cfg_.capital > 0.0) {
        nav_ = cfg_.capital;
    } else {
        AccountInfo ai;
        if (!broker_->account(ai, err)) return fail_("broker account: " + err);
        nav_ = ai.wallet_balance;
        if (nav_ <= 0.0) nav_ = 100000.0;   // sane default for a paper session
    }
    std::fprintf(stderr, "qse_live: nav %.2f USDT, poll %ds\n", nav_, cfg_.poll_sec);

    // 3) Backfill the curated→now gap and ingest it — the backfill events
    //    are appended exactly like a live poll, which evaluates every warm
    //    epoch (with rules loaded, so orders flow).
    try {
        FeedEvents bf;
        const bool full = feed_->backfill(warm_watermarks, bf, err);
        if (!full)
            std::fprintf(stderr, "qse_live: backfill partial: %s\n", err.c_str());
        ingest_(bf);
        if (!fatal_err_.empty()) return 1;
    } catch (const std::exception& e) {
        return fail_("backfill failed: " + std::string(e.what()));
    }

    std::int64_t offset = 0;
    if (feed_->server_time_offset(offset, err))
        std::fprintf(stderr, "qse_live: server clock offset %lld ms\n",
                     static_cast<long long>(offset));

    // 3) Tick loop.
    const qse::TsNs horizon = aligner_->grid().epoch_close.back();
    for (;;) {
        if (stop_ && *stop_) {
            exit_reason_ = "interrupted";
            break;
        }
        const std::int64_t now_ms = unix_ms();
        if (static_cast<qse::TsNs>(now_ms) * 1'000'000LL >= horizon - kHourNs) {
            exit_reason_ = "horizon reached - restart to re-warm";
            break;
        }
        tick_(now_ms);
        if (!fatal_err_.empty()) return 1;
        if (now_ms - last_status_ms_ >= 60000) {
            last_status_ms_ = now_ms;
            heartbeat_(now_ms);
        }
        struct timespec ts;
        ts.tv_sec = cfg_.poll_sec;
        ts.tv_nsec = 0;
        while (nanosleep(&ts, &ts) != 0 && errno == EINTR && !(stop_ && *stop_)) {}
    }
    return 0;
}

void LiveEngine::heartbeat_(std::int64_t now_ms) {
    // Realtime status line: model (backtest-accounted) PnL next to the
    // venue (testnet) PnL, so a slow-trading factor is still visibly alive.
    AccountInfo ai;
    std::string err;
    const bool have_venue = broker_->account(ai, err);
    if (!have_venue)
        std::fprintf(stderr, "qse_live: account: %s\n", err.c_str());
    std::vector<PosRisk> pos;
    const bool have_pos = broker_->positions(pos, err);

    std::size_t n_pos = 0;
    for (const double w : cur_w_)
        if (w != 0.0) ++n_pos;

    const auto& last = recs_.empty() ? LiveEpoch{} : recs_.back();
    std::printf("[status] %lld UTC  locked=%zu  model_equity=%.4f "
                "(pnl_net=%.6f ic=",
                static_cast<long long>(now_ms / 1000), aligner_->locked_rows(),
                equity_, last.pnl_net);
    json_num(stdout, last.ic);
    std::printf(")  venue_balance=%.2f venue_upl=%.2f  n_pos=%zu "
                "venue_pos=%zu  n_epochs=%zu  pending=%zu\n",
                have_venue ? ai.wallet_balance : kLiveNaN,
                have_venue ? ai.unrealized : kLiveNaN, n_pos,
                have_pos ? pos.size() : 0, recs_.size(), pending_.size());
}

int LiveEngine::run_replay_() {
    // CsvFeed needs no clock; run polls until exhausted.
    for (;;) {
        FeedEvents ev;
        std::string err;
        if (!feed_->poll(0, ev, err)) break;
        ingest_(ev);
        if (!fatal_err_.empty()) return 1;
    }
    return 0;
}

void LiveEngine::tick_(std::int64_t now_ms) {
    FeedEvents ev;
    std::string err;
    if (!feed_->poll(now_ms, ev, err)) {
        bool any = false;
        for (const auto& e : ev.errors) any = any || !e.empty();
        if (!any)
            std::fprintf(stderr, "qse_live: poll failed: %s\n", err.c_str());
    }
    ingest_(ev);
}

void LiveEngine::ingest_(const FeedEvents& ev) {
    const auto& g = aligner_->grid();
    const std::size_t N = g.N();
    if (ev.klines.size() != N || ev.funding.size() != N || ev.errors.size() != N) {
        fail_("feed returned malformed events");
        return;
    }

    // A whole tick with every symbol unreachable means the network is down:
    // fail fast instead of grinding per-symbol anomaly counters for hours.
    bool all_down = !ev.errors.empty();
    for (const auto& e : ev.errors) all_down = all_down && !e.empty();
    if (all_down && ++tick_fail_ >= 5) {
        fail_("network unreachable: " + ev.errors.front());
        return;
    }
    if (!all_down) tick_fail_ = 0;

    // Funding first, then klines — prints must reach the aligner before the
    // kline that locks their row (aligner enforces this with a throw).
    for (std::size_t n = 0; n < N; ++n) {
        if (!ev.errors[n].empty()) {
            if (++fetch_fail_[n] > kMaxAnomalies) {
                fail_("feed: too many consecutive failures for " + g.symbols[n] + ": " + ev.errors[n]);
                return;
            }
            std::fprintf(stderr, "qse_live: %s: %s\n", g.symbols[n].c_str(),
                         ev.errors[n].c_str());
        } else {
            fetch_fail_[n] = 0;
        }
        for (const auto& p : ev.funding[n]) {
            try {
                aligner_->append_funding(n, p.calc_ts, p.rate);
            } catch (const std::exception& e) {
                if (std::strstr(e.what(), "locked")) {
                    // Binance fundingTimes are symbol-dependent: some land
                    // exactly on the row close (…000) and can be published
                    // after that row already locked. Expected latency, not
                    // a bug: skip; the level self-heals at the next
                    // settlement. Warn once per symbol.
                    if (late_funding_warned_[n]++ == 0)
                        std::fprintf(stderr,
                                     "qse_live: %s: late funding print skipped: %s\n",
                                     g.symbols[n].c_str(), e.what());
                    continue;
                }
                // Out-of-order: a genuine feed bug.
                fail_("funding append for " + g.symbols[n] + ": " + e.what());
                return;
            }
        }
    }
    for (std::size_t n = 0; n < N; ++n) {
        for (const auto& fk : ev.klines[n]) {
            try {
                aligner_->append(n, fk.k, fk.taker_buy);
            } catch (const std::exception& e) {
                if (std::strstr(e.what(), "locked")) {
                    // A finalized bar landing in a locked epoch means the
                    // engine made a decision using data it must not have had.
                    fail_("append into locked epoch for " + g.symbols[n] +
                          ": " + e.what());
                    return;
                }
                // Regression/off-grid: exchange data quirk; skip and count.
                if (++fetch_fail_[n] > kMaxAnomalies) {
                    fail_("too many append anomalies for " + g.symbols[n]);
                    return;
                }
                std::fprintf(stderr, "qse_live: %s: skipped bar: %s\n",
                             g.symbols[n].c_str(), e.what());
            }
        }
    }
    on_rows_locked_();
}

void LiveEngine::on_rows_locked_() {
    const auto& g = aligner_->grid();
    const std::size_t N = g.N();
    const std::size_t L = aligner_->locked_rows();

    // Place books for every newly locked epoch. Row t locks at close_t ≈
    // open[t+1] + poll latency, so placing now is the live analogue of the
    // backtest's fill at open[t+1]. The mask uses only row t's validity —
    // whether the execution bar will be real is not knowable yet.
    while (next_epoch_ < L) {
        const std::size_t t = next_epoch_;
        if (t >= warmup_) {
            qse::xsec::GridView view(g, t);
            std::fill(scores_.begin(), scores_.end(), kLiveNaN);
            factor_->compute(view, scores_);
            for (std::size_t n = 0; n < N; ++n)
                if (!g.valid[t * N + n]) scores_[n] = kLiveNaN;

            const auto book = qse::xsec::build_book(scores_, cur_w_, cur_side_,
                                                    t, warmup_, cfg_.pf);
            double turnover = 0.0;
            for (std::size_t n = 0; n < N; ++n)
                turnover += std::abs(book.w[n] - cur_w_[n]);

            Pending p;
            p.t = t;
            p.scores = scores_;
            p.w = book.w;
            p.turnover = turnover;
            if (!cfg_.no_orders) place_orders_(t, book, p.fills, p.errors);

            pending_.push_back(std::move(p));
            cur_w_ = book.w;
            cur_side_ = book.side;
        }
        next_epoch_ = t + 1;
    }

    // Finalize epochs whose holding window (t+1, t+2] is fully locked.
    while (!pending_.empty() && pending_.front().t + 2 < L) {
        Pending p = std::move(pending_.front());
        pending_.pop_front();
        finalize_epoch_(p);
    }
}

void LiveEngine::place_orders_(std::size_t t, const qse::xsec::Book& book,
                               std::vector<FillReport>& fills,
                               std::vector<std::string>& errors) {
    const auto& g = aligner_->grid();
    const std::size_t N = g.N();
    std::vector<double> deltas(N), px(N);
    for (std::size_t n = 0; n < N; ++n) {
        deltas[n] = book.w[n] - cur_w_[n];
        // Execution price is not knowable at decision time; the venue fills
        // at market. Use the last known close for quantity sizing only.
        px[n] = g.close[t * N + n];
    }
    auto orders = map_deltas(deltas, px, rules_, nav_, kWMin, t,
                             g.symbols);
    double delta_sum = 0.0;
    for (const double d : deltas) delta_sum += std::abs(d);
    if (orders.empty()) {
        // Carry epochs legitimately produce no deltas; only call out a
        // rebalance whose deltas were all filtered by rules/rounding.
        if (jsonl_ && delta_sum > 0.0)
            std::fprintf(stderr,
                         "qse_live: t=%zu: no orders despite %g weight delta "
                         "(filtered by rules/rounding?)\n",
                         t, delta_sum);
        return;
    }

    std::string err;
    bool ok = broker_->place(orders, fills, err);
    if (!ok) {
        std::string err2;
        ok = broker_->place(orders, fills, err2);   // one retry, same ids;
        if (ok) err.clear();                        // fills append across tries
        else err = err2;
    }

    // Venue side from the order spec: fills carry no side, so re-derive it.
    // Processed even on failure: a partial failure's acked fills must still
    // update the venue bookkeeping.
    for (const auto& o : orders) {
        const auto it = std::find(g.symbols.begin(), g.symbols.end(), o.symbol);
        if (it == g.symbols.end()) continue;
        const std::size_t n = static_cast<std::size_t>(it - g.symbols.begin());
        const auto fit = std::find_if(
            fills.begin(), fills.end(),
            [&](const FillReport& f) { return f.client_id == o.client_id; });
        if (fit == fills.end() || fit->status != "FILLED") continue;
        if (!std::isfinite(fit->avg_price)) continue;
        venue_qty_[n] += o.buy ? fit->qty : -fit->qty;
    }
    for (const auto& f : fills)
        if (f.status != "FILLED") {
            const auto it = std::find(g.symbols.begin(), g.symbols.end(), f.symbol);
            if (it != g.symbols.end())
                venue_unknown_[static_cast<std::size_t>(it - g.symbols.begin())] = 1;
        }
    if (!ok) {
        errors.push_back("order placement failed: " + err);
        std::fprintf(stderr, "qse_live: t=%zu: %s\n", t, errors.back().c_str());
        return;
    }
    if (jsonl_)
        std::fprintf(stderr, "qse_live: t=%zu: %zu orders placed\n", t,
                     orders.size());
}

void LiveEngine::finalize_epoch_(const Pending& p) {
    const auto& g = aligner_->grid();
    const std::size_t N = g.N();
    const std::size_t t = p.t;

    LiveEpoch rec;
    rec.t = t;
    rec.ts = g.epoch_close[t];
    for (std::size_t n = 0; n < N; ++n)
        if (std::isfinite(p.scores[n])) ++rec.n_valid;

    std::vector<double> fwd(N);
    for (std::size_t n = 0; n < N; ++n) {
        const double o1 = g.open[(t + 1) * N + n];
        const double o2 = g.open[(t + 2) * N + n];
        fwd[n] = (std::isfinite(o1) && std::isfinite(o2) && o1 > 0.0)
                     ? o2 / o1 - 1.0
                     : kLiveNaN;
    }
    rec.ic = qse::xsec::spearman(p.scores, fwd);
    rec.turnover = p.turnover;
    rec.cost = p.turnover * (cfg_.pf.fee_bps + cfg_.pf.slip_bps) * 1e-4;

    for (std::size_t n = 0; n < N; ++n) {
        if (p.w[n] == 0.0) continue;
        if (std::isfinite(fwd[n])) rec.pnl_gross += p.w[n] * fwd[n];
        const double fp = g.funding_paid[(t + 1) * N + n];
        if (std::isfinite(fp)) rec.funding_pnl -= p.w[n] * fp;
    }
    rec.pnl_net = rec.pnl_gross - rec.cost + rec.funding_pnl;

    equity_ *= (1.0 + rec.pnl_net);
    rec.model_equity = equity_;
    qse::xsec::digest_add(digest_, equity_);
    qse::xsec::digest_add(digest_, std::isfinite(rec.ic) ? rec.ic : -999.0);

    std::string err;
    AccountInfo ai;
    if (broker_->account(ai, err)) {
        rec.venue_balance = ai.wallet_balance;
        rec.venue_unrealized = ai.unrealized;
    }
    rec.fills = p.fills;
    rec.errors = p.errors;
    recs_.push_back(rec);
    log_epoch_(rec);
}

void LiveEngine::log_epoch_(const LiveEpoch& rec) {
    if (!jsonl_) return;
    std::fprintf(jsonl_,
                 "{\"t\":%zu,\"ts\":%lld,\"ic\":", rec.t,
                 static_cast<long long>(rec.ts));
    json_num(jsonl_, rec.ic);
    std::fprintf(jsonl_,
                 ",\"pnl_gross\":");
    json_num(jsonl_, rec.pnl_gross);
    std::fprintf(jsonl_, ",\"cost\":");
    json_num(jsonl_, rec.cost);
    std::fprintf(jsonl_, ",\"funding_pnl\":");
    json_num(jsonl_, rec.funding_pnl);
    std::fprintf(jsonl_, ",\"pnl_net\":");
    json_num(jsonl_, rec.pnl_net);
    std::fprintf(jsonl_, ",\"turnover\":");
    json_num(jsonl_, rec.turnover);
    std::fprintf(jsonl_, ",\"n_valid\":%zu,\"model_equity\":", rec.n_valid);
    json_num(jsonl_, rec.model_equity);
    std::fprintf(jsonl_, ",\"venue_balance\":");
    json_num(jsonl_, rec.venue_balance);
    std::fprintf(jsonl_, ",\"venue_unrealized\":");
    json_num(jsonl_, rec.venue_unrealized);
    std::fprintf(jsonl_, ",\"orders\":[");
    bool first = true;
    double slip_sum = 0.0, slip_max = 0.0;
    const auto& g = aligner_->grid();
    const std::size_t N = g.N();
    for (const auto& f : rec.fills) {
        if (!first) std::fprintf(jsonl_, ",");
        first = false;
        std::fprintf(jsonl_,
                     "{\"sym\":\"%s\",\"side\":%d,\"qty\":",
                     f.symbol.c_str(), 0);
        json_num(jsonl_, f.qty);
        std::fprintf(jsonl_, ",\"client_id\":\"%s\",\"status\":\"%s\",\"venue_px\":",
                     f.client_id.c_str(), f.status.c_str());
        json_num(jsonl_, f.avg_price);
        const auto it = std::find(g.symbols.begin(), g.symbols.end(), f.symbol);
        if (it != g.symbols.end()) {
            const std::size_t n = static_cast<std::size_t>(it - g.symbols.begin());
            const double model = g.open[(rec.t + 1) * N + n];
            std::fprintf(jsonl_, ",\"model_px\":");
            json_num(jsonl_, model);
            if (std::isfinite(f.avg_price) && std::isfinite(model) && model > 0.0) {
                const double slip = std::abs(f.avg_price - model) / model * 1e4;
                slip_sum += slip;
                slip_max = std::max(slip_max, slip);
            }
        }
        std::fputs("}", jsonl_);
    }
    std::fprintf(jsonl_, "],\"slip_sum_bps\":");
    json_num(jsonl_, slip_sum);
    std::fprintf(jsonl_, ",\"slip_max_bps\":");
    json_num(jsonl_, slip_max);
    std::fprintf(jsonl_, ",\"errors\":[");
    first = true;
    for (const auto& e : rec.errors) {
        if (!first) std::fprintf(jsonl_, ",");
        first = false;
        std::fprintf(jsonl_, "\"%s\"", json_escape(e).c_str());
    }
    std::fprintf(jsonl_, "]}\n");
    std::fflush(jsonl_);

    std::printf("t=%zu ts=%lld ic=", rec.t, static_cast<long long>(rec.ts));
    json_num(stdout, rec.ic);
    std::printf(" pnl_net=");
    json_num(stdout, rec.pnl_net);
    std::printf(" equity=");
    json_num(stdout, rec.model_equity);
    std::printf(" venue=");
    json_num(stdout, rec.venue_balance);
    std::printf(" orders=%zu\n", rec.fills.size());
}

void LiveEngine::write_summary_() {
    if (!jsonl_) return;
    const std::string path = cfg_.log_dir + "/summary.json";
    if (std::FILE* f = std::fopen(path.c_str(), "w")) {
        double ic_sum = 0.0, ic_sum2 = 0.0, r_sum = 0.0, r_sum2 = 0.0;
        std::size_t ic_n = 0;
        for (const auto& e : recs_) {
            if (std::isfinite(e.ic)) { ic_sum += e.ic; ic_sum2 += e.ic * e.ic; ++ic_n; }
            r_sum += e.pnl_net;
            r_sum2 += e.pnl_net * e.pnl_net;
        }
        double mean_ic = kLiveNaN, ic_tstat = kLiveNaN, sharpe = kLiveNaN;
        double max_dd = 0.0, peak = 1.0;
        for (const auto& e : recs_) {
            peak = std::max(peak, e.model_equity);
            max_dd = std::max(max_dd, 1.0 - e.model_equity / peak);
        }
        if (ic_n >= 2) {
            const double m = static_cast<double>(ic_n);
            mean_ic = ic_sum / m;
            const double sd = std::sqrt(std::max(0.0, ic_sum2 / m - mean_ic * mean_ic));
            if (sd > 0.0) ic_tstat = mean_ic / sd * std::sqrt(m);
        }
        if (!recs_.empty()) {
            const double m = static_cast<double>(recs_.size());
            const double mean_r = r_sum / m;
            const double sd_r = std::sqrt(std::max(0.0, r_sum2 / m - mean_r * mean_r));
            if (sd_r > 0.0)
                sharpe = mean_r / sd_r *
                         std::sqrt(365.0 * 24.0 * 3600.0 * 1e9 /
                                   static_cast<double>(aligner_->grid().epoch_ns));
        }
        std::fprintf(f, "{\"factor\":\"%s\",\"mode\":\"%s\",\"exit\":\"%s\","
                        "\"replay_span\":\"%s..%s\",\"warm\":\"%s..%s\","
                        "\"rebalance_every\":%zu,\"q_out\":%.3f,\"capital\":",
                    cfg_.factor_name.c_str(), cfg_.replay ? "replay" : "live",
                    exit_reason_.c_str(), cfg_.start_ym.c_str(), cfg_.end_ym.c_str(),
                    start_ym_.c_str(), end_ym_.c_str(), cfg_.pf.rebalance_every,
                    cfg_.pf.q_out);
        json_num(f, nav_);
        std::fprintf(f, ",\"epochs\":%zu,\"mean_ic\":", recs_.size());
        json_num(f, mean_ic);
        std::fprintf(f, ",\"ic_tstat\":");
        json_num(f, ic_tstat);
        std::fprintf(f, ",\"ann_sharpe_net\":");
        json_num(f, sharpe);
        std::fprintf(f, ",\"total_return_net\":");
        json_num(f, equity_ - 1.0);
        std::fprintf(f, ",\"max_drawdown\":");
        json_num(f, max_dd);
        std::fprintf(f, ",\"model_digest\":\"%016llx\",\"model_equity\":",
                     static_cast<unsigned long long>(digest_));
        json_num(f, equity_);
        std::fprintf(f, ",\"venue_balance\":");
        json_num(f, recs_.empty() ? kLiveNaN : recs_.back().venue_balance);
        std::fprintf(f, "}\n");
        std::fclose(f);
    }
    std::printf("qse_live: %s: %zu epochs, model equity %.6f (digest %016llx)\n",
                exit_reason_.c_str(), recs_.size(), equity_,
                static_cast<unsigned long long>(digest_));
}

}  // namespace qse::live
