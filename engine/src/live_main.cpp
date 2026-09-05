// qse_live — live Binance paper-trading adapter.
//
// Signals come from REAL Binance USDT-M futures production data (public
// REST); orders go to the Binance Futures Testnet (signed REST). The engine
// reuses the EpochAligner streaming path, the shared build_book, and
// run_xsec's exact accounting, so --replay over curated CSVs must reproduce
// the backtest digest bit-for-bit.
//
// Usage:
//   qse_live --data-dir data/curated/um [--factor funding_carry_168h]
//            [--capital 10000] [--no-orders] [--replay --start YYYY-MM
//            --end YYYY-MM] [--horizon-days 365] [--poll-sec 12]
//            [--fee-bps 5] [--slip-bps 3] [--q-in 0.2] [--q-out 0.35]
//            [--rebalance-every 24] [--log-dir reports/live] [--selftest]
//
// Env: BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET (required
// unless --no-orders or --replay or --selftest).

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "qse/live/broker.hpp"
#include "qse/live/engine.hpp"
#include "qse/live/feed.hpp"
#include "qse/live/http_client.hpp"
#include "qse/live/testnet_broker.hpp"
#include "qse/xsec/aligner.hpp"
#include "qse/xsec/factor.hpp"
#include "qse/xsec/loader.hpp"
#include "qse/xsec/portfolio.hpp"

namespace {

using namespace qse::live;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

qse::Kline mk_bar(qse::TsNs open_time, double px, double vol = 100.0) {
    qse::Kline k;
    k.open_time = open_time;
    k.close_time = open_time + 3'600'000'000'000LL - 1;
    k.open = k.high = k.low = k.close = px;
    k.volume = vol;
    k.quote_volume = px * vol;
    return k;
}

// ---------------------------------------------------------------------------
// Selftests
// ---------------------------------------------------------------------------

void test_hmac() {
    struct Case { std::string key, msg, want; };
    const Case cases[] = {
        {std::string(20, '\x0b'), "Hi There",
         "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"},
        {"Jefe", "what do ya want for nothing?",
         "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"},
        {std::string(20, '\xaa'), std::string(50, '\xdd'),
         "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"},
    };
    for (const auto& c : cases)
        check(hmac_sha256_hex(c.key, c.msg) == c.want, "hmac rfc4231 vector");
    // Binance-style: sign a query string with timestamp.
    const std::string q = "symbol=BTCUSDT&recvWindow=5000&timestamp=1751328000000";
    check(hmac_sha256_hex("secret", q).size() == 64, "signer output length");
}

void test_chunked() {
    // Plain two chunks.
    std::string b = "4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
    check(decode_chunked_body(b) && b == "Wikipedia", "chunked basic");
    // Chunk extension + trailers after the 0 chunk (0x1c = 28 bytes).
    b = "1c;ext=abc\r\n0123456789012345678901234567\r\n"
        "0\r\nX-Trailer: 1\r\n\r\n";
    check(decode_chunked_body(b) && b == "0123456789012345678901234567",
          "chunked extension + trailer");
    // 0-size chunk alone.
    b = "0\r\n\r\n";
    check(decode_chunked_body(b) && b.empty(), "chunked empty");
    // Framing error.
    b = "zz\r\nnope";
    check(!decode_chunked_body(b), "chunked rejects garbage");
}

void test_feed_parsers() {
    // Real 1h kline array shape; the last bar is still forming (close in
    // the future) and must be dropped.
    const std::string klines =
        "[[1751328000000,\"0.01141900\",\"0.01147100\",\"0.01135400\","
        "\"0.01142900\",\"451543693.00000\",1751331599999,\"5152503.06\","
        "16427,\"201497819.00000\",\"2299502.30\",0],"
        "[1751331600000,\"0.01142900\",\"0.01143000\",\"0.01142000\","
        "\"0.01142500\",\"12345.00000\",1751335199999,\"141.05\",100,"
        "\"6000.00000\",\"68.55\",0],"
        "[1751335200000,\"0.01142500\",\"0.01143000\",\"0.01142000\","
        "\"0.01142800\",\"999.00000\",1751338799999,\"11.41\",9,"
        "\"500.00000\",\"5.71\",0]]";
    std::vector<FeedKline> bars;
    check(parse_klines_body(klines, 1751336000000LL, bars) && bars.size() == 2,
          "klines: forming bar dropped");
    if (bars.size() == 2) {
        check(bars[0].k.open_time == 1751328000000LL * 1000000,
              "klines: ms -> ns");
        check(bars[1].taker_buy == 6000.0, "klines: taker buy parsed");
    }
    // Off-grid bar (open_time not on the 1h boundary) is dropped.
    const std::string offgrid =
        "[[1751328060000,\"1\",\"1\",\"1\",\"1\",\"1\",1751331659999,\"1\",1]]";
    check(parse_klines_body(offgrid, 1751336000000LL, bars) && bars.empty(),
          "klines: off-grid bar dropped");
    // Garbage body.
    check(!parse_klines_body("not json", 0, bars), "klines: garbage rejected");

    const std::string fund =
        "[{\"symbol\":\"BTCUSDT\",\"fundingRate\":\"0.00010000\","
        "\"fundingTime\":1751328000007},"
        "{\"symbol\":\"BTCUSDT\",\"fundingRate\":\"0.00020000\","
        "\"fundingTime\":1751356800007}]";
    std::vector<FundingPrint> prints;
    check(parse_funding_body(fund, prints) && prints.size() == 2,
          "funding: two prints");
    if (prints.size() == 2)
        check(prints[0].calc_ts == 1751328000007LL * 1000000 &&
                  prints[1].rate == 0.0002,
              "funding: ms->ns and string rate");
    check(!parse_funding_body("[]", prints) || prints.empty(),
          "funding: empty array parses empty");
}

void test_qty() {
    check(format_qty(12345, 3) == "12.345", "qty: floor to step");
    check(format_qty(12, 0) == "12", "qty: whole units");
    check(format_qty(1000, 4) == "0.1000", "qty: padded decimals");
    check(format_qty(1, 8) == "0.00000001", "qty: fine step");

    std::vector<double> deltas = {0.05, 0.0, 0.0, -0.05};
    std::vector<double> px = {50000.0, 3000.0, 150.0, 2.0};
    std::vector<SymbolRules> rules = {
        {0.001, 0.001, 3}, {0.1, 0.1, 1}, {1.0, 1.0, 0}, {0.0001, 0.01, 4}};
    std::vector<std::string> syms = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"};
    auto orders = map_deltas(deltas, px, rules, 100000.0, 1e-4, 42, syms);
    check(orders.size() == 2, "qty: mixed long/short batch");
    if (orders.size() == 2) {
        check(orders[0].buy && orders[0].qty_str == "0.100", "qty: long side");
        check(!orders[1].buy && orders[1].client_id == "qse-42-XRPUSDT",
              "qty: short side + client id");
    }
    // Below minQty: ETH at step 0.1 needs >= 0.1 units; 2 USDT at 3000 = 0.
    deltas = {0.0, 0.00002, 0.0, 0.0};
    orders = map_deltas(deltas, px, rules, 100000.0, 1e-4, 42, syms);
    check(orders.empty(), "qty: below minQty skipped");
    // Below w_min weight.
    deltas = {1e-5, 0.0, 0.0, 0.0};
    orders = map_deltas(deltas, px, rules, 100000.0, 1e-4, 42, syms);
    check(orders.empty(), "qty: below w_min skipped");
}

// Deterministic synthetic grid: 4 symbols, LCG prices, one mid-run gap
// (forward-fill exercised) and one pre-listing symbol. The engine in replay
// must reproduce run_xsec's digest exactly — the core equivalence proof.
qse::xsec::MarketGrid synthetic_grid(std::size_t T, std::size_t N) {
    const qse::TsNs H = 3'600'000'000'000LL;
    qse::xsec::EpochAligner al({"A", "B", "C", "D"}, H, 0,
                               static_cast<qse::TsNs>(T - 1) * H);
    unsigned lcg = 42;
    const auto next = [&]() {
        lcg = lcg * 1664525u + 1013904223u;
        return static_cast<double>(lcg % 1000) / 1000.0;   // [0, 1)
    };
    // Funding first, exactly as the curated loader does — the aligner
    // rejects prints that land in already-locked rows. The rate varies per
    // symbol so the cross-sectional z-score has real spread.
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t t = 8; t < T; t += 8)
            al.append_funding(n, static_cast<qse::TsNs>(t) * H,
                              0.0001 * (static_cast<double>((t + n * 3) % 9) - 3.0));
    for (std::size_t n = 0; n < N; ++n) {
        double px = 100.0 + static_cast<double>(n) * 25.0;
        for (std::size_t t = 0; t < T; ++t) {
            // C lists at row 20; B has a gap in rows 30..35.
            if (n == 2 && t < 20) continue;
            if (n == 1 && t >= 30 && t <= 35) continue;
            px *= (1.0 + 0.002 * (next() - 0.5));          // random walk
            al.append(n, mk_bar(static_cast<qse::TsNs>(t) * H, px));
        }
    }
    al.seal();
    return al.grid();
}

void test_replay_equivalence() {
    const std::size_t T = 200, N = 4;
    const auto grid = synthetic_grid(T, N);

    qse::xsec::PortfolioConfig pf;
    pf.rebalance_every = 24;
    pf.q_out = 0.35;
    pf.min_names = 4;              // the synthetic universe is 4 symbols
    auto& reg = qse::xsec::xfactor_registry();
    auto f1 = reg.at("funding_carry_168h")();
    auto a = qse::xsec::run_xsec(grid, *f1, pf);
    check(a.epochs.size() > 10, "replay: backtest ran");
    check(a.equity.back() != 1.0, "replay: backtest actually trades");

    LiveConfig lc;
    lc.replay = true;
    lc.factor_name = "funding_carry_168h";
    lc.pf = pf;
    lc.capital = 100000.0;
    lc.start_ym = "2025-01";
    lc.end_ym = "2025-01";
    lc.log_dir = "/tmp/qse_replay_log";
    LiveEngine eng(lc, std::make_unique<CsvFeed>(grid),
                   std::make_unique<SimBroker>(), &grid);
    const int rc = eng.run();
    check(rc == 0, "replay: engine clean exit");
    check(eng.model_digest() == a.digest, "replay: digest equals run_xsec");
    check(eng.model_equity() == a.equity.back(),
          "replay: equity equals run_xsec");
}

int run_selftest() {
    test_hmac();
    test_chunked();
    test_feed_parsers();
    test_qty();
    test_replay_equivalence();
    if (failures == 0) std::printf("selftest OK\n");
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

}  // namespace

volatile std::sig_atomic_t g_stop = 0;

extern "C" void on_signal(int) { g_stop = 1; }

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --data-dir dir [--universe file] [--factor NAME]\n"
        "       [--capital USDT] [--horizon-days N] [--poll-sec N]\n"
        "       [--no-orders] [--replay --start YYYY-MM --end YYYY-MM]\n"
        "       [--fee-bps X] [--slip-bps X] [--q-in X] [--q-out X]\n"
        "       [--min-names N] [--rebalance-every N] [--log-dir dir]\n"
        "       [--selftest]\n"
        "env: BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET\n",
        prog);
}

int main(int argc, char** argv) {
    LiveConfig cfg;
    cfg.pf.rebalance_every = 24;    // live defaults: the promoted research config
    cfg.pf.q_out = 0.35;
    bool selftest = false;
    std::string data_dir;

    for (int i = 1; i < argc; ++i) {
        const auto take = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--data-dir") == 0) data_dir = take(argv[i]);
        else if (std::strcmp(argv[i], "--universe") == 0) cfg.universe_path = take(argv[i]);
        else if (std::strcmp(argv[i], "--factor") == 0) cfg.factor_name = take(argv[i]);
        else if (std::strcmp(argv[i], "--capital") == 0) cfg.capital = std::atof(take(argv[i]));
        else if (std::strcmp(argv[i], "--horizon-days") == 0) cfg.horizon_days = std::atoi(take(argv[i]));
        else if (std::strcmp(argv[i], "--poll-sec") == 0) cfg.poll_sec = std::atoi(take(argv[i]));
        else if (std::strcmp(argv[i], "--no-orders") == 0) cfg.no_orders = true;
        else if (std::strcmp(argv[i], "--replay") == 0) cfg.replay = true;
        else if (std::strcmp(argv[i], "--start") == 0) cfg.start_ym = take(argv[i]);
        else if (std::strcmp(argv[i], "--end") == 0) cfg.end_ym = take(argv[i]);
        else if (std::strcmp(argv[i], "--fee-bps") == 0) cfg.pf.fee_bps = std::atof(take(argv[i]));
        else if (std::strcmp(argv[i], "--slip-bps") == 0) cfg.pf.slip_bps = std::atof(take(argv[i]));
        else if (std::strcmp(argv[i], "--q-in") == 0) cfg.pf.q_in = std::atof(take(argv[i]));
        else if (std::strcmp(argv[i], "--q-out") == 0) cfg.pf.q_out = std::atof(take(argv[i]));
        else if (std::strcmp(argv[i], "--min-names") == 0) cfg.pf.min_names = static_cast<std::size_t>(std::atol(take(argv[i])));
        else if (std::strcmp(argv[i], "--rebalance-every") == 0) cfg.pf.rebalance_every = static_cast<std::size_t>(std::atol(take(argv[i])));
        else if (std::strcmp(argv[i], "--log-dir") == 0) cfg.log_dir = take(argv[i]);
        else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (selftest) return run_selftest();

    if (data_dir.empty()) {
        usage(argv[0]);
        return 2;
    }
    cfg.data_dir = data_dir;
    if (cfg.universe_path.empty()) cfg.universe_path = data_dir + "/universe.txt";

    if (!cfg.replay && !cfg.no_orders) {
        // Testnet execution needs keys; fail fast like the agents' config.
        const char* key = std::getenv("BINANCE_TESTNET_API_KEY");
        const char* secret = std::getenv("BINANCE_TESTNET_API_SECRET");
        if (!key || !*key || !secret || !*secret) {
            std::fprintf(stderr,
                "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET are "
                "required for testnet execution (or use --no-orders)\n");
            return 2;
        }
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        auto feed = std::make_unique<BinanceFeed>(
            qse::xsec::load_universe(cfg.universe_path),
            "https://fapi.binance.com", 5000);
        auto broker = std::make_unique<TestnetBroker>(
            key, secret, "https://testnet.binancefuture.com", 5000);
        try {
            LiveEngine eng(cfg, std::move(feed), std::move(broker), nullptr,
                           &g_stop);
            return eng.run();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "qse_live: %s\n", e.what());
            return 1;
        }
    }

    if (cfg.replay) {
        if (cfg.start_ym.empty() || cfg.end_ym.empty()) {
            std::fprintf(stderr, "--replay needs --start and --end\n");
            return 2;
        }
        try {
            const auto grid = qse::xsec::load_um_dir(
                cfg.data_dir, cfg.universe_path, cfg.start_ym, cfg.end_ym);
            auto feed = std::make_unique<CsvFeed>(grid);
            auto broker = std::make_unique<SimBroker>();
            LiveEngine eng(cfg, std::move(feed), std::move(broker), &grid);
            return eng.run();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "qse_live: %s\n", e.what());
            return 1;
        }
    }

    // --no-orders dry run: production data, model book, no venue calls.
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    auto feed = std::make_unique<BinanceFeed>(
        qse::xsec::load_universe(cfg.universe_path), "https://fapi.binance.com",
        5000);
    auto broker = std::make_unique<NullBroker>();
    try {
        LiveEngine eng(cfg, std::move(feed), std::move(broker), nullptr,
                       &g_stop);
        return eng.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qse_live: %s\n", e.what());
        return 1;
    }
}
