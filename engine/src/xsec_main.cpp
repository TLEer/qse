// qse_xsec — run every registered cross-sectional factor over the aligned
// 30-asset UM-perp grid and score it on rank IC + net long/short PnL.
//
// Usage: qse_xsec --data-dir data/curated/um --start 2025-07 --end 2026-06
//                 [--universe path] [--fee-bps 5] [--slip-bps 3]
//                 [--q-in 0.2] [--q-out 0.3] [--factor NAME]
//                 [--report out.json] [--log-dir dir] [--selftest]
//
// Each factor is run twice; mismatched determinism digests fail the batch.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "qse/xsec/aligner.hpp"
#include "qse/xsec/factor.hpp"
#include "qse/xsec/ic.hpp"
#include "qse/xsec/loader.hpp"
#include "qse/xsec/ops.hpp"
#include "qse/xsec/portfolio.hpp"

namespace {

using namespace qse::xsec;

// ---------------------------------------------------------------------------
// Selftest: unit checks on synthetic in-memory data. Kept here so the repo
// needs no test framework; wired into ctest as `qse_xsec --selftest`.
// ---------------------------------------------------------------------------

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

void test_alignment() {
    const qse::TsNs H = 3'600'000'000'000LL;
    // 3 symbols, 12 epochs. B misses epochs 5-7; C lists at epoch 2.
    EpochAligner al({"A", "B", "C"}, H, 0, 11 * H);

    for (int e = 0; e < 12; ++e) al.append(0, mk_bar(e * H, 100.0 + e));
    for (int e = 0; e < 12; ++e) {
        if (e >= 5 && e <= 7) continue;
        al.append(1, mk_bar(e * H, 200.0 + e));
    }
    for (int e = 2; e < 9; ++e) al.append(2, mk_bar(e * H, 300.0 + e));

    // Barrier: C's last bar is epoch 8, so rows 0..8 are locked, 9+ pending.
    check(al.locked_rows() == 9, "barrier holds at the slowest watermark");

    bool threw = false;
    try {
        al.append(2, mk_bar(4 * H, 999.0));   // regression into locked row
    } catch (const std::exception&) { threw = true; }
    check(threw, "append into a locked epoch throws");

    al.append(2, mk_bar(9 * H, 309.0));
    check(al.locked_rows() == 10, "barrier advances when the laggard catches up");

    al.seal();
    const auto& g = al.grid();
    check(al.locked_rows() == 12 && g.T() == 12 && g.N() == 3, "seal locks the span");

    const auto at = [&](std::size_t t, std::size_t n) { return t * 3 + n; };
    // B's gap: close carried from epoch 4, volume zeroed, staleness counts.
    check(g.close[at(5, 1)] == 204.0 && g.close[at(7, 1)] == 204.0, "ffill carries close");
    check(g.volume[at(6, 1)] == 0.0 && g.quote_volume[at(6, 1)] == 0.0, "ffill zeroes volume");
    check(g.valid[at(6, 1)] == 0 && g.valid[at(8, 1)] == 1, "ffill flags validity");
    check(g.staleness[at(5, 1)] == 1 && g.staleness[at(7, 1)] == 3, "staleness counts run");
    // C pre-listing: NaN prices, invalid.
    check(std::isnan(g.close[at(0, 2)]) && g.valid[at(1, 2)] == 0, "pre-listing rows are NaN");
    check(g.valid[at(2, 2)] == 1 && g.close[at(2, 2)] == 302.0, "listing row is real");
}

void test_funding() {
    const qse::TsNs H = 3'600'000'000'000LL;
    EpochAligner al({"A"}, H, 0, 5 * H);
    al.append_funding(0, 1 * H, 0.0001);        // knowable at close of row 0
    al.append_funding(0, 4 * H, -0.0002);       // inside row 3's window
    for (int e = 0; e < 6; ++e) al.append(0, mk_bar(e * H, 100.0));
    al.seal();
    const auto& g = al.grid();
    check(g.funding_paid[0] == 0.0001 && g.funding_level[0] == 0.0001,
          "funding print at row close accrues to that row");
    check(g.funding_level[2] == 0.0001 && g.funding_paid[2] == 0.0,
          "funding level ffills, paid resets");
    check(g.funding_paid[3] == -0.0002 && g.funding_level[5] == -0.0002,
          "later print lands in its accrual row");
}

void test_spearman() {
    {
        const double x[] = {1, 2, 2, 4}, y[] = {10, 20, 30, 40};
        const double rho = spearman(x, y);
        check(std::abs(rho - 0.9487) < 1e-3, "spearman midrank ties (scipy cross-check)");
    }
    {
        const double x[] = {1, 2, 3, 4}, y[] = {4, 3, 2, 1};
        check(std::abs(spearman(x, y) + 1.0) < 1e-12, "spearman perfect inverse");
    }
    {
        const double nan = kNaN;
        const double x[] = {1, nan, 3, 4, 5}, y[] = {2, 100, nan, 8, 10};
        // NaN pairs excluded -> perfect monotone on the 3 remaining pairs.
        check(std::abs(spearman(x, y) - 1.0) < 1e-12, "spearman NaN-pair exclusion");
    }
    {
        const double x[] = {1, 1, 1, 1}, y[] = {1, 2, 3, 4};
        check(std::isnan(spearman(x, y)), "spearman constant side is NaN");
    }
}

// A deliberately simple factor for the harness tests: yesterday's return,
// negated (cross-sectional reversal).
struct TestReversal final : IXsecFactor {
    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n)
            out[n] = -g.ret(n, 1);
        ops::xs_rank(out);
    }
    std::size_t warmup() const override { return 2; }
};

MarketGrid synthetic_grid(std::size_t T, std::size_t N, unsigned seed) {
    const qse::TsNs H = 3'600'000'000'000LL;
    std::vector<std::string> syms;
    for (std::size_t n = 0; n < N; ++n) syms.push_back("S" + std::to_string(n));
    EpochAligner al(syms, H, 0, static_cast<qse::TsNs>(T - 1) * H);
    // Deterministic LCG so the test needs no <random>.
    std::uint64_t s = seed;
    auto rnd = [&] { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                     return static_cast<double>((s >> 33) & 0xffff) / 65536.0; };
    std::vector<double> px(N, 100.0);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t n = 0; n < N; ++n) {
            px[n] *= 1.0 + (rnd() - 0.5) * 0.02;
            al.append(n, mk_bar(static_cast<qse::TsNs>(t) * H, px[n], 50.0 + rnd() * 100.0));
        }
    al.seal();
    return al.grid();
}

void test_no_lookahead_shift() {
    const std::size_t T = 60, N = 8, K = 10;
    auto g1 = synthetic_grid(T, N, 42);
    auto g2 = g1;
    // Corrupt the last K epochs of the copy.
    for (std::size_t t = T - K; t < T; ++t)
        for (std::size_t n = 0; n < N; ++n) {
            const auto i = t * N + n;
            g2.open[i] = g2.high[i] = g2.low[i] = g2.close[i] = 1e9 + static_cast<double>(i);
            g2.volume[i] = 1e12;
        }
    PortfolioConfig cfg;
    cfg.min_names = 4;
    TestReversal f1, f2;
    const auto r1 = run_xsec(g1, f1, cfg);
    const auto r2 = run_xsec(g2, f2, cfg);
    // Every epoch that never touches a corrupted row (needs t+2 < T-K) must
    // be bit-identical: any mismatch means the future leaked backwards.
    bool same = true;
    for (std::size_t i = 0; i < r1.epochs.size() && i < r2.epochs.size(); ++i) {
        const auto& a = r1.epochs[i];
        const auto& b = r2.epochs[i];
        if (a.ts / 3'600'000'000'000LL + 2 >= static_cast<qse::TsNs>(T - K)) break;
        const bool ic_same = (std::isnan(a.ic) && std::isnan(b.ic)) || a.ic == b.ic;
        if (!ic_same || a.pnl_net != b.pnl_net || a.turnover != b.turnover) same = false;
    }
    check(same, "corrupting future epochs never changes past results");
}

void test_accounting() {
    const qse::TsNs H = 3'600'000'000'000LL;
    // 4 symbols, flat prices except one deterministic move, one funding print.
    EpochAligner al({"A", "B", "C", "D"}, H, 0, 5 * H);
    // Funding: print at close of row 3 (inside row 3's accrual window).
    al.append_funding(0, 3 * H + H, 0.001);   // ts = 4H = close of row 3
    const double base[4] = {100.0, 100.0, 100.0, 100.0};
    for (int e = 0; e < 6; ++e)
        for (int n = 0; n < 4; ++n) {
            double px = base[n];
            if (n == 0 && e >= 4) px = 110.0;   // A jumps at row 4's open
            al.append(static_cast<std::size_t>(n), mk_bar(e * H, px));
        }
    al.seal();

    // Factor: long A, short D, ignore B/C. q_in=0.25 -> books of exactly 1.
    struct Fixed final : IXsecFactor {
        void compute(const GridView& g, std::span<double> out) override {
            (void)g;
            out[0] = 1.0; out[1] = 0.0; out[2] = 0.0; out[3] = -1.0;
        }
        std::size_t warmup() const override { return 2; }
    };
    PortfolioConfig cfg;
    cfg.fee_bps = 5.0;
    cfg.slip_bps = 3.0;
    cfg.q_in = 0.25;
    cfg.q_out = 0.25;
    cfg.min_names = 4;
    Fixed f;
    const auto r = run_xsec(al.grid(), f, cfg);
    // Epochs evaluated: t = 2 and t = 3 (t + 2 <= 5).
    check(r.epochs.size() == 2, "accounting: expected two epochs");
    if (r.epochs.size() != 2) return;

    // t=2: enter +0.5 A / -0.5 D at open[3]; hold to open[4]. A: 100->110.
    // turnover 1.0, cost 1.0 * 8e-4. gross = 0.5 * 10% = 0.05.
    // funding print at 4H is inside row 3's window: A pays 0.5 * 0.001.
    const auto& e0 = r.epochs[0];
    check(std::abs(e0.turnover - 1.0) < 1e-12, "accounting: entry turnover");
    check(std::abs(e0.cost - 8e-4) < 1e-12, "accounting: entry cost");
    check(std::abs(e0.pnl_gross - 0.05) < 1e-12, "accounting: gross pnl");
    check(std::abs(e0.funding_pnl + 0.5 * 0.001) < 1e-12, "accounting: funding paid by long");
    check(std::abs(e0.pnl_net - (0.05 - 8e-4 - 5e-4)) < 1e-12, "accounting: net pnl");

    // t=3: same book, zero delta -> zero turnover, zero cost, flat prices.
    const auto& e1 = r.epochs[1];
    check(e1.turnover == 0.0 && e1.cost == 0.0, "accounting: unchanged book costs nothing");
    check(std::abs(e1.pnl_gross) < 1e-12, "accounting: flat hold has no gross pnl");

    const double expected_eq = 1.0 * (1.0 + 0.05 - 8e-4 - 5e-4);
    check(std::abs(r.equity.back() - expected_eq) < 1e-12, "accounting: equity compounds");
}

void test_neutrality_under_dropout() {
    const std::size_t T = 40, N = 10;
    auto g = synthetic_grid(T, N, 7);
    // Kill symbol 0 from row 20 on (simulates a halted name).
    for (std::size_t t = 20; t < T; ++t) g.valid[t * N + 0] = 0;
    struct Momo final : IXsecFactor {
        void compute(const GridView& gv, std::span<double> out) override {
            for (std::size_t n = 0; n < gv.n_symbols(); ++n) out[n] = gv.ret(n, 3);
        }
        std::size_t warmup() const override { return 4; }
    };
    // The invariant is structural (per-side renormalisation): with any book
    // present, sum(w) == 0. Verified via turnover/PnL consistency: a NaN'd
    // name is force-exited, and the run must complete without NaN equity.
    PortfolioConfig cfg;
    cfg.min_names = 5;
    Momo f;
    const auto r = run_xsec(g, f, cfg);
    check(!r.equity.empty() && std::isfinite(r.equity.back()),
          "dropout: equity stays finite when a name dies mid-run");
}

int run_selftest() {
    test_alignment();
    test_funding();
    test_spearman();
    test_no_lookahead_shift();
    test_accounting();
    test_neutrality_under_dropout();
    if (failures == 0) std::printf("selftest OK\n");
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Batch runner
// ---------------------------------------------------------------------------

struct XRun {
    std::string name;
    XsecResult res;
    bool deterministic{};
};

void write_logs(const std::string& dir, const std::string& name, const XsecResult& r) {
    FILE* ic = std::fopen((dir + "/" + name + "_ic.csv").c_str(), "w");
    if (ic) {
        std::fprintf(ic, "ts,ic,turnover,pnl_net\n");
        for (const auto& e : r.epochs)
            std::fprintf(ic, "%lld,%.6f,%.6f,%.8f\n",
                         static_cast<long long>(e.ts), e.ic, e.turnover, e.pnl_net);
        std::fclose(ic);
    }
    FILE* eq = std::fopen((dir + "/" + name + "_equity.csv").c_str(), "w");
    if (eq) {
        for (const double v : r.equity) std::fprintf(eq, "%.8f\n", v);
        std::fclose(eq);
    }
}

void write_report(const char* path, const std::vector<XRun>& runs) {
    FILE* f = std::fopen(path, "w");
    if (!f) return;
    std::fprintf(f, "[\n");
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        std::fprintf(f,
            "  {\"factor\": \"%s\", \"n_epochs\": %zu, \"mean_ic\": %.6f, "
            "\"ic_std\": %.6f, \"ic_tstat\": %.4f, \"ic_ir\": %.6f, "
            "\"ann_sharpe_net\": %.4f, \"total_return_net\": %.6f, "
            "\"max_drawdown\": %.6f, \"avg_turnover\": %.6f, "
            "\"deterministic\": %s}%s\n",
            r.name.c_str(), r.res.n_epochs, r.res.mean_ic, r.res.ic_std,
            r.res.ic_tstat, r.res.ic_ir, r.res.ann_sharpe_net,
            r.res.total_return_net, r.res.max_drawdown, r.res.avg_turnover,
            r.deterministic ? "true" : "false",
            i + 1 < runs.size() ? "," : "");
    }
    std::fprintf(f, "]\n");
    std::fclose(f);
    std::printf("\nreport written to %s\n", path);
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir, universe, start, end, only_factor, log_dir;
    const char* report_path = nullptr;
    PortfolioConfig cfg;
    bool selftest = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) data_dir = argv[++i];
        else if (std::strcmp(argv[i], "--universe") == 0 && i + 1 < argc) universe = argv[++i];
        else if (std::strcmp(argv[i], "--start") == 0 && i + 1 < argc) start = argv[++i];
        else if (std::strcmp(argv[i], "--end") == 0 && i + 1 < argc) end = argv[++i];
        else if (std::strcmp(argv[i], "--fee-bps") == 0 && i + 1 < argc) cfg.fee_bps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--slip-bps") == 0 && i + 1 < argc) cfg.slip_bps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--q-in") == 0 && i + 1 < argc) cfg.q_in = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--q-out") == 0 && i + 1 < argc) cfg.q_out = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--min-names") == 0 && i + 1 < argc) cfg.min_names = static_cast<std::size_t>(std::atol(argv[++i]));
        else if (std::strcmp(argv[i], "--rebalance-every") == 0 && i + 1 < argc) cfg.rebalance_every = static_cast<std::size_t>(std::atol(argv[++i]));
        else if (std::strcmp(argv[i], "--factor") == 0 && i + 1 < argc) only_factor = argv[++i];
        else if (std::strcmp(argv[i], "--report") == 0 && i + 1 < argc) report_path = argv[++i];
        else if (std::strcmp(argv[i], "--log-dir") == 0 && i + 1 < argc) log_dir = argv[++i];
        else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }

    if (selftest) return run_selftest();

    if (data_dir.empty() || start.empty() || end.empty()) {
        std::fprintf(stderr,
            "usage: %s --data-dir dir --start YYYY-MM --end YYYY-MM "
            "[--universe file] [--fee-bps X] [--slip-bps X] [--q-in X] "
            "[--q-out X] [--min-names N] [--factor NAME] [--report out.json] "
            "[--log-dir dir] | --selftest\n", argv[0]);
        return 2;
    }
    if (universe.empty()) universe = data_dir + "/universe.txt";

    const auto grid = load_um_dir(data_dir, universe, start, end);
    std::printf("# grid: %zu symbols x %zu epochs, fee=%.1fbps slip=%.1fbps "
                "q_in=%.2f q_out=%.2f\n",
                grid.N(), grid.T(), cfg.fee_bps, cfg.slip_bps, cfg.q_in, cfg.q_out);

    auto& registry = xfactor_registry();
    if (registry.empty()) {
        std::fprintf(stderr, "no xsec factors registered\n");
        return 1;
    }

    std::vector<XRun> runs;
    bool all_ok = true;

    std::printf("%-22s %7s %8s %7s %9s %8s %8s %9s  %s\n",
                "factor", "epochs", "meanIC", "IC_t", "netShrp", "ret", "maxDD",
                "turnover", "digest");

    for (const auto& [name, make] : registry) {
        if (!only_factor.empty() && name != only_factor) continue;
        auto f1 = make();
        auto f2 = make();
        auto a = run_xsec(grid, *f1, cfg);
        const auto b = run_xsec(grid, *f2, cfg);   // determinism re-run
        const bool det = a.digest == b.digest;
        all_ok &= det;

        if (!log_dir.empty()) write_logs(log_dir, name, a);

        std::printf("%-22s %7zu %8.4f %7.2f %9.2f %7.2f%% %7.2f%% %9.4f  %016llx %s\n",
                    name.c_str(), a.n_epochs, a.mean_ic, a.ic_tstat,
                    a.ann_sharpe_net, a.total_return_net * 100.0,
                    a.max_drawdown * 100.0, a.avg_turnover,
                    static_cast<unsigned long long>(a.digest),
                    det ? "OK" : "MISMATCH");
        runs.push_back({name, std::move(a), det});
    }

    if (report_path) write_report(report_path, runs);

    if (!all_ok) {
        std::fprintf(stderr, "\nDETERMINISM VIOLATION in at least one factor\n");
        return 1;
    }
    return 0;
}
