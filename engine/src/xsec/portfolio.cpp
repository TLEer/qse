#include "qse/xsec/portfolio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "qse/xsec/ic.hpp"

namespace qse::xsec {

// FNV-1a fold over raw double bits: the determinism digest. Any FP
// divergence between the two runs of a factor shows up here.
void digest_add(std::uint64_t& h, double x) {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(x));
    std::memcpy(&bits, &x, sizeof(bits));
    for (int i = 0; i < 8; ++i) {
        h ^= (bits >> (i * 8)) & 0xff;
        h *= 1099511628211ULL;
    }
}

// Verbatim extraction of run_xsec's per-epoch book construction; the
// backtest loop and the live adapter must build identical books from
// identical inputs, so this stays a pure function of (scores, carry state,
// epoch). FP-op order matches the original loop statement-for-statement.
Book build_book(std::span<const double> scores, const std::vector<double>& prev_w,
                const std::vector<int>& prev_side, std::size_t epoch,
                std::size_t warmup, const PortfolioConfig& cfg) {
    const std::size_t N = scores.size();
    Book book;
    book.w.assign(N, 0.0);
    book.side.assign(N, 0);

    // Rank valid names by score, best first; ties broken by symbol index
    // so the book is deterministic.
    std::vector<std::size_t> order;
    order.reserve(N);
    for (std::size_t n = 0; n < N; ++n)
        if (std::isfinite(scores[n])) order.push_back(n);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    const std::size_t V = order.size();

    const bool rebalance =
        cfg.rebalance_every <= 1 ||
        (epoch - warmup) % cfg.rebalance_every == 0;

    if (!rebalance) {
        // Carry the book between rebalance epochs; nothing trades.
        book.w = prev_w;
        book.side = prev_side;
    } else if (V >= cfg.min_names) {
        const auto k_in = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::floor(cfg.q_in * static_cast<double>(V))));
        const auto k_out = std::max<std::size_t>(
            k_in, static_cast<std::size_t>(std::floor(cfg.q_out * static_cast<double>(V))));

        for (std::size_t p = 0; p < V; ++p) {
            const std::size_t n = order[p];
            const std::size_t rank_top = p + 1;        // 1 = best score
            const std::size_t rank_bot = V - p;        // 1 = worst score
            if (rank_top <= k_in || (prev_side[n] > 0 && rank_top <= k_out))
                book.side[n] = +1;
            else if (rank_bot <= k_in || (prev_side[n] < 0 && rank_bot <= k_out))
                book.side[n] = -1;
        }

        std::size_t L = 0, S = 0;
        for (std::size_t n = 0; n < N; ++n) {
            if (book.side[n] > 0) ++L;
            if (book.side[n] < 0) ++S;
        }
        if (L > 0 && S > 0) {
            // Per-side normalization keeps the book dollar-neutral even
            // when validity drops names asymmetrically.
            for (std::size_t n = 0; n < N; ++n) {
                if (book.side[n] > 0) book.w[n] = cfg.gross / (2.0 * static_cast<double>(L));
                if (book.side[n] < 0) book.w[n] = -cfg.gross / (2.0 * static_cast<double>(S));
            }
        } else {
            std::fill(book.side.begin(), book.side.end(), 0);
        }
    }
    return book;
}

XsecResult run_xsec(const MarketGrid& g, IXsecFactor& factor,
                    const PortfolioConfig& cfg) {
    const std::size_t T = g.T();
    const std::size_t N = g.N();

    XsecResult res;
    res.equity.push_back(1.0);
    res.digest = 1469598103934665603ULL;   // FNV offset basis

    // Side each name currently holds: +1 long, -1 short, 0 flat.
    std::vector<int> side(N, 0);
    std::vector<double> prev_w(N, 0.0);
    std::vector<double> scores(N), fwd(N);

    const std::size_t warmup = std::max<std::size_t>(factor.warmup(), 1);
    if (T < warmup + 3) return res;

    for (std::size_t t = warmup; t + 2 < T; ++t) {
        GridView view(g, t);
        std::fill(scores.begin(), scores.end(), kNaN);
        factor.compute(view, scores);

        // Tradability mask: need a real bar now and a real open to trade at.
        for (std::size_t n = 0; n < N; ++n)
            if (!g.valid[t * N + n] || !g.valid[(t + 1) * N + n])
                scores[n] = kNaN;

        // Forward open-to-open return over the holding epoch.
        for (std::size_t n = 0; n < N; ++n) {
            const double o1 = g.open[(t + 1) * N + n];
            const double o2 = g.open[(t + 2) * N + n];
            fwd[n] = (std::isfinite(o1) && std::isfinite(o2) && o1 > 0.0)
                         ? o2 / o1 - 1.0
                         : kNaN;
        }

        const Book book = build_book(scores, prev_w, side, t, warmup, cfg);
        const auto& w = book.w;
        const auto& new_side = book.side;

        // Valid-name count for the record: same set build_book ranks over.
        std::size_t V = 0;
        for (std::size_t n = 0; n < N; ++n)
            if (std::isfinite(scores[n])) ++V;

        EpochRecord rec;
        rec.ts = g.epoch_close[t];
        rec.n_valid = V;
        rec.ic = spearman(scores, fwd);

        for (std::size_t n = 0; n < N; ++n)
            rec.turnover += std::abs(w[n] - prev_w[n]);
        rec.cost = rec.turnover * (cfg.fee_bps + cfg.slip_bps) * 1e-4;

        for (std::size_t n = 0; n < N; ++n) {
            if (w[n] == 0.0) continue;
            if (std::isfinite(fwd[n])) rec.pnl_gross += w[n] * fwd[n];
            // Funding accrued over the hold: row t+1's accrual window is
            // (open[t+1], open[t+2]] on the epoch grid. Positive rate: longs pay.
            const double fp = g.funding_paid[(t + 1) * N + n];
            if (std::isfinite(fp)) rec.funding_pnl -= w[n] * fp;
        }
        rec.pnl_net = rec.pnl_gross - rec.cost + rec.funding_pnl;

        res.equity.push_back(res.equity.back() * (1.0 + rec.pnl_net));
        res.epochs.push_back(rec);
        prev_w = w;
        side = new_side;

        digest_add(res.digest, res.equity.back());
        digest_add(res.digest, std::isfinite(rec.ic) ? rec.ic : -999.0);
    }

    // ---- Aggregates ----
    double ic_sum = 0.0, ic_sum2 = 0.0, to_sum = 0.0;
    double r_sum = 0.0, r_sum2 = 0.0;
    std::size_t ic_n = 0;
    for (const auto& e : res.epochs) {
        if (std::isfinite(e.ic)) { ic_sum += e.ic; ic_sum2 += e.ic * e.ic; ++ic_n; }
        to_sum += e.turnover;
        r_sum += e.pnl_net;
        r_sum2 += e.pnl_net * e.pnl_net;
        if (e.turnover > 0.0) ++res.n_bets;   // a rebalance that actually traded
    }
    res.n_epochs = ic_n;
    if (ic_n >= 2) {
        const double m = static_cast<double>(ic_n);
        res.mean_ic = ic_sum / m;
        res.ic_std = std::sqrt(std::max(0.0, ic_sum2 / m - res.mean_ic * res.mean_ic));
        if (res.ic_std > 0.0) {
            res.ic_ir = res.mean_ic / res.ic_std;
            res.ic_tstat = res.ic_ir * std::sqrt(m);
        }
    }
    if (!res.epochs.empty()) {
        const double m = static_cast<double>(res.epochs.size());
        res.avg_turnover = to_sum / m;
        const double mean_r = r_sum / m;
        const double sd_r = std::sqrt(std::max(0.0, r_sum2 / m - mean_r * mean_r));
        const double epochs_per_year =
            365.0 * 24.0 * 3600.0 * 1e9 / static_cast<double>(g.epoch_ns);
        if (sd_r > 0.0)
            res.ann_sharpe_net = mean_r / sd_r * std::sqrt(epochs_per_year);

        // Moments of the same net-return series, for the deflated Sharpe:
        // T is the number of return draws the Sharpe was estimated from
        // (hourly epochs — a held book draws a fresh return every epoch, so
        // held positions are extra draws, not repeated ones), and the skew
        // and kurtosis thin the Sharpe's standard error.
        res.n_returns = res.epochs.size();
        if (sd_r > 0.0) {
            res.sharpe_epoch = mean_r / sd_r;
            double m3 = 0.0, m4 = 0.0;
            for (const auto& e : res.epochs) {
                const double d = e.pnl_net - mean_r;
                m3 += d * d * d;
                m4 += d * d * d * d;
            }
            m3 /= m;
            m4 /= m;
            res.skew = m3 / (sd_r * sd_r * sd_r);
            res.kurtosis = m4 / (sd_r * sd_r * sd_r * sd_r);
        }
    }
    res.total_return_net = res.equity.back() - 1.0;
    double peak = res.equity.front();
    for (double e : res.equity) {
        peak = std::max(peak, e);
        res.max_drawdown = std::max(res.max_drawdown, 1.0 - e / peak);
    }
    return res;
}

}  // namespace qse::xsec
