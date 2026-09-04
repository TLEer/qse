#include "qse/xsec/ops.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qse::xsec::ops {

namespace {

// Midranks over non-NaN entries, 1-based; NaN entries get NaN.
std::vector<double> midranks(std::span<const double> v) {
    std::vector<std::size_t> idx;
    idx.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        if (std::isfinite(v[i])) idx.push_back(i);
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });

    std::vector<double> r(v.size(), kNaN);
    std::size_t i = 0;
    while (i < idx.size()) {
        std::size_t j = i;
        while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]]) ++j;
        const double rank = 0.5 * static_cast<double>(i + j) + 1.0;  // mean of i+1..j+1
        for (std::size_t k = i; k <= j; ++k) r[idx[k]] = rank;
        i = j + 1;
    }
    return r;
}

}  // namespace

void xs_rank(std::span<double> v) {
    const auto r = midranks(v);
    std::size_t cnt = 0;
    for (double x : r)
        if (std::isfinite(x)) ++cnt;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(r[i])) { v[i] = kNaN; continue; }
        v[i] = cnt > 1 ? (r[i] - 1.0) / static_cast<double>(cnt - 1) - 0.5 : 0.0;
    }
}

void xs_zscore(std::span<double> v) {
    double sum = 0.0, sum2 = 0.0;
    std::size_t cnt = 0;
    for (double x : v)
        if (std::isfinite(x)) { sum += x; sum2 += x * x; ++cnt; }
    if (cnt < 2) { for (auto& x : v) if (std::isfinite(x)) x = 0.0; return; }
    const double mean = sum / static_cast<double>(cnt);
    const double var = std::max(0.0, sum2 / static_cast<double>(cnt) - mean * mean);
    const double sd = std::sqrt(var);
    for (auto& x : v) {
        if (!std::isfinite(x)) continue;
        x = sd > 0.0 ? (x - mean) / sd : 0.0;
    }
}

void xs_demean(std::span<double> v) {
    double sum = 0.0;
    std::size_t cnt = 0;
    for (double x : v)
        if (std::isfinite(x)) { sum += x; ++cnt; }
    if (cnt == 0) return;
    const double mean = sum / static_cast<double>(cnt);
    for (auto& x : v)
        if (std::isfinite(x)) x -= mean;
}

double clip(double x, double lo, double hi) {
    if (!std::isfinite(x)) return kNaN;
    return std::min(std::max(x, lo), hi);
}

namespace {

// Applies fn(value) for each of the `window` rows ending at r=0; returns
// false (→ NaN result) if any value in the window is non-finite.
template <typename Fn>
bool for_window(const GridView& g, Field f, std::size_t n, std::size_t window, Fn&& fn) {
    if (window == 0 || window > g.history()) return false;
    for (std::size_t r = 0; r < window; ++r) {
        const double x = g.get(f, n, r);
        if (!std::isfinite(x)) return false;
        fn(x);
    }
    return true;
}

}  // namespace

double ts_mean(const GridView& g, Field f, std::size_t n, std::size_t window) {
    double s = 0.0;
    if (!for_window(g, f, n, window, [&](double x) { s += x; })) return kNaN;
    return s / static_cast<double>(window);
}

double ts_sum(const GridView& g, Field f, std::size_t n, std::size_t window) {
    double s = 0.0;
    if (!for_window(g, f, n, window, [&](double x) { s += x; })) return kNaN;
    return s;
}

double ts_std(const GridView& g, Field f, std::size_t n, std::size_t window) {
    double s = 0.0, s2 = 0.0;
    if (window < 2) return kNaN;
    if (!for_window(g, f, n, window, [&](double x) { s += x; s2 += x * x; })) return kNaN;
    const double w = static_cast<double>(window);
    const double mean = s / w;
    return std::sqrt(std::max(0.0, s2 / w - mean * mean));
}

double ts_delta(const GridView& g, Field f, std::size_t n, std::size_t window) {
    const double a = g.get(f, n, 0), b = g.get(f, n, window);
    if (!std::isfinite(a) || !std::isfinite(b)) return kNaN;
    return a - b;
}

double ts_min(const GridView& g, Field f, std::size_t n, std::size_t window) {
    double m = std::numeric_limits<double>::infinity();
    if (!for_window(g, f, n, window, [&](double x) { m = std::min(m, x); })) return kNaN;
    return m;
}

double ts_max(const GridView& g, Field f, std::size_t n, std::size_t window) {
    double m = -std::numeric_limits<double>::infinity();
    if (!for_window(g, f, n, window, [&](double x) { m = std::max(m, x); })) return kNaN;
    return m;
}

double ts_ret(const GridView& g, std::size_t n, std::size_t window) {
    return g.ret(n, window);
}

double ts_ret_std(const GridView& g, std::size_t n, std::size_t window) {
    if (window < 2 || window + 1 > g.history()) return kNaN;
    double s = 0.0, s2 = 0.0;
    for (std::size_t r = 0; r < window; ++r) {
        const double a = g.close(n, r), b = g.close(n, r + 1);
        if (!std::isfinite(a) || !std::isfinite(b) || b == 0.0) return kNaN;
        const double x = a / b - 1.0;
        s += x;
        s2 += x * x;
    }
    const double w = static_cast<double>(window);
    const double mean = s / w;
    return std::sqrt(std::max(0.0, s2 / w - mean * mean));
}

double ts_corr(const GridView& g, Field a, Field b, std::size_t n, std::size_t window) {
    if (window < 2 || window > g.history()) return kNaN;
    double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
    for (std::size_t r = 0; r < window; ++r) {
        const double x = g.get(a, n, r), y = g.get(b, n, r);
        if (!std::isfinite(x) || !std::isfinite(y)) return kNaN;
        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
    }
    const double w = static_cast<double>(window);
    const double cov = sab / w - (sa / w) * (sb / w);
    const double va = saa / w - (sa / w) * (sa / w);
    const double vb = sbb / w - (sb / w) * (sb / w);
    if (va <= 0.0 || vb <= 0.0) return kNaN;
    return cov / std::sqrt(va * vb);
}

}  // namespace qse::xsec::ops
