#include "qse/xsec/ic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace qse::xsec {

namespace {

// 1-based midranks over the given values.
std::vector<double> midranks(const std::vector<double>& v) {
    std::vector<std::size_t> idx(v.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    std::vector<double> r(v.size());
    std::size_t i = 0;
    while (i < idx.size()) {
        std::size_t j = i;
        while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]]) ++j;
        const double rank = 0.5 * static_cast<double>(i + j) + 1.0;
        for (std::size_t k = i; k <= j; ++k) r[idx[k]] = rank;
        i = j + 1;
    }
    return r;
}

}  // namespace

double spearman(std::span<const double> x, std::span<const double> y) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (x.size() != y.size()) return nan;

    std::vector<double> xs, ys;
    xs.reserve(x.size());
    ys.reserve(y.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isfinite(x[i]) && std::isfinite(y[i])) {
            xs.push_back(x[i]);
            ys.push_back(y[i]);
        }
    }
    if (xs.size() < 3) return nan;

    const auto rx = midranks(xs);
    const auto ry = midranks(ys);

    const double m = static_cast<double>(rx.size());
    double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
    for (std::size_t i = 0; i < rx.size(); ++i) {
        sa += rx[i]; sb += ry[i];
        saa += rx[i] * rx[i]; sbb += ry[i] * ry[i]; sab += rx[i] * ry[i];
    }
    const double cov = sab / m - (sa / m) * (sb / m);
    const double va = saa / m - (sa / m) * (sa / m);
    const double vb = sbb / m - (sb / m) * (sb / m);
    if (va <= 0.0 || vb <= 0.0) return nan;
    return cov / std::sqrt(va * vb);
}

}  // namespace qse::xsec
