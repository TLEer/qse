#pragma once

// The operator vocabulary cross-sectional factors are written in. This is
// the complete surface the LLM Implementer is allowed to use besides GridView
// accessors — keep it small and NaN-disciplined: NaN in, NaN out, and NaN
// entries are excluded from (not poisoning) cross-sectional statistics.

#include <cstddef>
#include <span>

#include "qse/xsec/grid.hpp"

namespace qse::xsec::ops {

// ---- Cross-sectional transforms (in place on a score vector) ----

// Midrank (average-rank ties), normalized to [-0.5, +0.5]; NaNs stay NaN.
void xs_rank(std::span<double> v);
// (x - mean) / std over non-NaN entries; all-equal vectors become all 0.
void xs_zscore(std::span<double> v);
// x - mean over non-NaN entries.
void xs_demean(std::span<double> v);

double clip(double x, double lo, double hi);

// ---- Time-series stats over the grid, window rows ending at the current
// row (r = 0 .. window-1). NaN if any row in the window is missing. ----

double ts_mean(const GridView& g, Field f, std::size_t n, std::size_t window);
double ts_std(const GridView& g, Field f, std::size_t n, std::size_t window);
double ts_sum(const GridView& g, Field f, std::size_t n, std::size_t window);
// value now minus value `window` rows back.
double ts_delta(const GridView& g, Field f, std::size_t n, std::size_t window);
double ts_min(const GridView& g, Field f, std::size_t n, std::size_t window);
double ts_max(const GridView& g, Field f, std::size_t n, std::size_t window);
// close-to-close return over `window` rows (== g.ret(n, window)).
double ts_ret(const GridView& g, std::size_t n, std::size_t window);
// std of 1-row close returns over `window` rows.
double ts_ret_std(const GridView& g, std::size_t n, std::size_t window);
// Pearson correlation of two fields over the window.
double ts_corr(const GridView& g, Field a, Field b, std::size_t n, std::size_t window);

}  // namespace qse::xsec::ops
