#pragma once

// Rank IC: Spearman correlation between factor scores and forward returns.

#include <span>

namespace qse::xsec {

// Spearman rank correlation with midrank (average-rank) tie handling:
// midrank both vectors, then Pearson on the ranks. Pairs where either entry
// is NaN are excluded. Returns NaN with fewer than 3 usable pairs or when
// either side is constant.
double spearman(std::span<const double> x, std::span<const double> y);

}  // namespace qse::xsec
