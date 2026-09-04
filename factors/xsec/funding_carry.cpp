// Cross-sectional funding-rate carry.
//
// Perp funding transfers cash from the crowded side to the other side every
// interval. Shorting the names paying the highest funding and longing the
// names with the lowest (or negative) funding collects that transfer
// directly — the edge is a cash flow, not a price forecast, which is why
// this is the reference baseline for the whole xsec stack.
//
// score = -zscore( mean(funding_level, 16h) ): high funding -> short book.

#include "qse/xsec/factor.hpp"
#include "qse/xsec/ops.hpp"

namespace {

using namespace qse::xsec;

class FundingCarry final : public IXsecFactor {
public:
    explicit FundingCarry(std::size_t window) : window_(window) {}

    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n) {
            const double f = ops::ts_mean(g, Field::FundingLevel, n, window_);
            const double f1 = ops::ts_mean(g, Field::FundingLevel, n, 1);
            const double f_m1 = f - f1;

            // out[n] = std::isfinite(f) ? -((f1 / f) - (f1 / f_m1)) : kNaN;
            out[n] = std::isfinite(f) ? -f : kNaN;
        }
        ops::xs_zscore(out);
    }
    std::size_t warmup() const override { return window_ + 1; }

private:
    std::size_t window_;
};

}  // namespace

// The smoothing window is the carry book's churn dial: longer means fewer
// boundary swaps, and the funding transfer itself does not decay while the
// book sits still.
#define REGISTER_CARRY(NAME, W)                                             \
    static const ::qse::xsec::XFactorRegistrar qse_xregistrar_##NAME(       \
        #NAME, [] { return std::make_unique<FundingCarry>(W); })

REGISTER_CARRY(funding_carry_16h, 16);
REGISTER_CARRY(funding_carry_72h, 72);
REGISTER_CARRY(funding_carry_168h, 168);
 