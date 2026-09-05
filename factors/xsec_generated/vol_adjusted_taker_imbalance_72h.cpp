#include "qse/xsec/factor.hpp"
#include "qse/xsec/ops.hpp"

namespace {

using namespace qse::xsec;

class VolAdjustedTakerImbalance72h : public IXsecFactor {
public:
    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n) {
            const double num = ops::ts_mean(g, Field::TakerBuyRatio, n, 72);
            const double denom = ops::ts_std(g, Field::Close, n, 72) + 1e-6;
            out[n] = (std::isfinite(num) && std::isfinite(denom) && denom > 0) ? (num / denom) : kNaN;
        }
        ops::xs_zscore(out);
    }
    std::size_t warmup() const override { return 73; }
};

}  // namespace

QSE_REGISTER_XFACTOR(vol_adjusted_taker_imbalance_72h, VolAdjustedTakerImbalance72h);
