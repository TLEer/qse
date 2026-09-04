// Low-volatility tilt: long the calm names, short the wild ones. The
// cross-sectional low-vol premium survives in crypto because leverage-
// constrained longs crowd the high-beta end of the universe.
//
// score = -zscore( std of 1h returns over 168h )

#include "qse/xsec/factor.hpp"
#include "qse/xsec/ops.hpp"

namespace {

using namespace qse::xsec;

class LowVol final : public IXsecFactor {
public:
    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n) {
            const double v = ops::ts_ret_std(g, n, 168);
            out[n] = std::isfinite(v) ? -v : kNaN;
        }
        ops::xs_zscore(out);
    }
    std::size_t warmup() const override { return 170; }
};

}  // namespace

QSE_REGISTER_XFACTOR(low_vol_168h, LowVol);
