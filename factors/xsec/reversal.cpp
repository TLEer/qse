// Cross-sectional short-horizon reversal: fade the names that ran hardest
// against the cross-section over the past 1-3 days. Classic liquidity-
// provision premium; cost-sensitive, so registered at three horizons.
//
// score = -rank( ret(W) )

#include "qse/xsec/factor.hpp"
#include "qse/xsec/ops.hpp"

namespace {

using namespace qse::xsec;

class Reversal final : public IXsecFactor {
public:
    explicit Reversal(std::size_t window) : window_(window) {}

    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n) {
            const double r = ops::ts_ret(g, n, window_);
            out[n] = std::isfinite(r) ? -r : kNaN;
        }
        ops::xs_rank(out);
    }
    std::size_t warmup() const override { return window_ + 1; }

private:
    std::size_t window_;
};

}  // namespace

#define REGISTER_REVERSAL(NAME, W)                                          \
    static const ::qse::xsec::XFactorRegistrar qse_xregistrar_##NAME(       \
        #NAME, [] { return std::make_unique<Reversal>(W); })

REGISTER_REVERSAL(reversal_24h, 24);
REGISTER_REVERSAL(reversal_48h, 48);
REGISTER_REVERSAL(reversal_72h, 72);
