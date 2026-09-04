// Return deceleration: the return earned over a long window minus the return
// earned over the most recent single period. A large positive gap means the
// name already ran and is now stalling — fade it. A negative gap means the
// most recent bar is driving the whole window return — the move is fresh and
// may have room to continue.
//
// B   = (P_t / P_{t-window} - 1) - (P_t / P_{t-1} - 1)
//     = ts_ret(window) - ts_ret(1)
// score = -zscore(B)

#include "qse/xsec/factor.hpp"
#include "qse/xsec/ops.hpp"

namespace {

using namespace qse::xsec;

class ReturnDeceleration final : public IXsecFactor {
public:
    explicit ReturnDeceleration(std::size_t window) : window_(window) {}

    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n) {
            const double long_ret  = ops::ts_ret(g, n, window_);
            const double short_ret = ops::ts_ret(g, n, window_ /24);
            const double b = long_ret - short_ret + 1;
            out[n] = std::isfinite(b) ? b : kNaN;
        }
        ops::xs_zscore(out);
    }
    std::size_t warmup() const override { return window_ + 1; }

private:
    std::size_t window_;
};

}  // namespace

#define REGISTER_RETURN_DECEL(NAME, W)                                        \
    static const ::qse::xsec::XFactorRegistrar qse_xregistrar_##NAME(         \
        #NAME, [] { return std::make_unique<ReturnDeceleration>(W); })

REGISTER_RETURN_DECEL(return_deceleration_24h, 24);
REGISTER_RETURN_DECEL(return_deceleration_72h, 72);
REGISTER_RETURN_DECEL(return_deceleration_168h, 168);
