// Volume shock: fade attention spikes. Names whose recent volume runs far
// above their own weekly norm tend to be crowded retail flow that mean-
// reverts; quiet names earn the other side of the spread.
//
// score = -zscore( ln( mean(vol, 24h) / mean(vol, 168h) ) )

#include <cmath>

#include "qse/xsec/factor.hpp"
#include "qse/xsec/ops.hpp"

namespace {

using namespace qse::xsec;

class VolumeShock final : public IXsecFactor {
public:
    void compute(const GridView& g, std::span<double> out) override {
        for (std::size_t n = 0; n < g.n_symbols(); ++n) {
            const double fast = ops::ts_mean(g, Field::Volume, n, 24);
            const double slow = ops::ts_mean(g, Field::Volume, n, 168);
            out[n] = (std::isfinite(fast) && std::isfinite(slow) &&
                      fast > 0.0 && slow > 0.0)
                         ? -std::log(fast / slow)
                         : kNaN;
        }
        ops::xs_zscore(out);
    }
    std::size_t warmup() const override { return 169; }
};

}  // namespace

QSE_REGISTER_XFACTOR(volume_shock, VolumeShock);
