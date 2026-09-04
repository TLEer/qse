#pragma once

// Cross-sectional factor interface + registry. Mirrors the single-asset
// factor_registry pattern: ordered map, static registrar, one macro.

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>

#include "qse/xsec/grid.hpp"

namespace qse::xsec {

class IXsecFactor {
public:
    virtual ~IXsecFactor() = default;

    // Called once per locked epoch, t strictly increasing. Write one score
    // per symbol into `out` (size == g.n_symbols()); NaN = no opinion, the
    // portfolio excludes that name. Must be pure with respect to the view:
    // no I/O, clocks, randomness, or static mutable state (the determinism
    // digest rejects violations). Ordinary member state is fine.
    virtual void compute(const GridView& g, std::span<double> out) = 0;

    // Rows required before the first evaluation.
    virtual std::size_t warmup() const { return 24; }
};

using XFactorFactory = std::function<std::unique_ptr<IXsecFactor>()>;

// Ordered map so batch iteration is deterministic regardless of static-init
// order (same rationale as factor_registry()).
std::map<std::string, XFactorFactory>& xfactor_registry();

struct XFactorRegistrar {
    XFactorRegistrar(const std::string& name, XFactorFactory f);
};

}  // namespace qse::xsec

// Place at file scope in a factor .cpp:
//   QSE_REGISTER_XFACTOR(funding_carry, FundingCarry)
#define QSE_REGISTER_XFACTOR(NAME, TYPE)                                     \
    static const ::qse::xsec::XFactorRegistrar qse_xregistrar_##NAME(        \
        #NAME, [] { return std::make_unique<TYPE>(); })
