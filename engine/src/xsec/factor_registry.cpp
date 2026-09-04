#include "qse/xsec/factor.hpp"

namespace qse::xsec {

std::map<std::string, XFactorFactory>& xfactor_registry() {
    static std::map<std::string, XFactorFactory> reg;
    return reg;
}

XFactorRegistrar::XFactorRegistrar(const std::string& name, XFactorFactory f) {
    xfactor_registry().emplace(name, std::move(f));
}

}  // namespace qse::xsec
