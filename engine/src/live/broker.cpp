#include "qse/live/broker.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace qse::live {

// ---------------------------------------------------------------------------
// SimBroker / NullBroker — offline fill simulation at the model price.
// ---------------------------------------------------------------------------

bool SimBroker::connect(std::string&) { return true; }
bool SimBroker::setup_symbols(const std::vector<std::string>&, std::string&) {
    return true;
}
bool SimBroker::rules(const std::string&, SymbolRules& out) {
    // Default fine-granular rules; replay order counts are telemetry only,
    // model PnL comes from the grid weights regardless of fills.
    out = SymbolRules{0.001, 0.001, 3};
    return true;
}
bool SimBroker::account(AccountInfo& out, std::string&) {
    out = AccountInfo{};
    return true;
}
bool SimBroker::positions(std::vector<PosRisk>& out, std::string&) {
    out.clear();
    return true;
}
bool SimBroker::place(const std::vector<OrderSpec>& orders,
                      std::vector<FillReport>& out, std::string&) {
    placed += orders.size();
    out.clear();
    out.reserve(orders.size());
    for (const auto& o : orders) {
        FillReport f;
        f.client_id = o.client_id;
        f.symbol = o.symbol;
        f.status = "FILLED";
        f.qty = o.qty;
        f.avg_price = o.model_price;
        out.push_back(std::move(f));
    }
    return true;
}

bool NullBroker::connect(std::string&) { return true; }
bool NullBroker::setup_symbols(const std::vector<std::string>&, std::string&) {
    return true;
}
bool NullBroker::rules(const std::string&, SymbolRules& out) {
    out = SymbolRules{0.001, 0.001, 3};
    return true;
}
bool NullBroker::account(AccountInfo& out, std::string&) {
    out = AccountInfo{};
    return true;
}
bool NullBroker::positions(std::vector<PosRisk>& out, std::string&) {
    out.clear();
    return true;
}
bool NullBroker::place(const std::vector<OrderSpec>& orders,
                       std::vector<FillReport>& out, std::string&) {
    placed += orders.size();
    out.clear();
    out.reserve(orders.size());
    for (const auto& o : orders) {
        FillReport f;
        f.client_id = o.client_id;
        f.symbol = o.symbol;
        f.status = "FILLED";
        f.qty = o.qty;
        f.avg_price = o.model_price;
        out.push_back(std::move(f));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Order mapping
// ---------------------------------------------------------------------------

std::string format_qty(std::int64_t units, int step_decimals) {
    if (step_decimals > 12) step_decimals = 12;   // exchange precision is ≤ 8
    if (step_decimals <= 0) return std::to_string(units);
    std::int64_t scale = 1;
    for (int i = 0; i < step_decimals; ++i) {
        if (scale > 1000000000000LL) break;
        scale *= 10;
    }
    if (scale <= 1) return std::to_string(units);
    char buf[48];
    const std::int64_t whole = units / scale;
    const std::int64_t frac = units % scale;
    std::snprintf(buf, sizeof(buf), "%lld.%0*lld", static_cast<long long>(whole),
                  step_decimals, static_cast<long long>(frac));
    return buf;
}

std::vector<OrderSpec> map_deltas(std::span<const double> deltas,
                                  std::span<const double> exec_open,
                                  std::span<const SymbolRules> rules,
                                  double nav, double w_min,
                                  std::size_t epoch,
                                  const std::vector<std::string>& syms) {
    std::vector<OrderSpec> orders;
    if (deltas.size() != exec_open.size() || deltas.size() != rules.size() ||
        deltas.size() != syms.size())
        return orders;

    for (std::size_t n = 0; n < deltas.size(); ++n) {
        const double d = deltas[n];
        if (!std::isfinite(d) || std::abs(d) < w_min) continue;
        const double px = exec_open[n];
        if (!std::isfinite(px) || px <= 0.0) continue;
        const double step = rules[n].step_size;
        if (!(step > 0.0)) continue;

        // Integer units, floored; epsilon guards float-edge under-rounding.
        const double units_f =
            std::floor(std::abs(d) * nav / (px * step) + 1e-9);
        if (!(units_f >= 1.0)) continue;
        const std::int64_t units = static_cast<std::int64_t>(units_f);
        const double qty = static_cast<double>(units) * step;
        if (qty < rules[n].min_qty) continue;   // below exchange minimum

        OrderSpec o;
        o.symbol = syms[n];
        o.qty_str = format_qty(units, rules[n].step_decimals);
        o.qty = qty;
        o.buy = d > 0.0;
        o.client_id = "qse-" + std::to_string(epoch) + "-" + syms[n];
        o.model_price = px;
        orders.push_back(std::move(o));
    }
    return orders;
}

}  // namespace qse::live
