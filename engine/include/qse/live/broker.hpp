#pragma once

// Broker abstraction for the paper-trading engine.
//
// The engine speaks only to IBroker. Three implementations:
//   - SimBroker: fills every order instantly at the model execution price
//     (the epoch open the backtest would fill at) — used by --replay so the
//     full pipeline runs offline and deterministically.
//   - NullBroker: records would-be orders, fills at the model price, never
//     touches the network — used by --no-orders dry runs.
//   - TestnetBroker: real signed REST against testnet.binancefuture.com.
//
// Order mapping (map_deltas) converts weight deltas into exchange orders:
// qty is computed in integer step units and formatted from those units —
// never via %.Nf of a float, which produces 12.3449999-style artifacts that
// exchanges reject.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "qse/live/feed.hpp"   // kLiveNaN

namespace qse::live {

struct OrderSpec {
    std::string symbol;
    std::string qty_str;     // exchange-ready quantity, integer-unit formatted
    double qty = 0.0;        // same value as double (telemetry / Sim fills)
    bool buy = true;
    std::string client_id;   // idempotency: reused verbatim on retry
    double model_price = 0.0;  // epoch open the backtest would fill at
};

struct FillReport {
    std::string client_id;
    std::string symbol;
    std::string status;      // "FILLED" | "REJECTED" | "MISSING"
    double qty = 0.0;
    double avg_price = kLiveNaN;   // NaN when not filled
};

struct PosRisk {
    std::string symbol;
    double qty = 0.0;              // signed (negative = short)
    double entry_price = 0.0;
    double unrealized = 0.0;
};

struct AccountInfo {
    double wallet_balance = 0.0;   // USDT
    double margin_balance = 0.0;
    double unrealized = 0.0;
};

// Per-symbol exchange filters (LOT_SIZE). step_decimals is the digit count
// after the decimal point of the stepSize *string* — the exchange's own
// precision, used to format quantities.
struct SymbolRules {
    double step_size = 0.0;
    double min_qty = 0.0;
    int step_decimals = 0;
};

class IBroker {
public:
    virtual ~IBroker() = default;
    virtual bool connect(std::string& err) = 0;
    // One-time per-symbol setup (testnet: leverage/margin). Returns false
    // on failure; the engine then excludes that symbol from orders.
    virtual bool setup_symbols(const std::vector<std::string>& syms,
                               std::string& err) = 0;
    virtual bool rules(const std::string& sym, SymbolRules& out) = 0;
    virtual bool account(AccountInfo& out, std::string& err) = 0;
    virtual bool positions(std::vector<PosRisk>& out, std::string& err) = 0;
    // Place a batch of orders; one FillReport per order, in input order.
    virtual bool place(const std::vector<OrderSpec>& orders,
                       std::vector<FillReport>& out, std::string& err) = 0;
};

// Fills at model_price, instant, always accepted. For --replay.
class SimBroker final : public IBroker {
public:
    std::size_t placed = 0;   // telemetry for selftests
    bool connect(std::string& err) override;
    bool setup_symbols(const std::vector<std::string>& syms,
                       std::string& err) override;
    bool rules(const std::string& sym, SymbolRules& out) override;
    bool account(AccountInfo& out, std::string& err) override;
    bool positions(std::vector<PosRisk>& out, std::string& err) override;
    bool place(const std::vector<OrderSpec>& orders,
               std::vector<FillReport>& out, std::string& err) override;
};

// Records orders, fills at model_price, no network. For --no-orders.
class NullBroker final : public IBroker {
public:
    std::size_t placed = 0;
    bool connect(std::string& err) override;
    bool setup_symbols(const std::vector<std::string>& syms,
                       std::string& err) override;
    bool rules(const std::string& sym, SymbolRules& out) override;
    bool account(AccountInfo& out, std::string& err) override;
    bool positions(std::vector<PosRisk>& out, std::string& err) override;
    bool place(const std::vector<OrderSpec>& orders,
               std::vector<FillReport>& out, std::string& err) override;
};

// Weight deltas -> market orders. deltas[n] = w_new[n] - w_old[n]
// (signed weights: positive buys, negative sells — selling reduces a
// short). exec_open[n] is the execution price (grid open[t+1]). nav is the
// live NAV in USDT. Orders below w_min weight or below the exchange minQty
// are skipped (returned orders are loggable no-ops). qty is floored to the
// step; a tiny epsilon avoids float-edge under-rounding.
std::vector<OrderSpec> map_deltas(std::span<const double> deltas,
                                  std::span<const double> exec_open,
                                  std::span<const SymbolRules> rules,
                                  double nav, double w_min,
                                  std::size_t epoch,
                                  const std::vector<std::string>& syms);

// Integer-unit quantity formatting: never through a float. step_decimals is
// the exchange's own precision, e.g. 3 for stepSize "0.00100000".
std::string format_qty(std::int64_t units, int step_decimals);

}  // namespace qse::live
