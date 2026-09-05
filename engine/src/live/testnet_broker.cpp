#include "qse/live/testnet_broker.hpp"

#include <cstdio>
#include <cstdlib>

#include <nlohmann/json.hpp>

#include "qse/live/http_client.hpp"

namespace qse::live {

namespace {

constexpr std::int64_t kRecvWindowMs = 5000;

std::string host_of(const std::string& url) {
    const std::size_t p = url.find("://");
    const std::size_t start = (p == std::string::npos) ? 0 : p + 3;
    const std::size_t slash = url.find('/', start);
    return url.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
}

// stepSize arrives as a string like "0.00100000" or "1"; the exchange's own
// precision (digits after the point) is what quantities are formatted with.
SymbolRules rules_from_lot(const std::string& step_size,
                           const std::string& min_qty) {
    SymbolRules r;
    r.step_size = std::strtod(step_size.c_str(), nullptr);
    r.min_qty = std::strtod(min_qty.c_str(), nullptr);
    const std::size_t dot = step_size.find('.');
    if (dot != std::string::npos)
        r.step_decimals = static_cast<int>(step_size.size() - dot - 1);
    return r;
}

std::string api_error(const HttpResponse& r) {
    std::string msg;
    try {
        const auto j = nlohmann::json::parse(r.body);
        if (j.is_object()) {
            if (j.contains("msg") && j["msg"].is_string())
                msg = j["msg"].get<std::string>();
            if (j.contains("code") && j["code"].is_number_integer())
                msg = "code " + std::to_string(j["code"].get<std::int64_t>()) +
                      (msg.empty() ? "" : ": " + msg);
        }
    } catch (const nlohmann::json::exception&) {
    }
    if (msg.empty()) {
        // Non-JSON body (a WAF/geo block page): show a snippet so the
        // reason is visible in the logs.
        msg = "http " + std::to_string(r.status);
        if (!r.body.empty()) {
            std::string snip = r.body;
            const std::size_t nl = snip.find('\n');
            if (nl != std::string::npos) snip.resize(nl);
            if (snip.size() > 120) snip.resize(120);
            if (!snip.empty()) msg += ": " + snip;
        }
    }
    // Binance's -2015 body carries a giant URL list; cut it after the
    // first line so repeated failures stay readable.
    const std::size_t nl = msg.find('\n');
    if (nl != std::string::npos) msg.resize(nl);
    if (msg.size() > 300) msg.resize(300);
    // The most common paper-trading setup mistake: spot-testnet keys on the
    // futures-testnet endpoint.
    if (msg.find("-2015") != std::string::npos)
        msg += " — use FUTURES testnet keys from testnet.binancefuture.com "
               "(API Management), NOT testnet.binance.vision (spot), and "
               "whitelist your IP if the key is IP-restricted";
    return msg;
}

}  // namespace

TestnetBroker::TestnetBroker(std::string api_key, std::string api_secret,
                             std::string base_url, int timeout_ms)
    : api_key_(std::move(api_key)), api_secret_(std::move(api_secret)),
      base_url_(std::move(base_url)), timeout_ms_(timeout_ms) {
    host_ = host_of(base_url_);
}

bool TestnetBroker::signed_get(const std::string& path, const std::string& params,
                               HttpResponse& out, std::string& err) {
    std::string query = params;
    if (!query.empty()) query += "&";
    query += "recvWindow=" + std::to_string(kRecvWindowMs) +
             "&timestamp=" + std::to_string(unix_ms());
    const std::string sig = hmac_sha256_hex(api_secret_, query);
    out = http_request(host_, path + "?" + query + "&signature=" + sig, "GET",
                       "", {"X-MBX-APIKEY: " + api_key_}, timeout_ms_);
    if (!out.ok()) {
        err = api_error(out);
        return false;
    }
    return true;
}

bool TestnetBroker::signed_post(const std::string& path, const std::string& params,
                                HttpResponse& out, std::string& err) {
    std::string query = params;
    if (!query.empty()) query += "&";
    query += "recvWindow=" + std::to_string(kRecvWindowMs) +
             "&timestamp=" + std::to_string(unix_ms());
    const std::string sig = hmac_sha256_hex(api_secret_, query);
    const std::string body = query + "&signature=" + sig;
    out = http_request(host_, path, "POST", body,
                       {"X-MBX-APIKEY: " + api_key_}, timeout_ms_);
    if (!out.ok()) {
        err = api_error(out);
        return false;
    }
    return true;
}

bool TestnetBroker::fetch_exchange_info_(std::string& err) {
    const auto r = http_request(host_, "/fapi/v1/exchangeInfo", "GET", "",
                                {}, timeout_ms_);
    if (!r.ok()) {
        err = api_error(r);
        return false;
    }
    try {
        const auto j = nlohmann::json::parse(r.body);
        rules_.clear();
        for (const auto& s : j.at("symbols")) {
            const std::string sym = s.at("symbol").get<std::string>();
            SymbolRules rl;
            bool found = false;
            for (const auto& f : s.at("filters")) {
                if (f.at("filterType").get<std::string>() == "LOT_SIZE") {
                    rl = rules_from_lot(f.at("stepSize").get<std::string>(),
                                        f.at("minQty").get<std::string>());
                    found = true;
                    break;
                }
            }
            if (found) rules_[sym] = rl;
        }
    } catch (const nlohmann::json::exception& e) {
        err = "exchangeInfo parse: " + std::string(e.what());
        return false;
    }
    return true;
}

namespace {

// Balance fields arrive as JSON strings ("10000.00000000") or numbers.
bool num_or_str(const nlohmann::json& v, double& out) {
    if (v.is_number()) {
        out = v.get<double>();
        return true;
    }
    if (v.is_string()) {
        out = std::strtod(v.get_ref<const std::string&>().c_str(), nullptr);
        return true;
    }
    return false;
}

}  // namespace

bool TestnetBroker::fetch_account_(AccountInfo& out, std::string& err) {
    HttpResponse r;
    if (!signed_get("/fapi/v2/account", "", r, err)) return false;
    try {
        const auto j = nlohmann::json::parse(r.body);
        if (!num_or_str(j.at("totalWalletBalance"), out.wallet_balance) ||
            !num_or_str(j.at("totalMarginBalance"), out.margin_balance) ||
            !num_or_str(j.at("totalUnrealizedProfit"), out.unrealized)) {
            err = "account parse: non-numeric balance field";
            return false;
        }
    } catch (const nlohmann::json::exception& e) {
        err = "account parse: " + std::string(e.what());
        return false;
    }
    return true;
}

bool TestnetBroker::connect(std::string& err) {
    if (!fetch_exchange_info_(err)) return false;
    std::fprintf(stderr, "testnet: exchangeInfo cached (%zu symbols)\n",
                 rules_.size());
    return true;
}

bool TestnetBroker::setup_symbols(const std::vector<std::string>& syms,
                                  std::string& err) {
    err.clear();
    for (const auto& sym : syms) {
        // 1x leverage so notional ≈ margin; ISOLATED so one symbol's paper
        // PnL can never touch another's margin.
        bool ok = false;
        std::string e;
        for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
            HttpResponse r;
            if (signed_post("/fapi/v1/leverage", "symbol=" + sym + "&leverage=1",
                            r, e) &&
                signed_post("/fapi/v1/marginType",
                            "symbol=" + sym + "&marginType=ISOLATED", r, e))
                ok = true;
            else if (e.find("already") != std::string::npos ||
                     e.find("-4046") != std::string::npos)
                ok = true;   // idempotent: already configured (e.g. -4046
                             // "No need to change margin type.")
        }
        if (!ok) {
            bad_setup_.insert(sym);
            err += (err.empty() ? "" : "; ") + sym + ": " + e;
        }
    }
    return bad_setup_.empty();
}

bool TestnetBroker::rules(const std::string& sym, SymbolRules& out) {
    const auto it = rules_.find(sym);
    if (it == rules_.end()) return false;
    out = it->second;
    return true;
}

bool TestnetBroker::account(AccountInfo& out, std::string& err) {
    return fetch_account_(out, err);
}

bool TestnetBroker::positions(std::vector<PosRisk>& out, std::string& err) {
    HttpResponse r;
    if (!signed_get("/fapi/v2/positionRisk", "", r, err)) return false;
    try {
        const auto j = nlohmann::json::parse(r.body);
        out.clear();
        for (const auto& p : j) {
            if (!p.is_object()) continue;
            const double qty = std::strtod(
                p.value("positionAmt", "0").c_str(), nullptr);
            if (qty == 0.0) continue;
            PosRisk pr;
            pr.symbol = p.value("symbol", "");
            pr.qty = qty;
            pr.entry_price =
                std::strtod(p.value("entryPrice", "0").c_str(), nullptr);
            pr.unrealized =
                std::strtod(p.value("unrealizedProfit", "0").c_str(), nullptr);
            out.push_back(std::move(pr));
        }
    } catch (const nlohmann::json::exception& e) {
        err = "positionRisk parse: " + std::string(e.what());
        return false;
    }
    return true;
}

bool TestnetBroker::place(const std::vector<OrderSpec>& orders,
                          std::vector<FillReport>& out, std::string& err) {
    if (orders.empty()) return true;
    // Append, never clear: a retry after a partial failure must not lose
    // the fills of the orders the first attempt already acked.

    // Never resend an id that has already been acked this session.
    std::vector<OrderSpec> fresh;
    for (const auto& o : orders) {
        if (acked_ids_.count(o.client_id)) continue;
        if (bad_setup_.count(o.symbol)) continue;
        fresh.push_back(o);
    }
    if (fresh.empty()) return true;

    // Binance FUTURES batchOrders caps a batch at 5 orders (-4082 beyond)
    // and takes the batch as a form parameter `batchOrders=<json string>`
    // (the raw-JSON body is the SPOT convention). Params, encoding, and
    // signing all live in the body, exactly like the leverage/marginType
    // calls that already work.
    constexpr std::size_t kMaxBatch = 5;
    for (std::size_t off = 0; off < fresh.size(); off += kMaxBatch) {
        const std::size_t n = std::min(kMaxBatch, fresh.size() - off);
        nlohmann::json batch = nlohmann::json::array();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& o = fresh[off + i];
            batch.push_back({{"symbol", o.symbol},
                             {"side", o.buy ? "BUY" : "SELL"},
                             {"type", "MARKET"},
                             {"quantity", o.qty_str},
                             {"newClientOrderId", o.client_id}});
        }
        const std::string json_body = batch.dump();
        std::string enc;
        for (const unsigned char c : json_body) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
                enc += static_cast<char>(c);
            else {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%%%02X", static_cast<unsigned>(c));
                enc += buf;
            }
        }
        std::string form = "batchOrders=" + enc + "&recvWindow=" +
                           std::to_string(kRecvWindowMs) +
                           "&timestamp=" + std::to_string(unix_ms());
        const std::string sig = hmac_sha256_hex(api_secret_, form);

        HttpResponse r;
        r = http_request(host_, "/fapi/v1/batchOrders", "POST",
                         form + "&signature=" + sig,
                         {"X-MBX-APIKEY: " + api_key_}, timeout_ms_);
        if (!r.ok()) {
            err = api_error(r);
            return false;
        }
        try {
            const auto j = nlohmann::json::parse(r.body);
            if (!j.is_array()) {
                // An error object ({"code":...,"msg":...}) instead of the
                // order array — surface the API message.
                if (j.is_object() && j.contains("msg"))
                    err = api_error(r);
                else
                    err = "batchOrders: unexpected response shape";
                return false;
            }
            for (const auto& o : j) {
                FillReport f;
                f.client_id = o.value("clientOrderId", "");
                f.symbol = o.value("symbol", "");
                f.status = o.value("status", "MISSING");
                if (f.status != "FILLED" && f.status != "NEW" &&
                    f.status != "PARTIALLY_FILLED")
                    f.status = "REJECTED";
                f.qty = std::strtod(o.value("executedQty", "0").c_str(), nullptr);
                f.avg_price =
                    std::strtod(o.value("avgPrice", "0").c_str(), nullptr);
                if (f.avg_price <= 0.0) f.avg_price = kLiveNaN;
                if (!f.client_id.empty()) acked_ids_.insert(f.client_id);
                out.push_back(std::move(f));
            }
        } catch (const nlohmann::json::exception& e) {
            err = "batchOrders parse: " + std::string(e.what());
            return false;
        }
    }
    return true;
}

}  // namespace qse::live
