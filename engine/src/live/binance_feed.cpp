#include "qse/live/feed.hpp"

#include <charconv>
#include <cstdio>
#include <string>

#include <nlohmann/json.hpp>

#include "qse/live/http_client.hpp"

namespace qse::live {

namespace {

constexpr TsNs kHourNs = 3'600'000'000'000LL;
constexpr std::int64_t kHourMs = 3'600'000LL;
constexpr std::size_t kMaxKlineLimit = 1500;
constexpr std::size_t kMaxFundingLimit = 1000;

std::string host_of(const std::string& url) {
    const std::size_t p = url.find("://");
    const std::size_t start = (p == std::string::npos) ? 0 : p + 3;
    const std::size_t slash = url.find('/', start);
    return url.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
}

// Binance REST carries prices/volumes/rates as JSON strings (and sometimes
// numbers); accept both.
bool parse_double(const nlohmann::json& v, double& out) {
    if (v.is_number()) {
        out = v.get<double>();
    } else if (v.is_string()) {
        const auto& s = v.get_ref<const std::string&>();
        const char* begin = s.c_str();
        char* end = nullptr;
        out = std::strtod(begin, &end);
        if (end == begin || *end != '\0') return false;
    } else {
        return false;
    }
    return std::isfinite(out);
}

bool parse_i64(const nlohmann::json& v, std::int64_t& out) {
    if (!v.is_number_integer() && !v.is_number_unsigned()) return false;
    out = v.get<std::int64_t>();
    return true;
}

std::string parse_error(const std::string& body, const std::string& what) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (body: %.120s)", what.c_str(),
                  body.empty() ? "<empty>" : body.c_str());
    return buf;
}

// One GET with a single immediate retry on transport failure.
HttpResponse get_retry(const std::string& host, const std::string& target,
                       int timeout_ms) {
    HttpResponse r = http_request(host, target, "GET", "", {}, timeout_ms);
    if (!r.ok()) {
        if (!r.error.empty() || r.status == 0)   // transport-level: retry once
            r = http_request(host, target, "GET", "", {}, timeout_ms);
    }
    return r;
}

}  // namespace

bool parse_klines_body(const std::string& body, std::int64_t now_ms,
                       std::vector<FeedKline>& out) {
    out.clear();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (const nlohmann::json::exception& e) {
        return false;
    }
    if (!j.is_array()) return false;

    for (const auto& raw : j) {
        if (!raw.is_array() || raw.size() < 7) return false;

        std::int64_t open_ms = 0, close_ms = 0;
        double open = 0, high = 0, low = 0, close = 0, volume = 0;
        if (!parse_i64(raw[0], open_ms) || !parse_i64(raw[6], close_ms) ||
            !parse_double(raw[1], open) || !parse_double(raw[2], high) ||
            !parse_double(raw[3], low) || !parse_double(raw[4], close) ||
            !parse_double(raw[5], volume))
            return false;

        // Forming bar: its close_time is at or past now — not knowable yet.
        if (close_ms >= now_ms) continue;
        // The grid is 1-hour; a bar off the grid would throw in the aligner.
        if (open_ms % kHourMs != 0) continue;
        if (open_ms > close_ms) continue;

        FeedKline fk;
        fk.k.open_time = open_ms * 1'000'000LL;
        fk.k.close_time = close_ms * 1'000'000LL;
        fk.k.open = open;
        fk.k.high = high;
        fk.k.low = low;
        fk.k.close = close;
        fk.k.volume = volume;
        fk.k.trade_count = 0;
        if (raw.size() > 7) {
            double q = 0;
            if (!parse_double(raw[7], q)) return false;
            fk.k.quote_volume = q;
        }
        if (raw.size() > 9) {
            double tb = 0;
            if (parse_double(raw[9], tb)) fk.taker_buy = tb;
        }
        out.push_back(std::move(fk));
    }
    // The API returns bars newest-last; enforce ascending for the engine.
    for (std::size_t i = 1; i < out.size(); ++i)
        if (out[i].k.open_time <= out[i - 1].k.open_time) return false;
    return true;
}

bool parse_funding_body(const std::string& body, std::vector<FundingPrint>& out) {
    out.clear();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (const nlohmann::json::exception&) {
        return false;
    }
    if (!j.is_array()) return false;

    for (const auto& raw : j) {
        if (!raw.is_object()) return false;
        std::int64_t time_ms = 0;
        double rate = 0;
        const auto t = raw.find("fundingTime");
        const auto r = raw.find("fundingRate");
        if (t == raw.end() || r == raw.end()) return false;
        if (!parse_i64(*t, time_ms) || !parse_double(*r, rate)) return false;
        out.push_back({time_ms * 1'000'000LL, rate});
    }
    for (std::size_t i = 1; i < out.size(); ++i)
        if (out[i].calc_ts <= out[i - 1].calc_ts) return false;
    return true;
}

BinanceFeed::BinanceFeed(std::vector<std::string> symbols, std::string base_url,
                         int timeout_ms)
    : symbols_(std::move(symbols)), base_url_(std::move(base_url)),
      timeout_ms_(timeout_ms) {
    last_open_ns_.assign(symbols_.size(), 0);
    last_funding_ns_.assign(symbols_.size(), 0);
}

bool BinanceFeed::server_time_offset(std::int64_t& offset_ms, std::string& err) {
    const auto r = get_retry(host_of(base_url_), "/fapi/v1/time", timeout_ms_);
    if (!r.ok()) {
        err = r.error.empty() ? "server time: http " + std::to_string(r.status) : r.error;
        return false;
    }
    try {
        const auto j = nlohmann::json::parse(r.body);
        offset_ms = j.at("serverTime").get<std::int64_t>() - unix_ms();
        return true;
    } catch (const nlohmann::json::exception&) {
        err = "server time: bad body";
        return false;
    }
}

bool BinanceFeed::backfill(const std::vector<TsNs>& after_ns, FeedEvents& out,
                           std::string& err) {
    const std::string host = host_of(base_url_);
    const std::int64_t now_ms = unix_ms();
    out.klines.assign(symbols_.size(), {});
    out.funding.assign(symbols_.size(), {});
    out.errors.assign(symbols_.size(), {});
    err.clear();
    bool all_ok = true;
    int consecutive_fail = 0;

    for (std::size_t n = 0; n < symbols_.size(); ++n) {
        const TsNs after = n < after_ns.size() ? after_ns[n] : 0;
        // Klines: page with endTime until we cover the gap or hit now.
        std::vector<FeedKline> all;
        std::int64_t end_ms = now_ms - 1;   // inclusive endTime: last complete hour
        for (;;) {
            char target[512];
            std::snprintf(target, sizeof(target),
                          "/fapi/v1/klines?symbol=%s&interval=1h&limit=%zu&endTime=%lld",
                          symbols_[n].c_str(), kMaxKlineLimit,
                          static_cast<long long>(end_ms));
            const auto r = get_retry(host, target, timeout_ms_);
            if (!r.ok()) {
                out.errors[n] = "backfill klines http " + std::to_string(r.status);
                err += (err.empty() ? "" : "; ") + symbols_[n] + ": " + out.errors[n];
                all_ok = false;
                if (++consecutive_fail >= 3) {
                    err += "; aborting backfill (network unreachable?)";
                    return all_ok;
                }
                break;
            }
            consecutive_fail = 0;
            std::vector<FeedKline> page;
            if (!parse_klines_body(r.body, now_ms, page)) {
                out.errors[n] = parse_error(r.body, "bad klines body");
                err += (err.empty() ? "" : "; ") + symbols_[n] + ": " + out.errors[n];
                all_ok = false;
                if (++consecutive_fail >= 3) {
                    err += "; aborting backfill (network unreachable?)";
                    return all_ok;
                }
                break;
            }
            all.insert(all.end(), page.begin(), page.end());
            if (page.empty() || page.size() < kMaxKlineLimit) break;
            end_ms = static_cast<std::int64_t>(page.front().k.open_time / 1'000'000LL) - 1;
        }
        if (!all.empty() && all_ok) {
            // Cut to the requested window (ascending, strictly after after_ns).
            for (const auto& fk : all)
                if (fk.k.open_time > after) out.klines[n].push_back(fk);
            if (!out.klines[n].empty())
                last_open_ns_[n] = out.klines[n].back().k.open_time;
        } else if (all_ok) {
            last_open_ns_[n] = after;   // nothing newer exists yet
        }

        // Funding: single fetch, plenty of prints per 1000.
        if (!all_ok) continue;
        {
            char target[512];
            std::snprintf(target, sizeof(target),
                          "/fapi/v1/fundingRate?symbol=%s&limit=%zu",
                          symbols_[n].c_str(), kMaxFundingLimit);
            const auto r = get_retry(host, target, timeout_ms_);
            if (!r.ok()) {
                out.errors[n] = "backfill funding http " + std::to_string(r.status);
                err += (err.empty() ? "" : "; ") + symbols_[n] + ": " + out.errors[n];
                all_ok = false;
                continue;
            }
            std::vector<FundingPrint> prints;
            if (!parse_funding_body(r.body, prints)) {
                out.errors[n] = parse_error(r.body, "bad funding body");
                err += (err.empty() ? "" : "; ") + symbols_[n] + ": " + out.errors[n];
                all_ok = false;
                continue;
            }
            for (const auto& p : prints) {
                if (p.calc_ts > after) {
                    out.funding[n].push_back(p);
                    last_funding_ns_[n] = p.calc_ts;
                }
            }
        }
    }
    return all_ok;
}

bool BinanceFeed::poll(std::int64_t now_ms, FeedEvents& out, std::string& err) {
    const std::string host = host_of(base_url_);
    const std::int64_t now_ns = now_ms * 1'000'000LL;
    out.klines.assign(symbols_.size(), {});
    out.funding.assign(symbols_.size(), {});
    out.errors.assign(symbols_.size(), {});
    err.clear();
    bool all_ok = true;
    int consecutive_fail = 0;

    for (std::size_t n = 0; n < symbols_.size(); ++n) {
        if (consecutive_fail >= 3) {
            // Transport failures across symbols: the network is down. Do
            // not grind through the remaining symbols at ~10s apiece.
            out.errors[n] = "network unreachable (aborting tick)";
            all_ok = false;
            continue;
        }
        // Funding first: prints must reach the aligner before the kline
        // that locks their row.
        {
            char target[256];
            std::snprintf(target, sizeof(target),
                          "/fapi/v1/fundingRate?symbol=%s&limit=10",
                          symbols_[n].c_str());
            const auto r = get_retry(host, target, timeout_ms_);
            if (!r.ok()) {
                out.errors[n] = "funding http " + std::to_string(r.status);
                all_ok = false;
                if (r.status == 0) ++consecutive_fail;   // transport-level
                else consecutive_fail = 0;
                continue;
            }
            consecutive_fail = 0;
            std::vector<FundingPrint> prints;
            if (!parse_funding_body(r.body, prints)) {
                out.errors[n] = parse_error(r.body, "bad funding body");
                all_ok = false;
                continue;
            }
            for (const auto& p : prints) {
                if (p.calc_ts <= last_funding_ns_[n]) continue;
                if (p.calc_ts > now_ns) continue;    // never a future print
                out.funding[n].push_back(p);
                last_funding_ns_[n] = p.calc_ts;
            }
        }

        {
            char target[256];
            std::snprintf(target, sizeof(target),
                          "/fapi/v1/klines?symbol=%s&interval=1h&limit=20",
                          symbols_[n].c_str());
            const auto r = get_retry(host, target, timeout_ms_);
            if (!r.ok()) {
                out.errors[n] = "klines http " + std::to_string(r.status);
                all_ok = false;
                if (r.status == 0) ++consecutive_fail;
                continue;
            }
            consecutive_fail = 0;
            std::vector<FeedKline> bars;
            if (!parse_klines_body(r.body, now_ms, bars)) {
                out.errors[n] = parse_error(r.body, "bad klines body");
                all_ok = false;
                continue;
            }
            for (const auto& fk : bars) {
                if (fk.k.open_time <= last_open_ns_[n]) continue;
                if (fk.k.open_time > now_ns) continue;   // future bar: skip
                out.klines[n].push_back(fk);
                last_open_ns_[n] = fk.k.open_time;
            }
        }
    }
    return all_ok;
}

}  // namespace qse::live
