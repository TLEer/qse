#include "qse/xsec/loader.hpp"

#include <charconv>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "qse/xsec/aligner.hpp"

namespace qse::xsec {

namespace {

constexpr TsNs kHourNs = 3'600'000'000'000LL;

// Same normalisation as CsvKlineSource: Binance dumps carry s/ms/µs.
TsNs to_ns(std::int64_t t) {
    if (t > 10'000'000'000'000'000LL) return t * 1'000;   // µs
    if (t > 10'000'000'000'000LL) return t * 1'000;       // µs (15 digits)
    if (t > 10'000'000'000LL) return t * 1'000'000;       // ms
    return t * 1'000'000'000;                             // s
}

bool split_next(std::string_view& line, std::string_view& field) {
    if (line.empty()) return false;
    const auto pos = line.find(',');
    if (pos == std::string_view::npos) {
        field = line;
        line = {};
    } else {
        field = line.substr(0, pos);
        line.remove_prefix(pos + 1);
    }
    return true;
}

template <typename T>
T parse(std::string_view s) {
    T v{};
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc{})
        throw std::runtime_error("load_um_dir: bad field '" + std::string(s) + "'");
    return v;
}

struct YM {
    int y{}, m{};
    bool operator<=(const YM& o) const { return y < o.y || (y == o.y && m <= o.m); }
    void next() { if (++m > 12) { m = 1; ++y; } }
    std::string str() const {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%04d-%02d", y, m);
        return buf;
    }
};

YM parse_ym(const std::string& s) {
    if (s.size() != 7 || s[4] != '-')
        throw std::runtime_error("load_um_dir: bad month '" + s + "' (want YYYY-MM)");
    return {std::stoi(s.substr(0, 4)), std::stoi(s.substr(5, 2))};
}

TsNs month_start_ns(const YM& ym) {
    std::tm tm{};
    tm.tm_year = ym.y - 1900;
    tm.tm_mon = ym.m - 1;
    tm.tm_mday = 1;
    return static_cast<TsNs>(timegm(&tm)) * 1'000'000'000LL;
}

}  // namespace

std::vector<std::string> load_universe(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open universe " + path);
    std::vector<std::string> syms;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) syms.push_back(line);
    }
    if (syms.empty()) throw std::runtime_error("empty universe " + path);
    return syms;
}

std::vector<TsNs> warm_aligner_from_csv(EpochAligner& aligner,
                                        const std::string& data_dir,
                                        const std::string& start_ym,
                                        const std::string& end_ym) {
    const auto& symbols = aligner.grid().symbols;
    const YM start = parse_ym(start_ym), end = parse_ym(end_ym);
    if (!(start <= end)) throw std::runtime_error("warm_aligner_from_csv: start after end");
    // Span guard: a kline past the aligner span's end (open_time > end_open,
    // i.e. e >= T) would trip the force-lock path — append treats it as
    // "symbol complete" and locks the rest of the span forward-filled, which
    // must never happen in a live session. Such bars are skipped here
    // instead. A bar exactly at end_open is the last in-span row and is kept.
    const TsNs span_start = aligner.grid().epoch_close[0] - aligner.grid().epoch_ns;
    const TsNs span_end = aligner.grid().epoch_close.back();
    std::vector<TsNs> watermark(symbols.size(), span_start);

    // Funding first: prints must be buffered before kline appends start
    // locking rows (the barrier only advances once every symbol has bars,
    // so all funding is in place by then; the aligner enforces it anyway).
    for (std::size_t n = 0; n < symbols.size(); ++n) {
        for (YM ym = start; ym <= end; ym.next()) {
            std::ifstream f(data_dir + "/funding/" + symbols[n] + "-fundingRate-" +
                            ym.str() + ".csv");
            if (!f) continue;   // tolerated: symbol keeps NaN funding here
            std::string raw;
            while (std::getline(f, raw)) {
                std::string_view line = raw;
                if (line.empty() || line.front() < '0' || line.front() > '9') continue;
                std::string_view fld;
                split_next(line, fld);
                const TsNs ts = to_ns(parse<std::int64_t>(fld));
                split_next(line, fld);                       // funding_interval_hours
                split_next(line, fld);
                aligner.append_funding(n, ts, parse<double>(fld));
            }
        }
    }

    for (std::size_t n = 0; n < symbols.size(); ++n) {
        for (YM ym = start; ym <= end; ym.next()) {
            std::ifstream f(data_dir + "/klines/" + symbols[n] + "-1h-" +
                            ym.str() + ".csv");
            if (!f) continue;   // tolerated: aligner forward-fills the gap
            std::string raw;
            while (std::getline(f, raw)) {
                std::string_view line = raw;
                if (line.empty() || line.front() < '0' || line.front() > '9') continue;

                Kline k;
                std::string_view fld;
                split_next(line, fld); k.open_time  = to_ns(parse<std::int64_t>(fld));
                split_next(line, fld); k.open       = parse<double>(fld);
                split_next(line, fld); k.high       = parse<double>(fld);
                split_next(line, fld); k.low        = parse<double>(fld);
                split_next(line, fld); k.close      = parse<double>(fld);
                split_next(line, fld); k.volume     = parse<double>(fld);
                split_next(line, fld); k.close_time = to_ns(parse<std::int64_t>(fld));
                double taker_buy = kNaN;
                if (split_next(line, fld)) k.quote_volume = parse<double>(fld);
                if (split_next(line, fld)) k.trade_count = parse<std::uint32_t>(fld);
                if (split_next(line, fld)) taker_buy = parse<double>(fld);

                if (k.open_time > span_end) continue;    // past the span
                if (k.open_time > watermark[n]) watermark[n] = k.open_time;
                aligner.append(n, k, taker_buy);
            }
        }
    }
    return watermark;
}

MarketGrid load_um_dir(const std::string& data_dir,
                       const std::string& universe_path,
                       const std::string& start_ym,
                       const std::string& end_ym) {
    const auto symbols = load_universe(universe_path);
    const YM start = parse_ym(start_ym), end = parse_ym(end_ym);
    if (!(start <= end)) throw std::runtime_error("load_um_dir: start after end");

    const TsNs start_open = month_start_ns(start);
    YM after_end = end;
    after_end.next();
    const TsNs end_open = month_start_ns(after_end) - kHourNs;

    EpochAligner aligner(symbols, kHourNs, start_open, end_open);
    warm_aligner_from_csv(aligner, data_dir, start_ym, end_ym);
    aligner.seal();
    return aligner.grid();
}

}  // namespace qse::xsec
