#pragma once

#include <cstdint>

namespace qse {

// All timestamps are nanoseconds since the Unix epoch, UTC.
// A kline's information becomes knowable only at close_time, never
// open_time — this convention is the primary look-ahead-bias defence.
using TsNs = std::int64_t;

struct Kline {
    TsNs open_time{};
    TsNs close_time{};   // event visibility time
    double open{}, high{}, low{}, close{};
    double volume{};
    double quote_volume{};
    std::uint32_t trade_count{};
};

}  // namespace qse
