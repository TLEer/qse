#pragma once

// Loads curated Binance UM monthly dumps into an aligned MarketGrid.
// Expects the layout download_um.sh produces:
//   <data_dir>/klines/<SYM>-1h-<YYYY-MM>.csv
//   <data_dir>/funding/<SYM>-fundingRate-<YYYY-MM>.csv
//   <data_dir>/universe.txt          (one symbol per line, pinned order)

#include <string>
#include <vector>

#include "qse/xsec/grid.hpp"

namespace qse::xsec {

class EpochAligner;

// One symbol per line, pinned order — the universe column order the engine
// (and the live adapter) are indexed by.
std::vector<std::string> load_universe(const std::string& universe_path);

// Appends curated funding prints and klines for the inclusive month range
// [start_ym, end_ym] ("YYYY-MM") into an already-constructed aligner — the
// shared ingest behind the backtest loader and the live adapter's warmup.
// Funding is appended before klines, exactly as the aligner requires.
// Klines whose open_time is at or past the aligner span's end are skipped
// (they would trip the aligner's force-lock path). Missing monthly files
// are tolerated: the aligner forward-fills the gap and flags it invalid.
// Does NOT seal — callers decide when the live row barrier may advance.
// Returns, per symbol, the latest kline open_time appended (ns), or the
// aligner span's start where nothing was appended. The live adapter passes
// this straight to the feed's backfill: a single global watermark would
// skip symbols whose curated files lag behind the fastest symbol's.
std::vector<TsNs> warm_aligner_from_csv(EpochAligner& aligner,
                                        const std::string& data_dir,
                                        const std::string& start_ym,
                                        const std::string& end_ym);

// start/end are inclusive months, "YYYY-MM".
MarketGrid load_um_dir(const std::string& data_dir,
                       const std::string& universe_path,
                       const std::string& start_ym,
                       const std::string& end_ym);

}  // namespace qse::xsec
