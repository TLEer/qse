#!/usr/bin/env bash
# Immutable ingest of Binance UM perpetual 1h klines + funding rates for the
# cross-sectional universe in symbols.txt. Same contract as download.sh:
# raw zips checksummed and never modified, curated CSVs read-only.
#
# Two modes:
#   ./download_um.sh 2025-07 2026-06      # inclusive MONTH range (monthly dumps)
#   ./download_um.sh 2026-08-01 2026-08-20  # inclusive DAY range (daily dumps,
#                                          # merged into the month's curated CSV)
#
# The universe (data/curated/um/universe.txt) is pinned to what downloaded,
# but an incremental run NEVER drops existing members: a symbol with a
# listing gap is kept (the engine forward-fills gaps) and a run that
# downloads nothing does not rewrite the universe at all.
#
# Monthly mode probes the first month and the last COMPLETED month (a
# not-yet-finished month has no dump yet — fill it with daily mode).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL=1h
CURL="curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors"
SYMS_FILE="$(dirname "$0")/symbols.txt"
MAX_SYMS=30

start=${1:?start YYYY-MM or YYYY-MM-DD}; end=${2:?end YYYY-MM or YYYY-MM-DD}

mkdir -p "$ROOT/raw/um" "$ROOT/curated/um/klines" "$ROOT/curated/um/funding"

next_month() {
    date -j -v+1m -f %Y-%m-%d "$1-01" +%Y-%m 2>/dev/null \
        || date -d "$1-01 +1 month" +%Y-%m
}
prev_month() {
    date -j -v-1m -f %Y-%m-%d "$1-01" +%Y-%m 2>/dev/null \
        || date -d "$1-01 -1 month" +%Y-%m
}
next_day() {
    date -j -v+1d -f %Y-%m-%d "$1" +%Y-%m-%d 2>/dev/null \
        || date -d "$1 +1 day" +%Y-%m-%d
}
is_date() { [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; }

# ---------------------------------------------------------------------------
# Daily mode: fill the current (or any past) month from daily dumps.
# Days whose month already has a monthly dump are skipped — appending would
# duplicate rows and trip the aligner's regression guard.
#
# The dump host resets connections sporadically, so each file is fetched
# with retry/backoff and a 404 counts as "no dump" (skip, keep going); a
# file that still fails after all attempts is logged and skipped — the
# aligner forward-fills missing bars, so the run survives.
# ---------------------------------------------------------------------------
if is_date "$start"; then
    UNIVERSE="$ROOT/curated/um/universe.txt"
    [[ -f "$UNIVERSE" && -s "$UNIVERSE" ]] || {
        echo "daily mode needs a non-empty universe.txt from a monthly run" >&2
        exit 2
    }
    echo "daily dumps $start .. $end (months with existing monthly dumps are left untouched)"
    mkdir -p "$ROOT/raw/um/daily"
    # A previous interrupted run may have left partial month merges; appending
    # to them would duplicate rows and trip the aligner's regression guard.
    rm -f "$ROOT/curated/um/klines/"*.tmp "$ROOT/curated/um/funding/"*.tmp

    # $1 = url, $2 = output file. 0 = got it, 22 = no dump (404), 1 = failed.
    fetch_daily() {
        local url=$1 out=$2 attempt
        for attempt in 1 2 3 4 5 6 7 8; do
            if curl -fsS --max-time 60 "$url" -o "$out" 2>/dev/null; then
                return 0
            fi
            local code=$?
            if (( code == 22 )); then return 22; fi       # HTTP 404: no dump
            sleep $((attempt * 2))                        # backoff
        done
        return 1
    }

    n_touched=0 n_missing=0 n_failed=0
    while read -r SYM; do
        [[ -z "$SYM" ]] && continue
        day=$start
        sym_touched=0
        while [[ ! $day > $end ]]; do
            m="${day:0:7}"
            if [[ -f "$ROOT/curated/um/klines/${SYM}-${INTERVAL}-${m}.csv" ]]; then
                day=$(next_day "$day")   # month already complete: skip
                continue
            fi
            KBASE="https://data.binance.vision/data/futures/um/daily/klines/${SYM}/${INTERVAL}"
            FBASE="https://data.binance.vision/data/futures/um/daily/fundingRate/${SYM}"
            kf="${SYM}-${INTERVAL}-${day}.zip"
            ff="${SYM}-fundingRate-${day}.zip"
            if [[ ! -f "$ROOT/raw/um/daily/$kf" ]]; then
                case $(fetch_daily "$KBASE/$kf" "$ROOT/raw/um/daily/$kf") in
                    0)   fetch_daily "$KBASE/$kf.CHECKSUM" "$ROOT/raw/um/daily/$kf.CHECKSUM" >/dev/null \
                             && (cd "$ROOT/raw/um/daily" && shasum -a 256 -c "$kf.CHECKSUM" >/dev/null) ;;
                    22)  n_missing=$((n_missing + 1)); day=$(next_day "$day"); continue ;;
                    *)   echo "FAIL $kf (giving up after 8 attempts)"; n_failed=$((n_failed + 1)); continue ;;
                esac
            fi
            # Daily CSVs carry a header; the engine skips non-numeric
            # lines, so concatenation is safe.
            unzip -p "$ROOT/raw/um/daily/$kf" "*.csv" \
                >> "$ROOT/curated/um/klines/${SYM}-${INTERVAL}-${m}.csv.tmp"
            if [[ ! -f "$ROOT/raw/um/daily/$ff" ]]; then
                case $(fetch_daily "$FBASE/$ff" "$ROOT/raw/um/daily/$ff") in
                    0)   fetch_daily "$FBASE/$ff.CHECKSUM" "$ROOT/raw/um/daily/$ff.CHECKSUM" >/dev/null \
                             && (cd "$ROOT/raw/um/daily" && shasum -a 256 -c "$ff.CHECKSUM" >/dev/null) ;;
                    22)  n_missing=$((n_missing + 1)) ;;
                    *)   echo "FAIL $ff (giving up after 8 attempts)"; n_failed=$((n_failed + 1)) ;;
                esac
            fi
            [[ -f "$ROOT/raw/um/daily/$ff" ]] && \
                unzip -p "$ROOT/raw/um/daily/$ff" "*.csv" \
                    >> "$ROOT/curated/um/funding/${SYM}-fundingRate-${m}.csv.tmp"
            sym_touched=1
            day=$(next_day "$day")
            sleep 1   # be gentle: the dump host resets connections when hammered
        done
        if (( sym_touched )); then
            for f in "$ROOT/curated/um/klines/${SYM}-${INTERVAL}-"*.tmp; do
                [[ -f "$f" ]] || continue
                mv "$f" "${f%.tmp}"
                chmod a-w "${f%.tmp}"
            done
            for f in "$ROOT/curated/um/funding/${SYM}-fundingRate-"*.tmp; do
                [[ -f "$f" ]] || continue
                mv "$f" "${f%.tmp}"
                chmod a-w "${f%.tmp}"
            done
            n_touched=$((n_touched + 1))
        fi
    done < "$UNIVERSE"
    echo "done. merged daily dumps into data/curated/um ($n_touched symbols, "
    echo "      $n_missing missing days skipped, $n_failed failed; today's dump appears tomorrow)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Monthly mode.
# ---------------------------------------------------------------------------
[[ "$start" =~ ^[0-9]{4}-[0-9]{2}$ && "$end" =~ ^[0-9]{4}-[0-9]{2}$ ]] || {
    echo "usage: $0 YYYY-MM YYYY-MM | $0 YYYY-MM-DD YYYY-MM-DD" >&2
    exit 2
}

# An end month that is still in progress has no monthly dump yet: probe the
# last COMPLETED month instead, and download only up to it.
NOW_MONTH=$(date +%Y-%m)
last_complete=$end
if [[ "$end" > "$NOW_MONTH" ]]; then last_complete=$NOW_MONTH; fi
if [[ "$end" == "$NOW_MONTH" ]]; then last_complete=$(prev_month "$end"); fi
if [[ "$last_complete" < "$start" ]]; then
    echo "range $start..$end contains no completed month; use daily mode" >&2
    exit 2
fi

# Existing members survive incremental runs even with gaps (the engine
# forward-fills); a run that downloads nothing leaves the universe untouched.
# The universe is processed in TWO passes: existing members first (they can
# never be displaced by new candidates), then new candidates up to the cap.
OLD_UNIVERSE="$ROOT/curated/um/universe.txt"
declare -A OLD_SYMS
if [[ -f "$OLD_UNIVERSE" ]]; then
    while read -r s; do [[ -n "$s" ]] && OLD_SYMS[$s]=1; done < "$OLD_UNIVERSE"
fi
: > "$ROOT/curated/um/universe.txt.tmp"

n_kept=0
fetch_symbol() {   # $1 = symbol: probe, download missing months, add to tmp
    local SYM=$1
    local KBASE="https://data.binance.vision/data/futures/um/monthly/klines/${SYM}/${INTERVAL}"
    local FBASE="https://data.binance.vision/data/futures/um/monthly/fundingRate/${SYM}"
    # HEAD-probe first month and the last completed month; a 404 on either
    # means a listing gap. Returns 1 on gap (caller decides keep/skip).
    if ! curl -fsI "$KBASE/${SYM}-${INTERVAL}-${start}.zip" >/dev/null 2>&1 \
    || ! curl -fsI "$KBASE/${SYM}-${INTERVAL}-${last_complete}.zip" >/dev/null 2>&1; then
        return 1
    fi

    local cur=$start
    while [[ ! $cur > $last_complete ]]; do
        local kf="${SYM}-${INTERVAL}-${cur}.zip"
        if [[ ! -f "$ROOT/raw/um/$kf" ]]; then
            echo "fetching $kf"
            $CURL "$KBASE/$kf" -o "$ROOT/raw/um/$kf"
            $CURL "$KBASE/$kf.CHECKSUM" -o "$ROOT/raw/um/$kf.CHECKSUM"
            (cd "$ROOT/raw/um" && shasum -a 256 -c "$kf.CHECKSUM" >/dev/null)
        fi
        unzip -n -q "$ROOT/raw/um/$kf" -d "$ROOT/curated/um/klines/"
        chmod a-w "$ROOT/curated/um/klines/${SYM}-${INTERVAL}-${cur}.csv"

        local ff="${SYM}-fundingRate-${cur}.zip"
        if [[ ! -f "$ROOT/raw/um/$ff" ]]; then
            echo "fetching $ff"
            if $CURL "$FBASE/$ff" -o "$ROOT/raw/um/$ff"; then
                $CURL "$FBASE/$ff.CHECKSUM" -o "$ROOT/raw/um/$ff.CHECKSUM"
                (cd "$ROOT/raw/um" && shasum -a 256 -c "$ff.CHECKSUM" >/dev/null)
            else
                # No funding dump for this month (TONUSDT 2026-07 is missing
                # on the host): the engine keeps the symbol with NaN funding.
                echo "no funding dump: $ff"
                rm -f "$ROOT/raw/um/$ff"
                cur=$(next_month "$cur")
                continue
            fi
        fi
        unzip -n -q "$ROOT/raw/um/$ff" -d "$ROOT/curated/um/funding/"
        chmod a-w "$ROOT/curated/um/funding/${SYM}-fundingRate-${cur}.csv"

        cur=$(next_month "$cur")
    done
    echo "$SYM" >> "$ROOT/curated/um/universe.txt.tmp"
    n_kept=$((n_kept + 1))
}

# Pass 1: existing members, in their pinned order — kept even on gaps.
while read -r SYM; do
    [[ -z "$SYM" ]] && continue
    if (( n_kept >= MAX_SYMS )); then break; fi
    if fetch_symbol "$SYM"; then
        :
    else
        echo "KEEP $SYM (known gap)"
        echo "$SYM" >> "$ROOT/curated/um/universe.txt.tmp"
        n_kept=$((n_kept + 1))
    fi
done < "$OLD_UNIVERSE"

# Pass 2: new candidates from symbols.txt, only up to the remaining slots.
while read -r SYM; do
    [[ -z "$SYM" ]] && continue
    if (( n_kept >= MAX_SYMS )); then break; fi
    [[ -n "${OLD_SYMS[$SYM]:-}" ]] && continue
    if fetch_symbol "$SYM"; then
        :
    else
        echo "SKIP $SYM (listing gap)"
    fi
done < "$SYMS_FILE"

# Never replace the pinned universe with an empty file.
if [[ -s "$ROOT/curated/um/universe.txt.tmp" ]]; then
    mv "$ROOT/curated/um/universe.txt.tmp" "$ROOT/curated/um/universe.txt"
else
    echo "nothing downloaded; universe.txt left untouched" >&2
    rm -f "$ROOT/curated/um/universe.txt.tmp"
fi
echo "done. universe ($n_kept symbols) -> data/curated/um/universe.txt"
