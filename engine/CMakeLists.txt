# qse_core — the deterministic cross-sectional backtesting engine.
# Determinism requirements baked into the build:
#   * no -ffast-math (reordering FP ops changes results across builds)
#   * single compilation standard, no platform-conditional code paths

add_library(qse_core STATIC
    src/xsec/aligner.cpp
    src/xsec/ops.cpp
    src/xsec/ic.cpp
    src/xsec/factor_registry.cpp
    src/xsec/portfolio.cpp
    src/xsec/csv_loader.cpp
)
target_include_directories(qse_core PUBLIC include)
target_compile_options(qse_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion
    $<$<CONFIG:Release>:-O3>
)

# LLM-generated xsec factors are compiled in from factors/xsec_generated/;
# hand-written baselines from factors/xsec/. The GLOB is re-run every
# configure; the agent pipeline rebuilds after each new file.
file(GLOB XSEC_FACTORS CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/factors/xsec/*.cpp
     ${CMAKE_SOURCE_DIR}/factors/xsec_generated/*.cpp)

add_executable(qse_xsec src/xsec_main.cpp ${XSEC_FACTORS})
target_link_libraries(qse_xsec PRIVATE qse_core)
target_compile_options(qse_xsec PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# Live paper-trading adapter (qse_live). Links OpenSSL for the minimal HTTPS
# client; nlohmann/json is vendored under third_party/ and included as a
# SYSTEM header so its warnings stay quiet.
# ---------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)

add_library(qse_live_core STATIC
    src/live/http_client.cpp
    src/live/binance_feed.cpp
    src/live/broker.cpp
    src/live/testnet_broker.cpp
    src/live/live_engine.cpp
)
target_include_directories(qse_live_core PUBLIC include
    SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(qse_live_core PUBLIC qse_core OpenSSL::SSL OpenSSL::Crypto)
target_compile_options(qse_live_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion
    $<$<CONFIG:Release>:-O3>
)

add_executable(qse_live src/live_main.cpp ${XSEC_FACTORS})
target_link_libraries(qse_live PRIVATE qse_live_core)
target_compile_options(qse_live PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME xsec_selftest COMMAND qse_xsec --selftest)
add_test(NAME live_selftest COMMAND qse_live --selftest)
