#pragma once

// Minimal synchronous HTTPS client over OpenSSL, plus the two small helpers
// the Binance adapters need (HMAC-SHA256 signing, UTC millisecond clock).
//
// The engine ships zero linked dependencies; libcurl is not guaranteed on
// target machines, so this is a ~200-line HTTP/1.1 + TLS client instead.
// Deliberately one-shot (Connection: close, no pooling): the feed polls a
// few endpoints every ~12s, so connection churn is noise, and statelessness
// makes replay mode and failure handling trivial.

#include <cstdint>
#include <string>
#include <vector>

namespace qse::live {

struct HttpResponse {
    int status = 0;         // HTTP status; 0 = transport failure (see error)
    std::string body;       // decoded body (chunk framing stripped)
    std::string error;      // human-readable, empty on success
    bool ok() const { return status >= 200 && status < 300; }
};

// One HTTP/1.1 request over TLS. Never throws. `timeout_ms` is a hard
// wall-clock deadline for the whole exchange (DNS through body). CA
// verification uses the system store; SNI is always sent. The server must
// not compress the response (Accept-Encoding: identity is sent; a gzipped
// body would arrive undecoded — Binance respects this).
HttpResponse http_request(const std::string& host, const std::string& target,
                          const std::string& method, const std::string& body,
                          const std::vector<std::string>& extra_headers,
                          int timeout_ms);

// HMAC-SHA256 of `msg` under `key`, lowercase hex — Binance's request
// signing scheme.
std::string hmac_sha256_hex(const std::string& key, const std::string& msg);

// Decode a Transfer-Encoding: chunked body in place (chunk extensions and
// trailers handled). False on framing errors. Exposed for selftests.
bool decode_chunked_body(std::string& body);

// UTC milliseconds since the Unix epoch (CLOCK_REALTIME).
std::int64_t unix_ms();

}  // namespace qse::live
