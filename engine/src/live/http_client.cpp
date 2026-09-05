#include "qse/live/http_client.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/ssl.h>

namespace qse::live {

std::int64_t unix_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

std::string hmac_sha256_hex(const std::string& key, const std::string& msg) {
    unsigned char out[EVP_MAX_MD_SIZE];
    unsigned int out_len = 0;
    if (!HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
              reinterpret_cast<const unsigned char*>(msg.data()), msg.size(),
              out, &out_len))
        return {};
    static const char* hex = "0123456789abcdef";
    std::string s;
    s.reserve(static_cast<std::size_t>(out_len) * 2);
    for (unsigned int i = 0; i < out_len; ++i) {
        s.push_back(hex[out[i] >> 4]);
        s.push_back(hex[out[i] & 0xf]);
    }
    return s;
}

namespace {

// Wait for readability/writability until the deadline; returns the number
// of milliseconds still available, or -1 on timeout/error.
int poll_until(int fd, bool want_write, std::int64_t deadline_ms) {
    for (;;) {
        const std::int64_t now = unix_ms();
        if (now >= deadline_ms) return -1;
        struct pollfd p{};
        p.fd = fd;
        p.events = static_cast<short>(want_write ? POLLOUT : POLLIN);
        const int rc = poll(&p, 1, static_cast<int>(deadline_ms - now));
        if (rc > 0) return static_cast<int>(deadline_ms - unix_ms());
        if (rc < 0 && errno != EINTR) return -1;
    }
}

std::string hex_digest_available() {
    // Collect the most recent OpenSSL error for diagnostics; best effort.
    std::string s;
    unsigned long e = 0;
    while ((e = ERR_get_error()) != 0) {
        char buf[256];
        ERR_error_string_n(e, buf, sizeof(buf));
        if (!s.empty()) s += "; ";
        s += buf;
    }
    return s;
}

}  // namespace

// Decode a Transfer-Encoding: chunked body in place. Returns false on a
// framing error (leaves `body` as the raw payload in that case).
bool decode_chunked_body(std::string& body) {
    std::string out;
    std::size_t i = 0;
    for (;;) {
        // Read the chunk-size line: hex digits, optional ';' extension.
        std::size_t eol = body.find("\r\n", i);
        if (eol == std::string::npos) return false;
        std::size_t size = 0;
        std::size_t j = i;
        bool any = false;
        for (; j < eol; ++j) {
            const char c = body[j];
            if (c == ';') break;                      // chunk extension
            if (c == ' ') continue;                   // tolerated whitespace
            int d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else return false;
            if (size > (std::numeric_limits<std::size_t>::max() >> 4)) return false;
            size = (size << 4) | static_cast<std::size_t>(d);
            any = true;
        }
        if (!any) return false;
        i = eol + 2;
        if (size == 0) {
            // Trailer section ends with a blank line; anything after is
            // ignored (we requested identity, trailers are rare).
            std::size_t end = body.find("\r\n", i);
            if (end == std::string::npos) return false;
            i = end + 2;
            if (i < body.size() && body.compare(i, 2, "\r\n") == 0) i += 2;
            body = out;
            return true;
        }
        if (body.size() < i + size + 2) return false;
        out.append(body, i, size);
        i += size;
        if (body.compare(i, 2, "\r\n") != 0) return false;
        i += 2;
    }
}

namespace {

HttpResponse make_error(std::string what) {
    HttpResponse r;
    r.error = std::move(what);
    return r;
}

}  // namespace

HttpResponse http_request(const std::string& host, const std::string& target,
                          const std::string& method, const std::string& body,
                          const std::vector<std::string>& extra_headers,
                          int timeout_ms) {
    if (timeout_ms <= 0) return make_error("bad timeout");
    const std::int64_t deadline = unix_ms() + timeout_ms;

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* ai = nullptr;
    if (getaddrinfo(host.c_str(), "443", &hints, &ai) != 0 || !ai)
        return make_error("dns: " + host);

    // Try every address the resolver returned (poisoned/broken resolvers
    // often hand back an unroutable first address); each attempt is a
    // non-blocking connect with a hard deadline, so a blackholed IP cannot
    // hold the tick loop hostage for the kernel's default timeout.
    HttpResponse r;
    int fd = -1;
    for (const struct addrinfo* a = ai; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        const int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        const int crc = connect(fd, a->ai_addr, a->ai_addrlen);
        if (crc != 0 && errno != EINPROGRESS) {
            close(fd);
            fd = -1;
            continue;
        }
        if (crc != 0) {
            bool ok = poll_until(fd, true, deadline) >= 0;
            if (ok) {
                int soerr = 0;
                socklen_t len = sizeof(soerr);
                ok = getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) == 0 &&
                     soerr == 0;
            }
            if (!ok) {
                close(fd);
                fd = -1;
                continue;
            }
        }
        fcntl(fd, F_SETFL, flags);   // back to blocking for SSL I/O
        break;
    }
    freeaddrinfo(ai);
    if (fd < 0) {
        r = make_error("connect to " + host + ": timeout or refused");
        return r;
    }

    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        close(fd);
        return make_error("tls: no context");
    }
    SSL_CTX_set_default_verify_paths(ctx);
    SSL* ssl = SSL_new(ctx);
    if (!ssl) {
        SSL_CTX_free(ctx);
        close(fd);
        return make_error("tls: no ssl object");
    }
    SSL_set_tlsext_host_name(ssl, host.c_str());
    SSL_set_fd(ssl, fd);

    for (;;) {
        const int rc = SSL_connect(ssl);
        if (rc == 1) break;
        const int err = SSL_get_error(ssl, rc);
        if (err != SSL_ERROR_WANT_READ && err != SSL_ERROR_WANT_WRITE) {
            r = make_error("tls handshake: " + hex_digest_available());
            goto done;
        }
        if (poll_until(fd, err == SSL_ERROR_WANT_WRITE, deadline) < 0) {
            r = make_error("timeout during TLS handshake");
            goto done;
        }
    }

    {
        // Head + body in one buffer: a Content-Length with no body makes
        // the server wait forever (the bug that broke every signed POST).
        std::string request = method + " " + target + " HTTP/1.1\r\n" +
                              "Host: " + host + "\r\n" +
                              "User-Agent: qse-live/1.0\r\n" +
                              "Connection: close\r\n" +
                              "Accept-Encoding: identity\r\n";
        if (method == "POST") {
            // Default to form-encoding unless the caller supplied its own
            // Content-Type (e.g. application/json for batchOrders) — two
            // Content-Type headers make the server mis-parse the body and
            // signature verification then fails with -1022.
            bool has_ct = false;
            for (const auto& h : extra_headers)
                if (h.rfind("Content-Type:", 0) == 0) has_ct = true;
            if (!has_ct)
                request += "Content-Type: application/x-www-form-urlencoded\r\n";
            request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        }
        for (const auto& h : extra_headers) request += h + "\r\n";
        request += "\r\n";
        request += body;

        std::size_t sent = 0;
        while (sent < request.size()) {
            const int rc = SSL_write(ssl, request.data() + sent,
                                     static_cast<int>(request.size() - sent));
            if (rc <= 0) {
                const int err = SSL_get_error(ssl, rc);
                if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                    if (poll_until(fd, err == SSL_ERROR_WANT_WRITE, deadline) < 0) {
                        r = make_error("timeout writing request");
                        goto done;
                    }
                    continue;
                }
                r = make_error("tls write: " + hex_digest_available());
                goto done;
            }
            sent += static_cast<std::size_t>(rc);
        }
    }

    {
        std::string raw;
        char buf[16384];
        for (;;) {
            if (poll_until(fd, false, deadline) < 0) {
                r = make_error("timeout reading response");
                goto done;
            }
            const int rc = SSL_read(ssl, buf, sizeof(buf));
            if (rc > 0) {
                raw.append(buf, static_cast<std::size_t>(rc));
                continue;
            }
            const int err = SSL_get_error(ssl, rc);
            if (err == SSL_ERROR_WANT_READ) continue;
            if (err == SSL_ERROR_ZERO_RETURN) break;   // clean EOF
            r = make_error("tls read: " + hex_digest_available());
            goto done;
        }

        const std::size_t sep = raw.find("\r\n\r\n");
        if (sep == std::string::npos) {
            r = make_error("malformed response: no header terminator");
            goto done;
        }
        const std::string head = raw.substr(0, sep);
        std::string payload = raw.substr(sep + 4);
        if (head.size() < 12 || head.compare(0, 7, "HTTP/1.") != 0) {
            r = make_error("malformed response: bad status line");
            goto done;
        }
        r.status = std::atoi(head.c_str() + 9);

        bool chunked = false;
        std::size_t content_length = std::string::npos;
        std::size_t pos = 0;
        while (pos < head.size()) {
            const std::size_t eol = head.find("\r\n", pos);
            if (eol == std::string::npos) break;
            const std::string line = head.substr(pos, eol - pos);
            pos = eol + 2;
            const std::size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            if (!val.empty() && val.front() == ' ') val.erase(0, 1);
            for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (key == "transfer-encoding" && val.find("chunked") != std::string::npos)
                chunked = true;
            else if (key == "content-length")
                content_length = static_cast<std::size_t>(std::strtoull(val.c_str(), nullptr, 10));
        }

        if (chunked) {
            if (!decode_chunked_body(payload)) {
                r = make_error("malformed chunked body");
                goto done;
            }
            r.body = std::move(payload);
        } else if (content_length != std::string::npos && payload.size() > content_length) {
            payload.resize(content_length);
            r.body = std::move(payload);
        } else {
            r.body = std::move(payload);
        }
    }

done:
    SSL_shutdown(ssl);          // best effort
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    close(fd);
    return r;
}

}  // namespace qse::live
