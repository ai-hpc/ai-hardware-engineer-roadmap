# Lecture 06 - Transports, Remote Servers & OAuth 2.1

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Lecture 05](Lecture-05.md) | **Next:** [Lecture 07](Lecture-07.md)

---

Up to now every server you built ran as a local subprocess: Claude Desktop (or MCP Inspector) launched your Python process, piped JSON-RPC over its stdin/stdout, and trusted it implicitly because it was *your* process on *your* machine. That model is perfect for a single user with a server on their own laptop, and it is the entire reason `stdio` exists. But the moment you want one server to back many users — a hosted GitHub server, an internal data tool your whole team's agents reach, a SaaS product that exposes MCP — the subprocess model collapses. Nobody is going to launch your process; they are going to send it an HTTP request from the other side of the internet.

That single change — *the client no longer launches the server* — drags two hard problems in behind it. First, **transport**: HTTP is request/response, but MCP is a bidirectional, long-lived session with server-initiated messages (sampling, elicitation, notifications), so you need a wire format that can carry a stream of messages over one HTTP endpoint and correlate them to a session. Second, **authorization**: an open HTTP endpoint that runs tools is an open invitation, so you need to know *who* is calling before you do anything. This lecture covers both, spec-accurately, because getting either one subtly wrong is how MCP servers get breached.

The throughline: **stdio servers inherit the local user's identity and need no network auth; remote HTTP servers must implement OAuth 2.1 as a Resource Server, delegating identity to a separate Authorization Server.** We will name spec dates as we go, because the transport and auth stories both changed materially in 2025, and a stateless rewrite is landing in 2026.

---

## Learning objectives

By the end of this lecture you should be able to:

- **Compare** the `stdio` and Streamable HTTP transports across locality, process ownership, streaming, sessions, scaling, and auth — and explain the single-endpoint design and the `Mcp-Session-Id` header.
- **Serve** a FastMCP server over Streamable HTTP and read a real session-establishment exchange in raw HTTP headers.
- **Explain** why a remote MCP endpoint requires authorization and motivate OAuth 2.1 over a shared API key.
- **Diagram** the full MCP authorization flow — `401` → RFC 9728 Protected Resource Metadata → Authorization Server discovery → authorization-code + PKCE → `Bearer` token — naming each of the three roles.
- **Justify** audience-bound tokens via Resource Indicators (RFC 8707), and connect them forward to the confused-deputy and token-passthrough attacks of [Lecture 07](Lecture-07.md).
- **Validate** an incoming `Bearer` token on the server (issuer, audience, scopes, expiry) and state why `stdio` servers skip all of it.

---

## 1. Transports: stdio vs Streamable HTTP

MCP separates the **protocol** (JSON-RPC 2.0 messages, the lifecycle, the primitives) from the **transport** (how those bytes move between client and server). The spec defines two transports you should use, and one you must not build on.

| Dimension | **stdio** | **Streamable HTTP** |
|---|---|---|
| Locality | Local — same host | Remote — across a network |
| Who launches whom | **Client launches the server** as a subprocess | Server runs independently; **client connects** to a URL |
| Wire format | JSON-RPC over stdin/stdout, **newline-delimited** | JSON-RPC over HTTP POST to a **single endpoint** |
| Server→client streaming | Native (it's just the other pipe) | Server **may upgrade a response to SSE** for multiple messages |
| Sessions | Implicit — one process is one session | Explicit — `Mcp-Session-Id` HTTP header |
| Scaling | One process per client; not shared | One service, many clients; horizontally scalable |
| Auth needed | No — uses **local / ambient** credentials | **Yes** — OAuth 2.1 (covered below) |
| Best for | Local single-user (e.g. Claude Desktop launching a server) | Hosted / multi-user / SaaS servers |

**The single-endpoint design.** Older HTTP-based MCP used two endpoints (one to POST requests, one to hold open an SSE stream for responses). Streamable HTTP collapses that to **one endpoint**. The client POSTs a JSON-RPC message to it; the server then chooses how to reply:

- For a simple request/response (e.g. `tools/call` that returns immediately), it replies with a **single JSON response** — `Content-Type: application/json` — and the exchange is over.
- For anything that needs to send *multiple* messages back (progress notifications, server-initiated sampling/elicitation, a streamed result), it **upgrades the same response to an SSE stream** — `Content-Type: text/event-stream` — and emits a sequence of `event:`/`data:` frames until done.

One URL, two possible response shapes, chosen per request by the server. The client does not decide in advance whether it wants a stream; it POSTs and reads whatever content type comes back.

**Sessions and `Mcp-Session-Id`.** Because many clients share one endpoint, the server needs to tell their conversations apart. On the response to the `initialize` request, a stateful server returns an **`Mcp-Session-Id`** HTTP header. The client then **echoes that header on every subsequent request**. That is the entire session-binding mechanism — a single header, assigned once at initialize, repeated thereafter. (Contrast stdio, where the OS process *is* the session and no id is needed.)

**The deprecated transport — do not build on it.** The original two-endpoint **HTTP+SSE** transport has been **deprecated since spec revision 2025-03-26**. It still appears in older servers and tutorials. Name it only so you recognize and avoid it: **do not build new servers on HTTP+SSE.** Streamable HTTP is the supported remote transport.

**The 2026 direction — stateless.** The sticky-session model above (a server holds per-session state keyed by `Mcp-Session-Id`) makes horizontal scaling awkward: every request for a session must reach the node that holds its state, so you need sticky load balancing. The **2026-07-28 release candidate** pushes a **stateless** core, so servers can run on ordinary HTTP infrastructure and serverless platforms without sticky sessions — any node can serve any request. Track it as the direction of travel; the 2025-11-25 stable spec is what you build against today.

---

## 2. Running a remote server

FastMCP (the official Python SDK's high-level server, covered in [Lecture 05](Lecture-05.md)) speaks Streamable HTTP with a one-line change to how you run it. The tool, resource, and prompt definitions are identical to your stdio server — only the transport changes.

The simplest path is `mcp.run(transport="streamable-http")`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_forecast(city: str) -> str:
    """Return a short forecast for a city."""
    return f"{city}: clear, 22°C"

if __name__ == "__main__":
    # Serves Streamable HTTP on the default host/port at the /mcp path.
    mcp.run(transport="streamable-http")
```

For real deployments you usually want the server **mounted as an ASGI app** so you can run it under a production server (Uvicorn/Gunicorn) behind a reverse proxy, add middleware, and put authorization in front of it:

```python
import uvicorn
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_forecast(city: str) -> str:
    """Return a short forecast for a city."""
    return f"{city}: clear, 22°C"

# The Streamable HTTP endpoint as a mountable ASGI application.
app = mcp.streamable_http_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**The `Mcp-Session-Id` exchange in HTTP headers.** Here is what session establishment looks like on the wire. The client POSTs `initialize` to the single endpoint with no session id yet:

```http
POST /mcp HTTP/1.1
Host: weather.example.com
Content-Type: application/json
Accept: application/json, text/event-stream

{"jsonrpc":"2.0","id":1,"method":"initialize","params":{ ... }}
```

The server creates a session and returns its id in the response header:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Mcp-Session-Id: 1868a90c-7f3b-4e2a-9d11-5c0e2f8a4b6d

{"jsonrpc":"2.0","id":1,"result":{ ... }}
```

From now on, the client **echoes that header on every request** for the life of the session:

```http
POST /mcp HTTP/1.1
Host: weather.example.com
Content-Type: application/json
Accept: application/json, text/event-stream
Mcp-Session-Id: 1868a90c-7f3b-4e2a-9d11-5c0e2f8a4b6d

{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_forecast","arguments":{"city":"Lisbon"}}}
```

Note the `Accept` header on every request lists **both** `application/json` and `text/event-stream` — the client is telling the server "I can take either a single JSON response or an SSE stream," and the server picks per request (Section 1). That is the client side of the single-endpoint design.

---

## 3. Why remote needs auth

A `stdio` server is reachable by exactly one party: the user who launched it. Its security boundary is the operating system. There is nothing to authenticate, because the OS already decided who you are when it let you start the process — the server runs with the user's **ambient, local credentials** and trusts them.

A remote Streamable HTTP server has no such boundary. The instant you bind it to a public address, *anyone who can reach the URL can POST to it.* And MCP servers are not read-only data feeds — they expose **tools**, which **do things**: query a database, hit a downstream API on the caller's behalf, send a message, move money. An unauthenticated MCP endpoint is a remote-code-execution surface wearing a JSON-RPC costume.

The naive fix — a shared API key in an environment variable, checked on each request — fails for a multi-user server in the ways shared secrets always fail:

- **No per-user identity.** Every caller is the same anonymous key-bearer, so the server cannot scope what each user may do, cannot audit who did what, and cannot revoke one user without rotating the key for everyone.
- **The server ends up holding everyone's secret.** If the server needs to act on a downstream service per user, a single shared key means it impersonates *all* users to that service — a confused-deputy waiting to happen (Section 5, and [Lecture 07](Lecture-07.md)).
- **Distribution and rotation are unsolved.** Getting a long-lived secret safely into every client, and rotating it after a leak, is exactly the problem OAuth was built to retire.

So MCP does not invent its own auth. For the HTTP transport it adopts **OAuth 2.1**, the modern, security-hardened profile of OAuth — short-lived bearer tokens, mandatory PKCE, no implicit grant — and assigns each party a standard OAuth role. (`stdio` servers, with their local boundary, **skip all of this** — see Section 6.)

---

## 4. The OAuth 2.1 model in MCP

MCP authorization (HTTP transport only) is defined entirely in terms of standard OAuth roles. The key architectural decision — which landed in the **2025-06 spec revision** — is that the MCP server is **not** the thing that logs users in. It only *checks* tokens. A separate identity provider issues them.

The three roles:

| Role | Who plays it | OAuth role | Responsibility |
|---|---|---|---|
| **MCP client** | The host/agent (Claude, an agent runtime) | **OAuth 2.1 client** | Runs the authorization-code + PKCE flow; obtains and sends the token |
| **MCP server** | Your Streamable HTTP server | **OAuth 2.0 Resource Server** | Advertises *where* to get tokens; validates the `Bearer` token on every request |
| **Authorization Server (IdP)** | A **separate** service (Auth0, Entra, Keycloak, …) | **Authorization Server** | Authenticates the user and **issues** access tokens |

The MCP server's job shrinks to two things: **tell clients where its Authorization Server lives**, and **validate incoming tokens**. It never sees the user's password and never mints a token. That separation is what makes MCP auth composable — you front your server with whatever IdP your org already uses.

**Advertising the Authorization Server — RFC 9728.** A client arriving at your server does not know which IdP to use. The MCP server **MUST** implement **RFC 9728 Protected Resource Metadata (PRM)** — a small JSON document that names the Authorization Server(s) that issue tokens for this resource, served at a well-known URI.

**The full discovery + authorization flow.** Putting the three roles together, here is the end-to-end sequence the first time a client connects to a protected server:

```text
 MCP client (OAuth 2.1)        MCP server (Resource Server)      Authorization Server (IdP)
        │                               │                                  │
        │ 1. POST /mcp (no token)       │                                  │
        │──────────────────────────────▶                                  │
        │                               │                                  │
        │ 2. 401 Unauthorized           │                                  │
        │    WWW-Authenticate: ...      │                                  │
        │      resource_metadata=...    │                                  │
        │◀──────────────────────────────                                  │
        │                               │                                  │
        │ 3. GET Protected Resource Metadata (RFC 9728)                    │
        │──────────────────────────────▶                                  │
        │    { authorization_servers: [ AS ] }                            │
        │◀──────────────────────────────                                  │
        │                               │                                  │
        │ 4. GET AS metadata, then run OAuth 2.1 authorization-code        │
        │    + PKCE  (PKCE REQUIRED)     │                                  │
        │─────────────────────────────────────────────────────────────────▶
        │    user authenticates / consents; client exchanges code          │
        │◀─────────────────────────────────────────────────────────────────
        │    access token (audience-bound to THIS server — RFC 8707)       │
        │                               │                                  │
        │ 5. POST /mcp                  │                                  │
        │    Authorization: Bearer <tok>│                                  │
        │──────────────────────────────▶                                  │
        │    server validates token, serves the request                    │
        │◀──────────────────────────────                                  │
```

Step by step: the client hits the server with no token (1) and gets a **`401 Unauthorized`** carrying a **`WWW-Authenticate`** header that points at the resource-metadata document (2). The client fetches that **Protected Resource Metadata** (3), reads which Authorization Server to use, and runs the **OAuth 2.1 authorization-code flow with PKCE** against the IdP (4) — **PKCE is required**, not optional. It comes back with an access token whose audience is bound to *this* server, and finally retries the original request with that token as a **`Bearer`** credential (5).

**The `WWW-Authenticate` header (step 2).** The `401` tells the client *how* to authenticate by pointing at the metadata:

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer resource_metadata="https://weather.example.com/.well-known/oauth-protected-resource"
```

**The Protected Resource Metadata document (step 3).** The client GETs that URL and reads, at minimum, which Authorization Server(s) issue tokens for this resource:

```json
{
  "resource": "https://weather.example.com/mcp",
  "authorization_servers": [
    "https://login.example-idp.com"
  ],
  "bearer_methods_supported": ["header"],
  "scopes_supported": ["mcp:tools", "mcp:resources"]
}
```

The `authorization_servers` array is the load-bearing field: it is how a client that knows nothing but your URL discovers the IdP and bootstraps the whole flow. From there the client fetches the Authorization Server's *own* metadata (standard OAuth/OIDC discovery) to find its authorization and token endpoints, and runs the code+PKCE exchange.

---

## 5. Resource Indicators & audience binding (RFC 8707)

Step 4 above ended with "an access token whose audience is bound to *this* server." That phrase is doing critical security work, and the mechanism behind it is **Resource Indicators for OAuth 2.0 — RFC 8707**.

When the client asks the Authorization Server for a token, it includes a `resource` parameter naming the specific MCP server it intends to call. The Authorization Server then stamps that identity into the token's **audience** (`aud`). The result is a token that is only valid *for this one server* — it says, in effect, "the bearer is authorized to call `https://weather.example.com/mcp`, and nowhere else."

Why this matters: without audience binding, a token is a generic bearer credential — any server that receives it could **replay it** against a *different* downstream server that accepts the same Authorization Server's tokens, impersonating the user. That is the **confused-deputy / token-passthrough** problem: a server forwards a token it received to some other API, and that API, seeing a valid-looking token, acts on it. Audience binding shuts the door: a correctly validating downstream server rejects any token whose `aud` is not itself, so a passed-through token is dead on arrival.

This is the single most important reason an MCP server must **validate the audience** of every token (Section 6) and must **never** forward a token it received to a third party. We treat the attack itself — token passthrough and the confused deputy — in depth in [Lecture 07](Lecture-07.md); for now, hold onto the rule: **the token is bound to your server, your server checks that binding, and your server does not pass tokens onward.**

---

## 6. Practical: front the server with a real IdP

You do not implement an Authorization Server. You stand up a real IdP — **Auth0**, **Microsoft Entra ID**, or **Keycloak** are the common choices — register your MCP server as a protected resource / API in it, and let it do authentication, consent, token issuance, and refresh. Your MCP server's entire auth responsibility is two things from Section 4: **serve the Protected Resource Metadata** (so clients discover the IdP), and **validate the `Bearer` token on every request.**

Validation is not "is this header non-empty." A correct check verifies four things on every request:

1. **Signature & issuer** — the token is signed by the Authorization Server you trust (verify against its published JWKS), and its `iss` matches that IdP.
2. **Audience** — `aud` is *this* server (the RFC 8707 binding from Section 5). This is the check that defeats token passthrough.
3. **Scopes** — the token carries the scope the requested operation requires (e.g. `mcp:tools` to call a tool).
4. **Expiry** — `exp` is in the future (and `nbf`/`iat` are sane).

A token-validation skeleton you would run as middleware in front of the Streamable HTTP app from Section 2:

```python
import time
import jwt  # PyJWT
from jwt import PyJWKClient

ISSUER = "https://login.example-idp.com/"
AUDIENCE = "https://weather.example.com/mcp"   # MUST equal this server's identity
JWKS_URL = "https://login.example-idp.com/.well-known/jwks.json"

_jwks = PyJWKClient(JWKS_URL)   # fetches & caches the IdP's signing keys


def validate_bearer(auth_header: str | None) -> dict:
    """Validate an incoming Bearer token. Raises on any failure → caller returns 401/403."""
    if not auth_header or not auth_header.startswith("Bearer "):
        raise PermissionError("missing bearer token")          # → 401 + WWW-Authenticate

    token = auth_header.removeprefix("Bearer ").strip()
    signing_key = _jwks.get_signing_key_from_jwt(token).key

    # Signature + issuer + audience + expiry are all enforced here.
    claims = jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        issuer=ISSUER,            # checks iss
        audience=AUDIENCE,        # checks aud — defeats token passthrough (RFC 8707)
        options={"require": ["exp", "iss", "aud"]},
    )

    # Expiry is verified by jwt.decode; this is a defensive belt-and-suspenders check.
    if claims["exp"] <= time.time():
        raise PermissionError("token expired")

    # Scope check — gate the operation on the granted scopes.
    granted = set(claims.get("scope", "").split())
    if "mcp:tools" not in granted:
        raise PermissionError("insufficient scope")            # → 403

    return claims   # identity + scopes for per-user authorization / audit
```

On a missing or malformed token, respond `401 Unauthorized` **with the `WWW-Authenticate` header pointing at your resource metadata** (Section 4) so the client knows how to recover and start the flow. On a valid token with insufficient scope, respond `403 Forbidden`. Use the returned `claims` (subject, scopes) for per-user authorization and audit — this is the per-user identity a shared API key could never give you.

**stdio servers skip all of this.** Everything in Sections 3–6 applies to the **HTTP transport only**. A `stdio` server has no `WWW-Authenticate`, no PRM, no token validation, because it has no network boundary and no remote callers — it runs as a subprocess under the local user and uses that user's **local / ambient credentials** (whatever the launching process already has: env vars, local config, the OS keychain). Bolting OAuth onto a stdio server would be checking a ticket for a door that only its owner can open. Auth is the cost of going remote — which is exactly why you make the stdio-vs-HTTP decision (Section 1) deliberately.

---

## Current as of

June 2026 — pinned to the **2025-11-25** stable MCP specification revision. Key dates referenced: the legacy two-endpoint **HTTP+SSE** transport has been **deprecated since 2025-03-26** (do not build on it); the **Resource Server / Authorization Server split** and the OAuth 2.1 + RFC 9728 + RFC 8707 authorization model landed in the **2025-06** revision. The **2026-07-28 release candidate** moves the core toward a **stateless** design so Streamable HTTP servers scale on ordinary HTTP and serverless infrastructure without sticky sessions. Treat SDK surfaces (FastMCP `run(transport=...)` / ASGI mounting) and IdP integration details as snapshots and verify against your installed `mcp` package version and your Authorization Server's current metadata.
