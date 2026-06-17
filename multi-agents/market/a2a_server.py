"""
A2A (Agent-to-Agent) server for the Options Market Analyzer.

Exposes the multi-agent options pipeline over Google's A2A protocol so other
agents (or the A2A Inspector / any A2A client) can discover and call it.

Endpoints
---------
GET  /.well-known/agent-card.json   Agent Card (discovery) — current A2A spec
GET  /.well-known/agent.json        Agent Card (discovery) — legacy path
POST /                              JSON-RPC 2.0 endpoint (method: message/send)
GET  /health                        Simple health check

Run:
    python a2a_server.py
    # then open  http://localhost:8000/.well-known/agent-card.json
"""

import os
import sys
import uuid
import importlib.util

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --- make sibling agent modules importable --------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

load_dotenv()


def _load(filename: str):
    """Load a sibling module by filename (handles hyphens in names)."""
    path = os.path.join(_DIR, filename)
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").removesuffix(".py"), path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("analyzer.py")
reviewer = _load("reviewer.py")

# --- configuration --------------------------------------------------------
HOST = os.getenv("A2A_HOST", "0.0.0.0")
# Cloud Run injects PORT (usually 8080); fall back to A2A_PORT then 8000.
PORT = int(os.getenv("PORT", os.getenv("A2A_PORT", "8000")))
# Public base URL advertised in the Agent Card. If unset, it is derived from the
# incoming request (works automatically behind Cloud Run / proxies / tunnels).
PUBLIC_URL = os.getenv("A2A_PUBLIC_URL", "").rstrip("/")

AGENT_VERSION = "1.0.0"
PROTOCOL_VERSION = "0.3.0"


def _base_url(request: Request) -> str:
    """Resolve the public base URL for the Agent Card.

    Priority: explicit A2A_PUBLIC_URL env var, else derive from the incoming
    request (honoring proxy headers used by Cloud Run / load balancers).
    """
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if host:
        return f"{proto}://{host}"
    return str(request.base_url).rstrip("/")


def _agent_card(base_url: str) -> dict:
    """Build the A2A Agent Card describing this agent's identity and skills."""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "name": "Options Market Analyzer",
        "description": (
            "Autonomous multi-agent options-trading analyst. Fetches live market "
            "data for a watchlist (SOFI, NVDA, TSLA, AAPL, SPY, QQQ), recommends the "
            "single best risk-defined options trade in the 30-45 DTE window using "
            "Google Gemini, then has it independently reviewed by a second LLM (Groq)."
        ),
        "url": base_url.rstrip("/") + "/",
        "preferredTransport": "JSONRPC",
        "version": AGENT_VERSION,
        "provider": {
            "organization": "sujith-ai-systems",
            "url": "https://github.com/sujith-ai-systems/ai-agents",
        },
        "documentationUrl": "https://github.com/sujith-ai-systems/ai-agents",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "analyze-options-trade",
                "name": "Options Trade Analysis & Review",
                "description": (
                    "Analyze the live options market for the watchlist and return the "
                    "single best 30-45 DTE trade recommendation, together with an "
                    "independent risk review (verdict + rule-compliance check)."
                ),
                "tags": ["finance", "options", "trading", "analysis", "multi-agent"],
                "examples": [
                    "Analyze the watchlist and recommend the best options trade.",
                    "What's the best 30-45 DTE options setup right now?",
                    "Give me a reviewed options recommendation for NVDA or SPY.",
                ],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
            }
        ],
    }


app = FastAPI(title="Options Market Analyzer — A2A Server", version=AGENT_VERSION)


@app.get("/.well-known/agent-card.json")
def agent_card_current(request: Request):
    """A2A Agent Card (current spec path)."""
    return JSONResponse(_agent_card(_base_url(request)))


@app.get("/.well-known/agent.json")
def agent_card_legacy(request: Request):
    """A2A Agent Card (legacy spec path) for older clients."""
    return JSONResponse(_agent_card(_base_url(request)))


@app.get("/health")
def health():
    return {"status": "ok", "agent": "Options Market Analyzer", "version": AGENT_VERSION}


def _text_from_message(message: dict) -> str:
    """Extract concatenated text from an A2A message's parts."""
    parts = (message or {}).get("parts", []) or []
    chunks = []
    for p in parts:
        # A2A text part uses {"kind": "text", "text": "..."}; tolerate "type" too.
        if p.get("kind") == "text" or p.get("type") == "text":
            chunks.append(p.get("text", ""))
    return "\n".join(c for c in chunks if c).strip()


def _run_pipeline() -> str:
    """Run the analyzer + reviewer and return a combined text result."""
    recommendation = analyzer.run_analysis(verbose=False)
    if isinstance(recommendation, list):
        recommendation = "\n".join(str(x) for x in recommendation)

    try:
        review = reviewer.run_review(recommendation, verbose=False)
        if isinstance(review, list):
            review = "\n".join(str(x) for x in review)
    except Exception as e:  # reviewer is best-effort
        review = f"[Reviewer unavailable] {type(e).__name__}: {e}"

    return (
        "## Analyzer Recommendation\n\n"
        f"{recommendation}\n\n"
        "## Reviewer Verdict\n\n"
        f"{review}"
    )


def _make_agent_message(text: str) -> dict:
    """Build an A2A agent Message object containing the result text."""
    return {
        "role": "agent",
        "parts": [{"kind": "text", "text": text}],
        "messageId": str(uuid.uuid4()),
        "kind": "message",
    }


def _jsonrpc_error(req_id, code: int, message: str):
    return JSONResponse(
        {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
    )


@app.post("/")
async def jsonrpc(request: Request):
    """A2A JSON-RPC 2.0 endpoint. Supports the 'message/send' method."""
    try:
        body = await request.json()
    except Exception:
        return _jsonrpc_error(None, -32700, "Parse error: invalid JSON")

    req_id = body.get("id")
    method = body.get("method")

    if body.get("jsonrpc") != "2.0" or not method:
        return _jsonrpc_error(req_id, -32600, "Invalid Request")

    if method in ("message/send", "message/stream"):
        params = body.get("params", {}) or {}
        # The user's message text is optional — the pipeline runs over a fixed
        # watchlist regardless, but we accept it for A2A compliance.
        _ = _text_from_message(params.get("message", {}))
        try:
            result_text = _run_pipeline()
        except Exception as e:
            return _jsonrpc_error(req_id, -32603, f"Internal error: {e}")

        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id, "result": _make_agent_message(result_text)}
        )

    return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  Options Market Analyzer — A2A Server")
    print("=" * 60)
    _shown = PUBLIC_URL or f"http://localhost:{PORT}"
    print(f"  Agent Card: {_shown}/.well-known/agent-card.json")
    print(f"  JSON-RPC:   {_shown}/  (method: message/send)")
    print(f"  Listening on http://{HOST}:{PORT}")
    print("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT)
