# 📈 Multi-Agent Options Market Analyzer

> An autonomous, multi-agent AI pipeline that analyzes live options-market data, recommends a risk-defined trade, has it independently reviewed by a second LLM, and delivers the result to Slack — orchestrated end-to-end with **LangGraph**.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 Overview

This project demonstrates a production-style **multi-agent system** where specialized AI agents collaborate through a directed graph. It pulls **live options data** for a watchlist of stocks and ETFs, computes professional-grade technical and volatility metrics in Python, and then uses LLMs to make and critique a single, rules-compliant options recommendation — fully autonomously, with no human in the loop.

The design intentionally separates **deterministic quantitative computation** (done in Python) from **judgment and reasoning** (done by the LLMs), keeping the AI calls cheap, fast, and auditable.

## 🏗️ Architecture

```
        ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
START ─▶│   Analyzer    │ ──▶ │   Reviewer    │ ──▶ │    Notifier   │ ─▶ END
        │   (Gemini)    │     │    (Groq)     │     │    (Slack)    │
        └───────────────┘     └───────────────┘     └───────────────┘
         recommends best       independent risk        posts result
         30–45 DTE trade        review & verdict        to Slack
```

The pipeline is a **LangGraph `StateGraph`** with three nodes passing a shared typed state:

| Node | Engine | Responsibility |
|------|--------|----------------|
| **Analyzer** | Google Gemini 2.5 Flash | Fetches live data, computes indicators, recommends the single best trade |
| **Reviewer** | Groq (Llama 3.3 70B) | Independently validates the trade against strict strategy / DTE / risk rules |
| **Notifier** | Slack (Webhook or Bot API) | Formats and delivers the recommendation + review to a Slack channel |

## ✨ Key Features

- **Multi-agent orchestration** with LangGraph — clean, inspectable node/edge DAG.
- **Live market data** via `yfinance` — real quotes, options chains, and earnings dates.
- **Quant engine in Python** — RSI-14, MACD(12,26,9), SMA-20/50, trend classification, Black-Scholes deltas, and an IV Rank / IV Percentile approximation from 1-year historical volatility.
- **Cost-efficient LLM usage** — all tool results are pre-computed and packed into a *single* analyzer LLM call instead of a chatty agent loop.
- **Independent review agent** — a *different* model critiques the first agent's output to reduce single-model bias.
- **Strict rule enforcement** — only 7 risk-defined strategies, a hard 30–45 DTE window, IV-Rank-to-strategy matching, and no earnings inside the expiry.
- **Slack delivery** with automatic Markdown → Slack `mrkdwn` conversion and Block Kit formatting.
- **A2A-ready** — exposed over Google's Agent-to-Agent protocol with a discoverable Agent Card, so other agents can find and call it.
- **Resilient by design** — every node degrades gracefully; a failure in one stage is captured and passed downstream rather than crashing the pipeline.

## 📊 Supported Strategies

The agents are constrained to **7 risk-defined options strategies**, matched to the volatility environment:

| Strategy | Bias | Volatility Fit |
|----------|------|----------------|
| Long Call | Bullish | Low IV |
| Long Put | Bearish | Low IV |
| Bull Call Spread | Bullish | Low IV (debit) |
| Bear Put Spread | Bearish | Low IV (debit) |
| Bull Put Spread | Bullish | High IV (credit) |
| Bear Call Spread | Bearish | High IV (credit) |
| Iron Condor | Neutral / Ranging | High IV (credit) |

**Watchlist:** `SOFI`, `NVDA`, `TSLA`, `AAPL`, `SPY`, `QQQ`

## 🛠️ Tech Stack

- **Orchestration:** LangGraph (`StateGraph`)
- **Agent framework:** LangChain
- **LLMs:** Google Gemini (analyzer) · Groq / Llama 3.3 70B (reviewer)
- **Market data:** yfinance
- **Quant:** NumPy · pandas · SciPy
- **Delivery:** Slack (Incoming Webhook or Bot Web API)
- **Interop:** A2A protocol server via FastAPI + Uvicorn (Agent Card + JSON-RPC)
- **Config:** python-dotenv

## 📁 Project Structure

```
multi-agents/market/
├── orchestrator.py     # LangGraph pipeline wiring all three agents
├── analyzer.py         # Agent 1 — live data, quant engine, Gemini recommendation
├── reviewer.py         # Agent 2 — independent Groq-based trade review
├── agent_factory.py    # Reusable LLM builders (Gemini, Groq, Grok, etc.)
├── slack_notifier.py   # Slack delivery + Markdown→mrkdwn formatting
├── a2a_server.py       # A2A protocol server + Agent Card (FastAPI)
└── README.md
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-core langchain-google-genai \
            langchain-openai langchain-groq langchain-community langgraph \
            yfinance numpy pandas scipy python-dotenv
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
# Analyzer (Google Gemini)
GOOGLE_API_KEY=your_gemini_key

# Reviewer (Groq — free tier)
GROQ_API_KEY=your_groq_key

# Slack delivery (choose ONE)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
# or:
# SLACK_BOT_TOKEN=xoxb-...
# SLACK_CHANNEL=#trading-alerts
```

| Provider | Where to get a key |
|----------|--------------------|
| Google Gemini | https://aistudio.google.com/apikey |
| Groq | https://console.groq.com/keys |
| Slack Webhook | https://api.slack.com/messaging/webhooks |

### 3. Run the pipeline

```bash
python multi-agents/market/orchestrator.py
```

The pipeline will fetch live data, generate a recommendation, review it, print everything to the console, and post the result to Slack.

## 🌐 A2A Server (Google Agent-to-Agent Protocol)

The agent is also exposed over the **[A2A protocol](https://a2a-protocol.org/)**, so other agents and A2A clients can discover and call it through a standard **Agent Card**.

### Start the server

```bash
python multi-agents/market/a2a_server.py
```

| Endpoint | Purpose |
|----------|---------|
| `GET /.well-known/agent-card.json` | **Agent Card** (discovery) — current A2A spec |
| `GET /.well-known/agent.json` | Agent Card (legacy path) |
| `POST /` | JSON-RPC 2.0 endpoint — method `message/send` |
| `GET /health` | Health check |

### View the Agent Card

```bash
curl http://localhost:8000/.well-known/agent-card.json
```

```jsonc
{
  "protocolVersion": "0.3.0",
  "name": "Options Market Analyzer",
  "description": "Autonomous multi-agent options-trading analyst ...",
  "url": "http://localhost:8000/",
  "preferredTransport": "JSONRPC",
  "version": "1.0.0",
  "capabilities": { "streaming": false, "pushNotifications": false },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {
      "id": "analyze-options-trade",
      "name": "Options Trade Analysis & Review",
      "tags": ["finance", "options", "trading", "multi-agent"]
    }
  ]
}
```

### Call the agent (A2A `message/send`)

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Recommend a trade"}],
        "messageId": "req-1"
      }
    }
  }'
```

The response is an A2A agent `message` whose text contains the analyzer recommendation plus the reviewer verdict.

> **Tip:** Set `A2A_PUBLIC_URL` (e.g. to an ngrok/Cloudflare tunnel URL) so the Agent Card advertises a publicly reachable address. Configure host/port with `A2A_HOST` / `A2A_PORT`.

## 📤 Sample Output


```
📈 Options Pipeline Result

Analyzer Recommendation
NVDA Iron Condor — 36 DTE
  Sell 750 Put / Buy 749 Put  ·  Net credit $4.05
  Risk/Reward 4.26:1 · IV Rank 65.2 · Trend: ranging

Reviewer Verdict
✅ Strategy permitted · ✅ DTE in window · ✅ IV Rank ≥ 50
Verdict: APPROVED WITH NOTES
```

## 🧠 Design Decisions & Lessons Learned

- **Pre-compute, then call once.** Early versions let the agent loop call tools repeatedly, which exhausted free-tier quotas fast. Refactoring to compute all metrics in Python and make a *single* LLM call cut cost dramatically and made runs deterministic.
- **Two models, not one.** Using a different provider for review (Groq) than for analysis (Gemini) provides a genuine second opinion and guards against a single model's blind spots.
- **Graph over glue code.** Modeling the workflow as a LangGraph `StateGraph` makes each stage independently testable and the data flow explicit.
- **Fail soft.** Each node wraps its work in error handling so a quota limit or network blip degrades gracefully instead of taking down the whole run.

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is **not financial advice**. Options trading involves substantial risk of loss. Always do your own research and consult a licensed professional before trading.

---

*Built as a portfolio project to demonstrate multi-agent orchestration, applied quantitative finance, and pragmatic LLM engineering.*
