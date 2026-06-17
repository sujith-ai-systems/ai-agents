"""
Orchestrator — LangGraph pipeline for the two-agent options workflow.

    START ──▶ analyzer (Gemini) ──▶ reviewer (Groq) ──▶ notify (Slack) ──▶ END

  1. analyzer node (Gemini)          — fetches live options data and recommends the
                                      best 30-45 DTE trade from the watchlist.
  2. reviewer node (Groq) — independently reviews the recommendation:
                                      tries Groq → Ollama → HuggingFace (whichever is available)
  3. notify node (Slack)             — posts the recommendation + review to Slack
                                      (no-op if SLACK_WEBHOOK_URL / SLACK_BOT_TOKEN unset).

Run:
    python orchestrator.py
"""

import sys
import os
import importlib.util
from typing import TypedDict

# Make sure sibling modules are importable when run from any directory
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()


def _load(filename: str):
    """Load a sibling module by filename (handles hyphens in names)."""
    path = os.path.join(_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_").removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("analyzer.py")
reviewer = _load("reviewer.py")
slack = _load("slack_notifier.py")

DIVIDER = "=" * 60


def _extract_content(text) -> str:
    """Normalise LLM response content to a plain string."""
    if isinstance(text, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )
    return str(text)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    recommendation: str   # produced by the analyzer node
    review: str           # produced by the reviewer node


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def analyzer_node(state: PipelineState) -> PipelineState:
    """Stage 1 — Gemini market analyzer."""
    print("[ NODE: analyzer ]  Running Market Analyzer (Gemini)...\n")
    try:
        recommendation = _extract_content(analyzer.run_analysis(verbose=True))
    except Exception as e:
        print(f"\n❌ Analyzer Error: {type(e).__name__}: {str(e)[:200]}")
        recommendation = f"[ANALYZER FAILED] {type(e).__name__}: {str(e)[:200]}"

    print()
    print(DIVIDER)
    print("  ANALYZER OUTPUT")
    print(DIVIDER)
    print(recommendation)
    print()

    return {"recommendation": recommendation}


def reviewer_node(state: PipelineState) -> PipelineState:
    """Stage 2 — Groq independent reviewer."""
    print("[ NODE: reviewer ]  Running Market Reviewer (Groq)...\n")
    try:
        review = _extract_content(
            reviewer.run_review(state["recommendation"], verbose=True)
        )
    except Exception as e:
        print(f"\n❌ Reviewer Error: {type(e).__name__}: {str(e)[:200]}")
        review = f"[REVIEWER FAILED] {type(e).__name__}: {str(e)[:200]}"

    print()
    print(DIVIDER)
    print("  REVIEWER OUTPUT  (Groq independent review)")
    print(DIVIDER)
    print(review)
    print()

    return {"review": review}


def notify_node(state: PipelineState) -> PipelineState:
    """Stage 3 — post the results to Slack (graceful no-op if unconfigured)."""
    print("[ NODE: notify ]  Sending results to Slack...\n")
    try:
        slack.send_to_slack(
            state.get("recommendation", ""),
            state.get("review", ""),
            verbose=True,
        )
    except Exception as e:
        print(f"  ✗ Slack notify error: {type(e).__name__}: {str(e)[:160]}")

    print()
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_pipeline():
    """Compile and return the LangGraph pipeline."""
    graph = StateGraph(PipelineState)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("notify", notify_node)
    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "reviewer")
    graph.add_edge("reviewer", "notify")
    graph.add_edge("notify", END)
    return graph.compile()


def main() -> None:
    print(DIVIDER)
    print("  OPTIONS TRADING PIPELINE  (LangGraph: Gemini → Groq → Slack)")
    print(DIVIDER)
    print()

    pipeline = build_pipeline()
    try:
        final_state = pipeline.invoke({"recommendation": "", "review": ""})
    except Exception as e:
        print(f"\n❌ Pipeline Error: {type(e).__name__}: {str(e)[:200]}")
        final_state = None

    print(DIVIDER)
    print("  Pipeline complete.")
    print(DIVIDER)

    return final_state


if __name__ == "__main__":
    main()
