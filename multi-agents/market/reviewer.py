"""
Market Reviewer Agent powered by Groq (free tier).

Receives the trade recommendation produced by market-analyzer and
performs an independent, critical review.  Call `run_review()` from
the orchestrator.

Uses Groq llama-3.3-70b-versatile (GROQ_API_KEY) as the reviewer LLM.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agent_factory import build_groq_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Reviewer system prompt
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT = """You are a senior options trading risk manager and independent trade reviewer.
Your job is to critically review trade recommendations produced by an automated market-analysis agent.

You MUST evaluate every recommendation against these rules and flag any violation:

STRATEGY RULES
- Only these 7 strategies are permitted: Long Call, Long Put, Bull Call Spread,
  Bear Put Spread, Bull Put Spread, Bear Call Spread, Iron Condor.
- Long Call / Long Put: only valid when trend is strong_uptrend / strong_downtrend
  AND IV Rank <= 40. High-IV debit buying destroys edge.
- Bull Call Spread / Bear Put Spread (debit spreads): only valid when IV Rank <= 35.
- Bull Put Spread / Bear Call Spread (credit spreads): only valid when IV Rank >= 60.
- Iron Condor: only valid when trend is ranging AND IV Rank >= 50.

DTE RULES
- Expiry must be 30–45 DTE from today.  Any expiry outside this window is a violation.
- Never recommend a trade whose expiry overlaps a known earnings announcement.

RISK/REWARD RULES
- Credit spreads must collect at least $0.30 net credit per spread (not worth the risk otherwise).
- Debit spreads must have max reward >= 1.5× max risk.
- Iron Condors must have at least 1.5× reward-to-risk ratio.
- Stop-loss recommendation is required for every trade.

YOUR OUTPUT FORMAT
1. **Summary** — One sentence restating what was recommended.
2. **Rule Compliance Check** — Go through each applicable rule above and mark ✅ PASS or ❌ FAIL.
3. **Risk Assessment** — 3–5 bullet points on key risks (market risk, IV crush/expansion, gamma risk, event risk).
4. **Verdict** — One of: APPROVED / APPROVED WITH NOTES / REVISE / REJECTED
5. **Reviewer Notes** — If APPROVED WITH NOTES or REVISE: list specific changes needed.
   If REJECTED: explain why and suggest what setup would be better.
"""


def run_review(recommendation: str, verbose: bool = True) -> str:
    """
    Review the trade recommendation string using Groq.

    Args:
        recommendation: The full text output from market-analyzer's run_analysis().
        verbose:        When True (default), print progress to stdout.

    Returns:
        The full review text from Groq.
    """
    if verbose:
        print("=" * 60)
        print("  Market Reviewer  (Groq)")
        print("=" * 60)
        print("Querying Groq for independent review...\n")

    llm = build_groq_llm(temperature=0.1)

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Please review the following trade recommendation produced by "
                "an automated options market analysis agent:\n\n"
                f"{recommendation}"
            )
        ),
    ]

    response = llm.invoke(messages)

    content = response.content
    if isinstance(content, list):
        text = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = content

    return text
