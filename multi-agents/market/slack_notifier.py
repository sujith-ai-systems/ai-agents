"""
Slack notifier — posts the pipeline results to a Slack channel.

Supports two mechanisms (auto-detected from environment):

  1. Incoming Webhook (simplest)  — set SLACK_WEBHOOK_URL
     Create one at https://api.slack.com/messaging/webhooks

  2. Bot Token (Web API)          — set SLACK_BOT_TOKEN (xoxb-...) and SLACK_CHANNEL
     Create a bot with chat:write scope at https://api.slack.com/apps

If neither is configured, send_to_slack() is a graceful no-op so the
rest of the pipeline keeps working.
"""

import json
import os
import urllib.request
import urllib.error

from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "").strip()
CHANNEL = os.getenv("SLACK_CHANNEL", "").strip()

# Slack truncates very long messages; keep each block well under the 3000-char
# section text limit.
_MAX_BLOCK_CHARS = 2900


def _post_json(url: str, payload: dict, headers: dict | None = None) -> tuple[int, str]:
    """POST a JSON payload and return (status_code, body)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def _to_mrkdwn(text: str) -> str:
    """
    Convert standard Markdown (as produced by the LLMs) into Slack's
    mrkdwn syntax so messages render correctly in Slack.

    Key differences handled:
      - ***bold italic*** / **bold** / __bold__ -> *bold*
      - ~~strike~~ -> ~strike~
      - # Header  -> *Header*  (Slack has no headings)
      - - / * / + bullets -> •
      - [text](url) -> <url|text>
    """
    import re

    if not text:
        return "(empty)"

    out_lines: list[str] = []
    for line in text.splitlines():
        # Headings (#, ##, ###) -> bold line
        m = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
        if m:
            line = f"*{m.group(1).strip()}*"
        else:
            # Bullet markers: "- " or "* " or "+ " -> "• "
            line = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", line)

        # Links [text](url) -> <url|text>
        line = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<\2|\1>", line)
        # Bold-italic ***text*** -> *text* (Slack has no combined marker)
        line = re.sub(r"\*\*\*([^*]+)\*\*\*", r"*\1*", line)
        # Bold: **text** or __text__ -> *text*
        line = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", line)
        line = re.sub(r"__([^_]+)__", r"*\1*", line)
        # Strikethrough: ~~text~~ -> ~text~
        line = re.sub(r"~~([^~]+)~~", r"~\1~", line)

        out_lines.append(line)

    return "\n".join(out_lines)


def _chunk(text: str, size: int = _MAX_BLOCK_CHARS) -> list[str]:
    """Split text into Slack-safe chunks."""
    text = text or "(empty)"
    return [text[i : i + size] for i in range(0, len(text), size)] or ["(empty)"]


def _build_blocks(recommendation: str, review: str) -> list[dict]:
    """Build Slack Block Kit blocks for the recommendation + review."""
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📈 Options Pipeline Result"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Analyzer Recommendation*"},
        },
    ]
    for part in _chunk(_to_mrkdwn(recommendation)):
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": part}}
        )

    blocks.append({"type": "divider"})
    blocks.append(
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Reviewer Verdict*"}}
    )
    for part in _chunk(_to_mrkdwn(review)):
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": part}}
        )

    return blocks


def send_to_slack(recommendation: str, review: str, verbose: bool = True) -> bool:
    """
    Post the pipeline results to Slack.

    Returns True on success, False on failure or if Slack is not configured.
    Never raises — failures are logged and swallowed so the pipeline continues.
    """
    blocks = _build_blocks(recommendation, review)
    fallback_text = "Options Pipeline Result (see attached recommendation & review)"

    # --- Mechanism 1: Incoming Webhook ------------------------------------
    if WEBHOOK_URL:
        status, body = _post_json(
            WEBHOOK_URL, {"text": fallback_text, "blocks": blocks}
        )
        if status == 200:
            if verbose:
                print("  ✓ Posted results to Slack (webhook)")
            return True
        if verbose:
            print(f"  ✗ Slack webhook failed: {status} {body[:120]}")
        return False

    # --- Mechanism 2: Bot Token (Web API) ---------------------------------
    if BOT_TOKEN and CHANNEL:
        status, body = _post_json(
            "https://slack.com/api/chat.postMessage",
            {"channel": CHANNEL, "text": fallback_text, "blocks": blocks},
            headers={"Authorization": f"Bearer {BOT_TOKEN}"},
        )
        ok = False
        try:
            ok = json.loads(body).get("ok", False)
        except json.JSONDecodeError:
            pass
        if status == 200 and ok:
            if verbose:
                print("  ✓ Posted results to Slack (bot token)")
            return True
        if verbose:
            print(f"  ✗ Slack API failed: {status} {body[:160]}")
        return False

    # --- Not configured ----------------------------------------------------
    if verbose:
        print(
            "  ⓘ Slack not configured — set SLACK_WEBHOOK_URL "
            "(or SLACK_BOT_TOKEN + SLACK_CHANNEL) in .env to enable notifications."
        )
    return False


if __name__ == "__main__":
    # Only send a test message when explicitly requested, so running this
    # file by accident doesn't spam the Slack channel.
    import sys

    if "--test" in sys.argv:
        send_to_slack(
            recommendation="*TEST* — NVDA Iron Condor, 36 DTE, $4.05 credit.",
            review="*Verdict:* APPROVED WITH NOTES. This is a test message.",
        )
    else:
        print(
            "slack_notifier loaded OK. "
            "Run with '--test' to send a sample message to Slack."
        )
