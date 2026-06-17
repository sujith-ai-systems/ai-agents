"""
Options Market Analysis Agent using Google Gemini and LangChain.

Autonomously analyzes SOFI, NVDA, TSLA, AAPL, SY, QQQ and recommends
the best options trade using LIVE market data from yfinance.
"""

import warnings
warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from agent_factory import build_gemini_agent

load_dotenv()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _technicals(close: pd.Series) -> dict:
    """Compute RSI-14, MACD(12,26,9), SMA-20, SMA-50 from a price series."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = round((100 - 100 / (1 + gain / loss)).iloc[-1], 1)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig

    if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
        macd_lbl = "bullish_crossover"
    elif hist.iloc[-1] > 0:
        macd_lbl = "bullish"
    elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
        macd_lbl = "bearish_crossover"
    else:
        macd_lbl = "bearish"

    sma20 = round(close.rolling(20).mean().iloc[-1], 2)
    sma50 = round(close.rolling(50).mean().iloc[-1], 2)
    return dict(rsi=rsi, macd=macd_lbl, sma20=sma20, sma50=sma50)


def _trend(price, sma20, sma50, macd_lbl) -> str:
    if price > sma20 > sma50 and "bullish" in macd_lbl:
        return "strong_uptrend"
    if price > sma20 > sma50:
        return "uptrend"
    if sma50 < price < sma20:
        return "pullback_in_uptrend"
    if price < sma20 < sma50 and "bearish" in macd_lbl:
        return "downtrend"
    if price < sma20 and price > sma50:
        return "ranging_above_support"
    return "ranging"


def _bs_delta(S: float, K: float, T: float, sigma: float, opt: str = "call") -> float:
    """Black-Scholes delta (risk-free rate = 0)."""
    if T <= 0 or sigma <= 0:
        return (1.0 if opt == "call" and S > K else 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return round(norm.cdf(d1) if opt == "call" else norm.cdf(d1) - 1, 3)


def _iv_rank(ticker_obj: yf.Ticker, hist: pd.DataFrame):
    """
    Approximate IV Rank/Percentile using current ATM call IV vs
    1-year rolling 30-day historical volatility range.
    Returns (iv_rank, iv_percentile, atm_iv_pct).
    """
    try:
        expiries = ticker_obj.options
        if not expiries:
            return None, None, None
        chain = ticker_obj.option_chain(expiries[0])
        price = hist["Close"].iloc[-1]

        calls = chain.calls.copy()
        calls["_dist"] = (calls["strike"] - price).abs()
        atm_iv = calls.nsmallest(1, "_dist")["impliedVolatility"].iloc[0]
        if pd.isna(atm_iv) or atm_iv == 0:
            return None, None, None

        lr = np.log(hist["Close"] / hist["Close"].shift(1))
        hv30 = (lr.rolling(30).std() * np.sqrt(252)).dropna()
        if len(hv30) < 30:
            return None, None, round(atm_iv * 100, 1)

        hv_min, hv_max = hv30.min(), hv30.max()
        rank = 50.0 if hv_max == hv_min else max(0.0, min(100.0,
            (atm_iv - hv_min) / (hv_max - hv_min) * 100))
        pct = (hv30 < atm_iv).mean() * 100
        return round(rank, 1), round(pct, 1), round(atm_iv * 100, 1)
    except Exception:
        return None, None, None


def _days_to_earnings(ticker_obj: yf.Ticker):
    """Return days until next earnings, or None if unavailable."""
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                return max(0, (pd.to_datetime(dates[0]) - pd.Timestamp.now()).days)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            return max(0, (pd.to_datetime(cal.iloc[0, 0]) - pd.Timestamp.now()).days)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tools — live data via yfinance
# ---------------------------------------------------------------------------

@tool
def get_stock_snapshot(ticker: str) -> str:
    """
    Live market snapshot: price, change %, volume vs average, 52-week range,
    RSI-14, MACD signal, SMA-20/50, trend, next earnings date, and latest news.
    Supported tickers: SOFI, NVDA, TSLA, AAPL, SY, QQQ.
    """
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        hist = t.history(period="1y")
        if hist.empty:
            return f"No price history available for {ticker}."

        price     = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
        prev      = info.get("previousClose") or float(hist["Close"].iloc[-2])
        chg_pct   = (price - prev) / prev * 100
        w52h      = info.get("fiftyTwoWeekHigh", "N/A")
        w52l      = info.get("fiftyTwoWeekLow",  "N/A")
        avg_vol   = info.get("averageVolume", 1)
        vol       = info.get("volume") or int(hist["Volume"].iloc[-1])
        vol_ratio = vol / avg_vol if avg_vol else 0

        tech  = _technicals(hist["Close"])
        trend = _trend(price, tech["sma20"], tech["sma50"], tech["macd"])

        days_earn = _days_to_earnings(t)
        earn_str  = f"{days_earn} days" if days_earn is not None else "N/A"

        try:
            news_list = t.news
            if news_list:
                n = news_list[0]
                headline = (n.get("title") or n.get("content", {}).get("title") or "No headline.")
            else:
                headline = "No recent news."
        except Exception:
            headline = "No recent news."

        return (
            f"=== {ticker.upper()} Live Snapshot ===\n"
            f"Price:         ${price:.2f}  ({chg_pct:+.2f}% today)\n"
            f"Prev Close:    ${prev:.2f}\n"
            f"52-Week Range: ${w52l} - ${w52h}\n"
            f"Volume:        {vol:,} ({vol_ratio:.1f}x avg)\n"
            f"Trend:         {trend}\n"
            f"RSI (14):      {tech['rsi']}  "
            f"{'(overbought)' if tech['rsi'] > 70 else '(oversold)' if tech['rsi'] < 30 else ''}\n"
            f"MACD:          {tech['macd']}\n"
            f"SMA 20/50:     ${tech['sma20']} / ${tech['sma50']}\n"
            f"Next Earnings: {earn_str}\n"
            f"News:          {headline}"
        )
    except Exception as e:
        return f"Error fetching snapshot for {ticker}: {e}"


@tool
def get_options_chain(ticker: str, option_type: str = "both") -> str:
    """
    Live options chain filtered to the 30-45 DTE expiry window.
    Shows ATM +/- 3 strikes for the best matching expiry.
    Includes bid, ask, implied volatility, Black-Scholes delta, and open interest.
    option_type: 'calls', 'puts', or 'both' (default).
    Supported tickers: SOFI, NVDA, TSLA, AAPL, SY, QQQ.
    """
    try:
        t = yf.Ticker(ticker.upper())
        expiries = t.options
        if not expiries:
            return f"No options data available for {ticker}."

        hist  = t.history(period="5d")
        price = float(hist["Close"].iloc[-1])
        today = pd.Timestamp.now()

        # Find expiry closest to the 30-45 DTE window (target 37 DTE)
        target_dte = 37
        best_exp = min(
            expiries,
            key=lambda e: abs((pd.to_datetime(e) - today).days - target_dte)
        )
        dte = (pd.to_datetime(best_exp) - today).days
        T   = max(dte / 365, 0.001)

        lines = [
            f"=== {ticker.upper()} Options Chain (30-45 DTE) ===",
            f"Underlying Price: ${price:.2f}",
            f"Selected Expiry: {best_exp}  ({dte} DTE)", ""
        ]

        chain = t.option_chain(best_exp)

        def fmt(df, opt_type):
            df = df.copy()
            df["_dist"] = (df["strike"] - price).abs()
            nearby = df.nsmallest(7, "_dist").sort_values("strike")
            rows = [f"  [{best_exp}]  {opt_type.upper()}  ({dte} DTE)"]
            rows.append(f"  {'Strike':>8}  {'Bid':>6}  {'Ask':>6}  {'IV':>6}  {'Delta':>7}  {'OI':>8}")
            for _, r in nearby.iterrows():
                iv    = float(r.get("impliedVolatility") or 0)
                oi    = int(r.get("openInterest") or 0)
                bid   = float(r.get("bid") or 0)
                ask   = float(r.get("ask") or 0)
                delta = _bs_delta(price, float(r["strike"]), T, iv,
                                  "call" if opt_type == "calls" else "put")
                rows.append(
                    f"  {r['strike']:>8.2f}  {bid:>6.2f}  {ask:>6.2f}  "
                    f"{iv:>5.0%}  {delta:>+7.3f}  {oi:>8,}"
                )
            rows.append("")
            return rows

        opt = option_type.lower()
        if opt in ("calls", "both"):
            lines.extend(fmt(chain.calls, "calls"))
        if opt in ("puts", "both"):
            lines.extend(fmt(chain.puts, "puts"))

        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching options for {ticker}: {e}"


@tool
def get_spread_candidates(ticker: str) -> str:
    """
    For a given ticker, find the best-fit expiry in the 30-45 DTE window and
    calculate specific spread/condor leg combinations:
      - Bull Call Spread  (debit, bullish)
      - Bear Put Spread   (debit, bearish)
      - Bull Put Spread   (credit, bullish)
      - Bear Call Spread  (credit, bearish)
      - Iron Condor       (credit, neutral/range-bound)
      - Long Call         (debit, strong bullish)
      - Long Put          (debit, strong bearish)
    Returns strikes, bid/ask, net debit/credit, max risk, max reward for each.
    Supported tickers: SOFI, NVDA, TSLA, AAPL, SY, QQQ.
    """
    try:
        t       = yf.Ticker(ticker.upper())
        expiries = t.options
        if not expiries:
            return f"No options data for {ticker}."

        hist  = t.history(period="5d")
        price = float(hist["Close"].iloc[-1])
        today = pd.Timestamp.now()

        # Pick expiry closest to 37 DTE within 30-45 window
        target_dte = 37
        best_exp = min(
            expiries,
            key=lambda e: abs((pd.to_datetime(e) - today).days - target_dte)
        )
        dte = (pd.to_datetime(best_exp) - today).days
        T   = max(dte / 365, 0.001)

        chain  = t.option_chain(best_exp)
        calls  = chain.calls.copy().sort_values("strike").reset_index(drop=True)
        puts   = chain.puts.copy().sort_values("strike").reset_index(drop=True)

        def atm_idx(df):
            return (df["strike"] - price).abs().idxmin()

        def row(df, idx):
            r   = df.iloc[idx]
            bid = float(r.get("bid") or 0)
            ask = float(r.get("ask") or 0)
            iv  = float(r.get("impliedVolatility") or 0)
            oi  = int(r.get("openInterest") or 0)
            return float(r["strike"]), bid, ask, iv, oi

        ci = atm_idx(calls)
        pi = atm_idx(puts)

        # ATM and one-strike OTM/ITM
        c_atm  = row(calls, ci)
        c_otm1 = row(calls, min(ci + 1, len(calls) - 1))
        c_otm2 = row(calls, min(ci + 2, len(calls) - 1))
        p_atm  = row(puts,  pi)
        p_otm1 = row(puts,  max(pi - 1, 0))
        p_otm2 = row(puts,  max(pi - 2, 0))

        # Standard spread width (~5% OTM for wing)
        width_c = round(c_otm1[0] - c_atm[0], 2)
        width_p = round(p_atm[0]  - p_otm1[0], 2)

        lines = [
            f"=== {ticker.upper()} Spread Candidates ===",
            f"Underlying: ${price:.2f}  |  Expiry: {best_exp}  ({dte} DTE)",
            ""
        ]

        # 1. Long Call (debit, strong bullish)
        net = round((c_atm[2] + c_otm1[2]) / 2, 2)  # mid of ATM call ask
        net = round(c_atm[2], 2)
        lines += [
            "[1] LONG CALL  (debit | strong bullish)",
            f"    Buy  {c_atm[0]:.2f} Call  ask=${c_atm[2]:.2f}  IV={c_atm[3]:.0%}  OI={c_atm[4]:,}",
            f"    Net Debit:  ${c_atm[2]:.2f}  |  Max Risk: ${c_atm[2]:.2f}  |  Max Reward: unlimited",
            ""
        ]

        # 2. Long Put (debit, strong bearish)
        lines += [
            "[2] LONG PUT  (debit | strong bearish)",
            f"    Buy  {p_atm[0]:.2f} Put   ask=${p_atm[2]:.2f}  IV={p_atm[3]:.0%}  OI={p_atm[4]:,}",
            f"    Net Debit:  ${p_atm[2]:.2f}  |  Max Risk: ${p_atm[2]:.2f}  |  Max Reward: unlimited",
            ""
        ]

        # 3. Bull Call Spread (debit, bullish)
        debit = round(c_atm[2] - c_otm1[1], 2)
        reward = round(width_c - debit, 2)
        lines += [
            "[3] BULL CALL SPREAD  (debit | bullish)",
            f"    Buy  {c_atm[0]:.2f} Call  ask=${c_atm[2]:.2f}",
            f"    Sell {c_otm1[0]:.2f} Call  bid=${c_otm1[1]:.2f}",
            f"    Net Debit:   ${debit:.2f}  |  Max Risk: ${debit:.2f}  |  Max Reward: ${reward:.2f}  (width ${width_c:.2f})",
            ""
        ]

        # 4. Bear Put Spread (debit, bearish)
        debit_p = round(p_atm[2] - p_otm1[1], 2)
        reward_p = round(width_p - debit_p, 2)
        lines += [
            "[4] BEAR PUT SPREAD  (debit | bearish)",
            f"    Buy  {p_atm[0]:.2f} Put   ask=${p_atm[2]:.2f}",
            f"    Sell {p_otm1[0]:.2f} Put   bid=${p_otm1[1]:.2f}",
            f"    Net Debit:   ${debit_p:.2f}  |  Max Risk: ${debit_p:.2f}  |  Max Reward: ${reward_p:.2f}  (width ${width_p:.2f})",
            ""
        ]

        # 5. Bull Put Spread (credit, bullish)
        credit_bp = round(p_atm[1] - p_otm1[2], 2)
        risk_bp   = round(width_p - credit_bp, 2)
        lines += [
            "[5] BULL PUT SPREAD  (credit | bullish)",
            f"    Sell {p_atm[0]:.2f} Put   bid=${p_atm[1]:.2f}",
            f"    Buy  {p_otm1[0]:.2f} Put   ask=${p_otm1[2]:.2f}",
            f"    Net Credit:  ${credit_bp:.2f}  |  Max Reward: ${credit_bp:.2f}  |  Max Risk: ${risk_bp:.2f}  (width ${width_p:.2f})",
            ""
        ]

        # 6. Bear Call Spread (credit, bearish)
        credit_bc = round(c_atm[1] - c_otm1[2], 2)
        risk_bc   = round(width_c - credit_bc, 2)
        lines += [
            "[6] BEAR CALL SPREAD  (credit | bearish)",
            f"    Sell {c_atm[0]:.2f} Call  bid=${c_atm[1]:.2f}",
            f"    Buy  {c_otm1[0]:.2f} Call  ask=${c_otm1[2]:.2f}",
            f"    Net Credit:  ${credit_bc:.2f}  |  Max Reward: ${credit_bc:.2f}  |  Max Risk: ${risk_bc:.2f}  (width ${width_c:.2f})",
            ""
        ]

        # 7. Iron Condor (credit, neutral)
        # Sell ATM put + buy lower put; sell ATM call + buy higher call
        ic_put_credit  = round(p_atm[1]  - p_otm1[2], 2)
        ic_call_credit = round(c_atm[1]  - c_otm1[2], 2)
        ic_total       = round(ic_put_credit + ic_call_credit, 2)
        ic_risk        = round(max(width_p, width_c) - ic_total, 2)
        lines += [
            "[7] IRON CONDOR  (credit | neutral / range-bound)",
            f"    Sell {p_atm[0]:.2f} Put  bid=${p_atm[1]:.2f}  /  Buy  {p_otm1[0]:.2f} Put  ask=${p_otm1[2]:.2f}",
            f"    Sell {c_atm[0]:.2f} Call bid=${c_atm[1]:.2f}  /  Buy  {c_otm1[0]:.2f} Call ask=${c_otm1[2]:.2f}",
            f"    Net Credit:  ${ic_total:.2f}  |  Max Reward: ${ic_total:.2f}  |  Max Risk: ${ic_risk:.2f}",
            f"    Profit zone: ${p_atm[0]:.2f} – ${c_atm[0]:.2f} at expiry",
            ""
        ]

        return "\n".join(lines)
    except Exception as e:
        return f"Error computing spread candidates for {ticker}: {e}"


@tool
def get_iv_environment(ticker: str) -> str:
    """
    Live IV environment: current ATM implied volatility, IV Rank and IV Percentile
    (derived from 1-year rolling 30-day historical volatility as proxy).
    Returns strategy bias: buy vs sell premium.
    Supported tickers: SOFI, NVDA, TSLA, AAPL, SY, QQQ.
    """
    try:
        t    = yf.Ticker(ticker.upper())
        hist = t.history(period="1y")
        iv_rank, iv_pct, atm_iv = _iv_rank(t, hist)

        if iv_rank is None:
            return f"Could not compute IV data for {ticker}."

        days_earn = _days_to_earnings(t)
        earn_str  = f"{days_earn} days" if days_earn is not None else "N/A"

        if iv_rank >= 60:
            bias = "SELL premium - IV elevated; favor credit spreads, iron condors, covered calls"
        elif iv_rank <= 30:
            bias = "BUY premium - IV low; favor debit spreads, long calls/puts"
        else:
            bias = "NEUTRAL - lean on directional bias; debit or credit both viable"

        return (
            f"=== {ticker.upper()} IV Environment ===\n"
            f"Current ATM IV:  {atm_iv}%\n"
            f"IV Rank (1yr):   {iv_rank} / 100\n"
            f"IV Percentile:   {iv_pct}%\n"
            f"Next Earnings:   {earn_str}\n"
            f"Strategy Bias:   {bias}"
        )
    except Exception as e:
        return f"Error computing IV for {ticker}: {e}"


@tool
def score_option_setup(ticker: str) -> str:
    """
    Score a ticker's options setup out of 10 using live data:
    trend strength, volume surge, RSI momentum, MACD signal, IV rank,
    and earnings proximity risk.
    Supported tickers: SOFI, NVDA, TSLA, AAPL, SY, QQQ.
    """
    try:
        t    = yf.Ticker(ticker.upper())
        info = t.info
        hist = t.history(period="1y")
        if hist.empty:
            return f"No data for {ticker}."

        price     = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
        avg_vol   = info.get("averageVolume", 1)
        vol       = info.get("volume") or int(hist["Volume"].iloc[-1])
        vol_ratio = vol / avg_vol if avg_vol else 1

        tech      = _technicals(hist["Close"])
        trend     = _trend(price, tech["sma20"], tech["sma50"], tech["macd"])
        iv_rank, _, _ = _iv_rank(t, hist)
        days_earn = _days_to_earnings(t)

        score, reasons = 0, []

        if trend == "strong_uptrend":
            score += 2; reasons.append("+2 strong uptrend")
        elif trend in ("uptrend", "pullback_in_uptrend", "ranging_above_support"):
            score += 1; reasons.append(f"+1 {trend}")
        elif trend == "downtrend":
            score += 1; reasons.append("+1 clear downtrend (put opportunity)")

        if vol_ratio >= 1.5:
            score += 2; reasons.append(f"+2 elevated volume ({vol_ratio:.1f}x avg)")
        elif vol_ratio >= 1.1:
            score += 1; reasons.append(f"+1 above-avg volume ({vol_ratio:.1f}x avg)")

        if 55 <= tech["rsi"] <= 70:
            score += 1; reasons.append(f"+1 RSI bullish momentum ({tech['rsi']})")
        elif tech["rsi"] > 70:
            score += 1; reasons.append(f"+1 RSI overbought - strong momentum ({tech['rsi']})")
        elif tech["rsi"] < 35:
            score += 1; reasons.append(f"+1 RSI oversold ({tech['rsi']})")

        if "bullish" in tech["macd"]:
            score += 1; reasons.append(f"+1 MACD {tech['macd']}")
        elif "bearish" in tech["macd"]:
            score += 1; reasons.append(f"+1 MACD {tech['macd']} (put bias)")

        if iv_rank is not None:
            if iv_rank >= 60:
                score += 1; reasons.append(f"+1 high IV rank ({iv_rank}) - premium selling opportunity")
            elif iv_rank <= 25:
                score += 1; reasons.append(f"+1 low IV rank ({iv_rank}) - cheap options to buy")

        if days_earn is not None and 0 <= days_earn < 7:
            score -= 1; reasons.append(f"-1 earnings in {days_earn} days - binary risk")

        score = max(0, min(10, score))
        return (
            f"=== {ticker.upper()} Options Setup Score ===\n"
            f"Score: {score}/10\n"
            f"Live Data: Price=${price:.2f}, RSI={tech['rsi']}, "
            f"MACD={tech['macd']}, IV Rank={iv_rank}, Vol={vol_ratio:.1f}x avg\n"
            f"Breakdown:\n" + "\n".join(f"  {r}" for r in reasons)
        )
    except Exception as e:
        return f"Error scoring {ticker}: {e}"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

WATCHLIST = ["SOFI", "NVDA", "TSLA", "AAPL", "SPY", "QQQ"]

SYSTEM_PROMPT = """You are an expert options trader and market analyst.

Your job is to autonomously analyze a watchlist of stocks and ETFs using live
market data and identify the single best options trade for the 30-45 DTE window.

=== ALLOWED STRATEGIES (choose ONE) ===
1. Long Call          — debit, strong directional bullish
2. Long Put           — debit, strong directional bearish
3. Bull Call Spread   — debit, moderately bullish
4. Bear Put Spread    — debit, moderately bearish
5. Bull Put Spread    — credit, bullish / support-hold
6. Bear Call Spread   — credit, bearish / resistance-hold
7. Iron Condor        — credit, neutral / range-bound (requires IV Rank >= 50)

NO other strategy types are permitted.

=== EXPIRY RULE ===
ALL trades MUST use the expiry in the 30-45 DTE window.
Do NOT recommend trades outside this window.

=== ANALYSIS WORKFLOW ===
1. Call get_stock_snapshot for EVERY ticker in the watchlist.
2. Call get_iv_environment for EVERY ticker.
3. Call score_option_setup for EVERY ticker.
4. Rank tickers. Call get_spread_candidates for the top 2-3.
5. Pick the single best trade from the allowed strategies above.

=== STRATEGY SELECTION RULES ===
- IV Rank >= 60  → prefer credit strategies (Bull Put, Bear Call, Iron Condor)
- IV Rank <= 35  → prefer debit strategies (Long Call/Put, Bull Call, Bear Put)
- Iron Condor    → only when IV Rank >= 50 AND trend is "ranging"
- Long Call/Put  → only when trend is strong_uptrend / downtrend AND IV Rank <= 40
- Avoid any ticker with earnings within the chosen expiry window

=== FINAL RECOMMENDATION FORMAT ===
- **Ticker & Direction**
- **Strategy** (from allowed list only)
- **Expiry** (must be 30-45 DTE)
- **Leg(s)**: Strike, Bid/Ask for each leg
- **Entry**: net debit or credit limit
- **Max Risk / Max Reward**
- **Breakeven price(s)**
- **Target / Stop**
- **Rationale**: 4-5 bullets — technicals, IV environment, volume, DTE logic, catalyst
- **Risk Disclaimer**
"""


def create_finance_agent():
    """Create and return the options market analysis agent."""
    return build_gemini_agent(
        tools=[
            get_stock_snapshot,
            get_iv_environment,
            score_option_setup,
            get_options_chain,
            get_spread_candidates,
        ],
        system_prompt=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Pre-collect all data in Python, then make ONE LLM call
# This avoids burning API quota on agentic tool-call loops.
# ---------------------------------------------------------------------------

def _collect_market_report() -> str:
    """Run all data tools for every ticker and return one consolidated report string."""
    sections = []
    for ticker in WATCHLIST:
        print(f"  Fetching {ticker}...", flush=True)
        sections.append(get_stock_snapshot.invoke({"ticker": ticker}))
        sections.append(get_iv_environment.invoke({"ticker": ticker}))
        sections.append(score_option_setup.invoke({"ticker": ticker}))
        sections.append(get_spread_candidates.invoke({"ticker": ticker}))
        sections.append("")  # blank separator
    return "\n".join(sections)


def run_analysis(verbose: bool = True) -> str:
    """
    Collect live market data for the watchlist and return the LLM trade
    recommendation as a plain string.  Call this from the orchestrator.

    Args:
        verbose: When True (default), print progress to stdout.

    Returns:
        The full trade recommendation text from Gemini.
    """
    if verbose:
        print("=" * 60)
        print("  Market Analyzer  (Gemini + yfinance)")
        print("=" * 60)
        print(f"Watchlist: {', '.join(WATCHLIST)}")
        print("Collecting live market data...\n")

    report = _collect_market_report()

    if verbose:
        print("\nData collected. Querying Gemini (1 API call)...\n")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
    )

    prompt = f"""{SYSTEM_PROMPT}

Below is the complete live market data already gathered for you.
Do NOT call any tools — all data is provided.
Analyze it and output your single best trade recommendation.

{report}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    content = response.content
    if isinstance(content, list):
        text = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = content

    return text


def main():
    result = run_analysis(verbose=True)
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
