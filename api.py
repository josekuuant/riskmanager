"""Risk Manager API — Anthropic Messages API con structured output via tool calling.

Esta API recibe trades + metrics ya calculados por el frontend (TanStack
Start / Lovable) y delega el AUDIT a Claude usando tool calling — Claude
no recalcula, solo analiza los findings y produce el email + recomendación
estructurada (12 campos).

Contract (compatible con riskmanager.functions.ts del frontend):

  POST /audit | /full | /analyze | /narrate   (los 4 son aliases)
  Headers: X-API-Key: <API_SECRET>
  Body JSON:
    {
      "account":  { accountNumber, traderName, accountType, phase,
                    accountSize, initialBalance, ... },
      "preset":   { id, name, dailyLossPercent, maxLossPercent,
                    profitTargetPercent, ... },
      "trades":   [ { ticket, symbol, side, volume, entryPrice,
                      stopLoss, profit, ... }, ... ],
      "metrics":  { totalClosedPnL, bestTradingDay,
                    consistencyPercentage, profitTargetReached,
                    maxClosedLossPercent, tradesWithoutSL, ... }
    }

  Response:
    {
      ok: True,
      data: {
        finalDecision: "PASSED" | "WARNING" | "BREACH" | "MANUAL_REVIEW",
        severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
        executiveSummary: str,
        confirmedBreaches: [{...}],
        estimatedBreaches: [...],
        warnings: [...],
        notEnoughData: [...],
        ruleByRuleAnalysis: [...],
        evidenceTable: [{...}],
        internalRecommendation: "APPROVE PAYOUT" | "REJECT PAYOUT" |
                                "PARTIAL APPROVAL" | "MANUAL REVIEW REQUIRED",
        emailSubject: str,
        emailBody: str
      },
      latencyMs: int,
      model: str,
      usage: {...},
      trades_dropped: int
    }

  GET /health  — sin auth, para healthchecks (no expone secrets)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import anthropic
import httpx
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

API_SECRET = os.environ.get("API_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")

# External APIs
APIFY_API_KEY = os.environ.get("APIFY_API_KEY", "")
APIFY_ECONOMIC_CALENDAR_TOKEN = os.environ.get("APIFY_ECONOMIC_CALENDAR_TOKEN", "")
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
# Mock / synthetic responses are deliberately NOT supported. Every audit is
# a real Anthropic call. If ANTHROPIC_API_KEY is missing or the call fails,
# run_audit raises 503 / 502 / 504 with an explicit error — never a fake
# audit. This is the firm's compliance requirement.

# ─── Anthropic pricing (USD per million tokens) — keep aligned with the
# Anthropic published rate card. cache reads are charged at a discount.
ANTHROPIC_PRICES_PER_M = {
    "claude-opus-4-7": {"in": 5.0, "out": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0, "cache_read": 0.10, "cache_write": 1.25},
}


def _estimate_cost_usd(model: str, usage: dict) -> float:
    """Best-effort cost estimate from token usage. Returns 0 if model unknown."""
    rates = ANTHROPIC_PRICES_PER_M.get(model)
    if not rates:
        return 0.0
    cost = (
        usage.get("input_tokens", 0) * rates["in"]
        + usage.get("output_tokens", 0) * rates["out"]
        + usage.get("cache_read_input_tokens", 0) * rates["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * rates["cache_write"]
    ) / 1_000_000.0
    return round(cost, 6)
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "12000"))
ANTHROPIC_TIMEOUT_SECONDS = float(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "120"))
ANTHROPIC_MAX_RETRIES = int(os.environ.get("ANTHROPIC_MAX_RETRIES", "3"))

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:8080,https://lovable.dev",
    ).split(",")
    if o.strip()
]

# Trades muy grandes saturan el context window. Capamos el sample que mandamos
# al modelo y dejamos los counts en metrics — el system prompt avisa.
MAX_TRADES_IN_PROMPT = int(os.environ.get("MAX_TRADES_IN_PROMPT", "300"))

# Tamaño máximo del JSON body. Por encima rechazamos para evitar OOM en el worker.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(5 * 1024 * 1024)))

# Anti-leak: nunca permitir estos strings en emailBody, sin importar lo que
# devuelva el modelo. Si aparecen, rechazamos la respuesta y devolvemos 502.
EMAIL_FORBIDDEN_SUBSTRINGS = [
    "claude",
    "anthropic",
    "openai",
    "gpt",
    " llm",
    "ai-generated",
    "ai generated",
    "automated by",
    "this is an automated",
    "generated by ai",
    "language model",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("riskmanager")

app = FastAPI(
    title="NYS Risk Manager API",
    description="Payout audit con Claude (Anthropic Messages API + tool calling)",
    version="3.4.0",
)

# CORS: orígenes desde env, headers explícitos (no wildcard) y solo POST/GET.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=False,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def require_api_key(api_key: str = Security(api_key_header)) -> str:
    if not API_SECRET:
        raise HTTPException(503, "API_SECRET not configured on server")
    if api_key != API_SECRET:
        raise HTTPException(403, "Invalid API key")
    return api_key


# ────────────────────────────────────────────────────────────────────────────
# System prompt — covers every standard prop-firm rule across program / phase
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NYS Markets Senior Risk & Compliance Analyst. You audit a trader's MetaTrader account for payout eligibility against the NYS Markets program rules. Your output must hold up in an institutional compliance review.

# Program rulebook (the AUTHORITATIVE source of truth)

The NYS Markets rule documents define five distinct programs:

## 1-Step Evaluation (ONE_STEP)
- Profit target: 10% of initial balance.
- Min trading days: 3.
- Max Daily Loss: 3% (equity-based, day-start = max(balance, equity)).
- Max Loss: 6% fixed from initial balance (equity-based).
- Max leverage: 1:30.
- Daily reset: 21:00 UTC (server rollover / NY close).
- Allowed: scalping, EAs, news trading, overnight, weekend.
- Prohibited: arbitrage, latency arbitrage, HFT order flooding, server spamming, strategies exploiting platform delays.
- NO consistency rule. NO max risk per trade idea. NO max exposure per instrument. NO news restriction.

## 2-Step Phase 1 (TWO_STEP × PHASE_1)
- Profit target: 8%.
- Min trading days: 3.
- Max Daily Loss: 5% (equity-based).
- Max Loss: 10% fixed (equity-based).
- Max leverage: 1:100.
- Same risk limits stay active in Phase 2.
- Same allowed/prohibited list as 1-Step.

## 2-Step Phase 2 (TWO_STEP × PHASE_2)
- Profit target: 5%.
- Min trading days: 3.
- Max Daily Loss: 5%, Max Loss: 10% (carried over from Phase 1).

## Instant (INSTANT) — live from day one
- NO profit target (the account is already live).
- Max Daily Loss: 3% (equity-based, day-start = max(balance, equity)).
- Max Trailing Loss: 5% (equity-trailing, NEVER moves down, no reset). Initial limit = initial_balance − 5%. As equity makes new highs, the limit floats up by the same amount.
- Max Open Risk: 1% (total simultaneous floating loss across all open positions must never exceed 1% of balance).
- Consistency Rule: 15% — the most profitable trading day cannot exceed 15% of total profit generated. INSTANT ONLY.
- Max Risk per Trade Idea: 2% of balance. A trade idea = same instrument + same direction + within 5 minutes. Splitting positions to bypass this is prohibited.
- Max Exposure per Instrument: 4% of balance.
- Min Profitable Days: 7 within each 30-day period before a payout can be requested. A "profitable day" = net profit ≥ 0.25% of initial balance.
- News restriction: 5 minutes before / 5 minutes after high-impact news. Can NOT open new positions inside the window (existing positions may stay open).
- Max leverage: 1:50.
- Payout: 30 days after first trade, then every 14 days. Profit split 80% (90% after 3 successful payouts, up to 100% after 3 consecutive profitable months). Min payout: 3% of initial.
- Prohibited: arbitrage, latency arbitrage, HFT, tick scalping, server spamming, coordinated hedging across accounts, reverse trading, extreme over-leveraging, abrupt position-size escalation inconsistent with prior behavior.

## LiveFunded — 1-Step origin (ONE_STEP × LIVE)
LiveFunded is NOT a standalone evaluation. Risk limits are INHERITED from the originating ONE_STEP program. The Live Stage document adds (or overrides) only these rules:
- Max Daily Loss: 3% (inherited from 1-Step — preserved).
- Max Loss: 6% fixed (inherited from 1-Step — preserved).
- NO profit target (live trading).
- NO min trading days.
- NO consistency rule.
- ADD: Max Risk per Trade Idea = 3% (NEW in Live Stage).
- ADD: Max Exposure per Instrument = 4% (NEW).
- ADD: News restriction 5 min before / 5 min after high-impact events (NEW).
- ADD: Multi-account strategy restrictions (opposite positions across accounts, mirrored hedging, guaranteed-profit setups).
- Payout: first withdrawal 14 days after activation, then weekly Tuesdays. 80% base profit split, scaling up.

## LiveFunded — 2-Step origin (TWO_STEP × LIVE)
- Max Daily Loss: 5% (inherited from 2-Step — preserved).
- Max Loss: 10% fixed (inherited from 2-Step — preserved).
- Rest identical to LiveFunded 1-Step origin: 3% risk per trade idea, 4% exposure, news restriction, multi-account restrictions, weekly Tuesday payouts.

# Rule inheritance — non-negotiable

If account.phase == "LIVE", you MUST:
1. Identify the originating program (account.accountType: ONE_STEP or TWO_STEP).
2. Apply the originating program's daily-loss / max-loss limits as-is — DO NOT replace them with averaged or generic Live values.
3. Layer the Live Stage additions on top (max risk per trade idea 3%, max exposure 4%, news restriction).
4. Never inherit phase-only rules (profit target, min trading days) into Live.
For INSTANT accounts the inheritance step does not apply — INSTANT is its own program with its own rulebook.

# Authoritative thresholds

The preset object delivered in the payload carries the EXACT thresholds the firm has configured for this account. Treat preset values as the ground truth — they already reflect the rulebook resolution above. If a preset field is null/undefined, the rule does NOT apply to this program. Never invent a default.

The goal is NOT to simply say "pass" or "fail". The goal is to produce a complete risk report explaining:
- Which rules were respected.
- Which rules were breached.
- Which rules require manual review.
- Which specific trades/tickets caused concern.
- How each calculation was made.
- Why a specific action should be applied: approved, warning, remove profit, partial payout reduction, breach, account closure, or disqualification.

You receive:
- Account metadata (program / phase / size / requested payout / timezone / dailyResetTime).
- Selected rule preset (the firm's per-program thresholds — the source of truth).
- A sample of parsed trades (may be capped — see "trades_total_count" and "trades_sample" in the payload).
- Deterministic metrics computed by the frontend on the FULL trade set (authoritative ground truth).
- SL/TP data per trade.
- Balance and equity if available.

# CRITICAL ANALYSIS PRINCIPLES

Be strict, precise, and evidence-based. Do not hallucinate. Do not assume facts that are not in the data or rule configuration. If data is missing, clearly mark it as missing.

Every finding must include:
- Rule name and allowed threshold from the preset.
- Observed value (with the same units as the threshold).
- Calculation method.
- Evidence (which tickets / which observation).
- Severity.
- Recommended enforcement action.
- Confidence level (CONFIRMED / ESTIMATED / NOT_ENOUGH_DATA).

# Hard rules of conduct

1. Do not invent data. Never fabricate prices, tickets, timestamps, equity, SL, TP or calculations.
2. Treat the deterministic metrics as ground truth. Explain, contextualize and present them — do not recompute them.
3. If a rule threshold is missing from the preset (e.g. preset.profitTargetPercent is null/undefined), set the rule status to NOT_ENOUGH_DATA and explain what is missing. NEVER invent or substitute an "industry standard" default.
4. Use exact tickets, symbols, timestamps, volumes, entry prices, close prices, SL, TP and P&L from the provided data — never approximate or invent these values.
5. Clearly separate confirmed breaches (from uploaded data) from estimated breaches (from heuristics like SL distance × volume).
6. If a rule cannot be evaluated from the data, set status NOT_ENOUGH_DATA or MANUAL_REVIEW.
7. Generate a professional email the admin can copy and send manually. Do not send emails automatically.
8. Use firm, professional, compliance-oriented language. No accusations without confirmed data.

# CALCULATION STANDARDS

Use precise formulas and explain them in the rule's `calculation` field.

## Daily Loss Limit
Source of truth: metrics.dailyLossAnalysis (per-day series computed by the engine).
- threshold_amount = day_start_reference × preset.dailyLossPercent / 100
  where day_start_reference = max(balance_at_day_start, equity_at_day_start)
  When equity is unavailable, use balance — flag confidence accordingly.
- breach when: lowest_intraday_equity ≤ day_start_reference − threshold_amount
  When lowest intraday equity is unavailable, fall back to closed-only realized P&L for that day and mark MANUAL_REVIEW if the result is within 10% of the limit.
- relatedTickets: every ticket that contributed to the breaching day.

## Maximum Loss (Overall Drawdown)
Source: metrics.maxLossAnalysis and metrics.maxClosedLossPercent.
- threshold_amount = initial_balance × preset.maxLossPercent / 100
- breach_threshold_balance = initial_balance − threshold_amount
- breach when: minimum observed (equity if available, else balance) ≤ breach_threshold_balance.
- Distinguish closed-only drawdown from peak-to-trough equity drawdown when both are available.

## Trailing Loss Limit (INSTANT only — preset.maxTrailingLossPercent)
- initial_limit = initial_balance × (1 − preset.maxTrailingLossPercent / 100)
- trailing_limit moves UP every time equity reaches a new high: new_trailing_limit = highest_equity_observed × (1 − preset.maxTrailingLossPercent / 100), but NEVER decreases.
- breach when: equity ≤ current trailing_limit.
- Example: initial $100,000 with 5% trailing → initial limit $95,000. After equity high of $104,000, limit ratchets to $99,000. If equity later dips to $99,000, BREACH.
- Without a tick-by-tick equity curve, approximate from the highest closed-trade balance high-watermark and flag dataQuality=ESTIMATED.
- For non-INSTANT programs: include the rule with status="PASSED" and explanation="Rule applies to Instant program only.".

## Min Profitable Days (INSTANT only — preset.minProfitableDaysIn30)
- A "profitable day" = a calendar day on which net realized P&L ≥ preset.profitableDayMinPercent% of initial balance (default 0.25%).
- Trader must achieve ≥ preset.minProfitableDaysIn30 profitable days within the last rolling 30-day period BEFORE a payout can be requested.
- For accounts requesting a payout: status="BREACH" (or "MANUAL_REVIEW" depending on firm policy) if the count is below threshold. Include the actual count of profitable days observed.
- For non-INSTANT programs: PASSED / not applicable.

## Max Risk per Trade Idea
Source: metrics.tradeIdeaRiskAnalysis. A trade idea = trades sharing the same symbol + direction + a tight time window (default 60–300 s; respect what the engine grouped).
- risk_per_lot (BUY)  = open_price − stop_loss
- risk_per_lot (SELL) = stop_loss − open_price
- monetary_risk = risk_per_lot × lot_size × contract_size × conversion_rate
- idea_total_risk = sum(monetary_risk for trades in the idea)
- idea_risk_percent = idea_total_risk / account_balance_reference × 100
- BREACH if idea_risk_percent > preset.maxRiskPerTradeIdeaPercent.
- WARNING if 85% ≤ idea_risk_percent ≤ 100% of the threshold.
- If an idea has no SL on any leg, theoretical ex-ante risk is unbounded → set status MANUAL_REVIEW (or BREACH if the preset's prohibitedStrategies includes "no-SL high risk") with relatedTickets listing every leg. Do not claim a specific risk number.
- dataQuality = "ESTIMATED" (risk is derived from SL distance × volume, not actual fills).

## Max Exposure per Instrument
- exposure_percent = total_open_notional_or_risk_per_symbol / account_balance × 100
- For closed trades only, derive simultaneous exposure from overlapping open/close timestamps.
- BREACH if exposure_percent > preset.maxExposurePerSymbolPercent.

## Maximum Open Risk (INSTANT program ONLY — HARD BREACH)
- For account.accountType == "INSTANT", threshold = preset.maxOpenRiskPercent if present, else default 1.0 (1%).
- Total floating loss across all simultaneously open positions must never exceed the threshold. Source: metrics.openTradesExposure.
- BREACH if any moment's open-risk sum exceeds the threshold AND all relevant SLs are known.
- MANUAL_REVIEW if any moment had ≥2 simultaneously open positions and at least one had no stop loss.
- PASSED if always under threshold and all SLs are known throughout.
- For non-INSTANT accounts: include the rule with explanation="Rule applies to INSTANT program only.", status="PASSED".

## Profit Target — applies ONLY to evaluation phases (PHASE_1, PHASE_2)
- For account.phase == "LIVE": include one ruleByRuleAnalysis entry with status="PASSED", dataQuality="CONFIRMED", explanation="Not applicable on live funded accounts." Do NOT count this as a breach.
- For evaluation phases: target_amount = initial_balance × preset.profitTargetPercent / 100.
- status = "PASSED" if reached, "MANUAL_REVIEW" if not reached and trader is requesting payout, "NOT_ENOUGH_DATA" if metric is "NOT_ENOUGH_DATA".

## Minimum Trading Days
- A "trading day" is a calendar day in account.serverTimezone (reset at account.dailyResetTime) with at least one trade OPENED.
- Source: metrics.tradingDaysCount and metrics.minTradingDaysAnalysis.
- BREACH if tradingDaysCount < preset.minTradingDays.
- Flag micro-trades (very small lot size, < 1 minute holding, no SL) if the preset's policy treats them as abuse of the rule.

## Minimum Trades
- Count valid market trades only; exclude deposits, withdrawals, credits, corrections, internal adjustments.

## Consistency Rule — INSTANT program ONLY
- The consistency cap applies exclusively to the INSTANT program (account.accountType == "INSTANT"). The default cap is 15% of total profit on the best trading day.
- For non-INSTANT accounts (ONE_STEP, TWO_STEP at any phase, LIVE funded): include a single ruleByRuleAnalysis entry with status="PASSED" and explanation="Consistency rule does not apply to this program — Instant-only." Do NOT compute a consistency percentage and do NOT count it as a breach.
- For INSTANT accounts:
  - threshold = preset.consistencyRulePercent (default 15 when unset).
  - consistency_percent = bestTradingDayProfit / totalClosedPnL × 100, only when totalClosedPnL > 0.
  - If totalClosedPnL ≤ 0 → NOT_ENOUGH_DATA (cannot evaluate consistency on a losing or break-even period).
  - BREACH if consistency_percent > threshold; WARNING if within 90–100% of threshold.

## Minimum 2 Assets Rule (if configured)
- Count distinct symbols traded. If the rule requires N distinct instruments, BREACH if < N.

## Trades Without Stop Loss
- Source: metrics.tradesWithoutSL.
- WARNING if non-empty; promote to BREACH if any such trade lost > 2% of initial balance, or if preset.prohibitedStrategies includes "no-SL high risk" / "Trades without SL".
- relatedTickets MUST include the ticket numbers.

## Prohibited Strategy Patterns
Source: metrics.prohibitedPatterns + inspection of trades_sample. Default thresholds (unless preset overrides):
- Martingale: 3+ consecutive same-symbol/same-direction trades each ≥1.5× the previous volume AND each closed at a loss → WARNING; ≥5 in a row or doubled volume → BREACH.
- Grid / Position Stacking: 5+ trades on the same symbol with overlapping open windows and evenly-spaced entry prices → WARNING; identical entries pyramided → BREACH if account policy prohibits.
- Stacking (lighter): 3+ same-symbol/side trades opened within 60 s → WARNING.
- Tick scalping: median holding time < 60 s across ≥30 trades → WARNING; if the preset prohibits it and the median is < 15 s → BREACH.
- Lot-size spike: volume ≥ 3× rolling median of the trader's prior 30 trades → WARNING (BREACH if combined with no-SL).
- Hedging within account: opposing positions on the same symbol opened within 60 s → MANUAL_REVIEW.
- High-Frequency Trading: very high trades/day on the same symbol with sub-minute holds — WARNING / MANUAL_REVIEW unless explicitly prohibited.
Each detected pattern → one ruleByRuleAnalysis entry with relatedTickets listing every involved ticket.

## News Restrictions
- Evaluate ONLY if preset.newsRestrictionEnabled is true.
- If true and metrics.newsTradingFindings is present: enforce per-event windows (preset.newsMinutesBefore / preset.newsMinutesAfter) and flag any trade opened/closed inside a restricted window as BREACH.
- If true but news data is unavailable: NOT_ENOUGH_DATA with explanation="News-time data not available in this payload — requires server-side news calendar.".
- If preset.newsRestrictionEnabled is false: status="PASSED" with explanation="News restrictions are not enabled for this preset.".

## Policies that CANNOT be proven from the HTML alone
The following policies typically require data the MT statement does not carry:
- Platform exploit / latency arbitrage / spread manipulation.
- Copy trading detection (requires multi-account correlation).
- Account sharing (requires IP / device fingerprints).
- Multi-account hedging (requires cross-account correlation).
- EA usage (requires server-side logs).
For each of these, output a rule entry with status="MANUAL_REVIEW" and explanation that lists the data needed for confirmation (IP logs, device fingerprints, execution latency, cross-account correlation, trade-copier matching, server logs, tick data, liquidity data). Never call these a definitive breach without external evidence.

## Payout Eligibility
A payout is eligible when ALL of the following are true:
  1. Profit Target rule is PASSED or N/A (LIVE phase).
  2. Minimum Trading Days rule is PASSED.
  3. No CONFIRMED breaches across any of the above rules.
  4. Consistency Rule is PASSED or N/A (only INSTANT enforces consistency).
  5. requestedPayout (if provided) ≤ closed P&L available.
Compute payoutEligibility as one ruleByRuleAnalysis entry. Do NOT block a payout on consistency for non-INSTANT accounts.

# internalRecommendation enum mapping

Map the institutional decision to one of the existing tool-call enum values:
- "APPROVE PAYOUT" — all 5 payout-eligibility conditions PASSED, no BREACH-severity items.
- "PARTIAL APPROVAL" — most conditions PASSED but consistency or daily-loss has WARNING (firm policy allows partial), OR specific offending trades should have their profit removed (payout-reduction-recommended).
- "REJECT PAYOUT" — confirmed BREACH on a material rule, account-closure or disqualification recommended.
- "MANUAL REVIEW REQUIRED" — material rules are NOT_ENOUGH_DATA or MANUAL_REVIEW; suspicious prohibited-pattern findings without conclusive evidence.

# finalDecision and severity

- finalDecision = "BREACH" if any confirmedBreaches has status="BREACH".
- finalDecision = "WARNING" if there are warnings/estimatedBreaches but no confirmed BREACH.
- finalDecision = "MANUAL_REVIEW" if material rules are NOT_ENOUGH_DATA or MANUAL_REVIEW and the decision cannot be made.
- finalDecision = "PASSED" only when all applicable rules pass with confirmed data and there are no warnings of concern.
- severity reflects the worst rule: LOW (no findings), MEDIUM (warnings only), HIGH (one confirmed breach), CRITICAL (multiple confirmed breaches or prohibited-pattern BREACH).

# evidenceTable

Up to 30 rows. Each row is one observation supporting a finding. observation and ruleRef are mandatory; ticket/symbol/time are optional (e.g. "Minimum Trading Days" has no specific ticket — use observation="Trader logged 4 distinct trading days, below the 5-day minimum.").

# Extended optional fields (populate when relevant — UI will use them when present)

- `tradeIdeaGroups`: array of { ideaId, symbol, direction, startTime, endTime, tickets[], totalLots, estimatedTotalRiskAmount, estimatedTotalRiskPercentage, allowedRiskPercentage, status, explanation }.
- `dailyAnalysis`: array of { date, dayStartReference, dailyPnL, lowestObserved, dailyLossLimitAmount, dailyLossLimitPercent, status, affectedTickets[], explanation }.
- `prohibitedPolicyReview`: array of { policyName, status (no_evidence|suspicious|breached|manual_review_required|not_applicable), severity, evidence, affectedTickets[], explanation, dataNeededForConfirmation[] }.
- `accountSummaryExtras`: { netProfit, grossProfit, grossLoss, winningTrades, losingTrades, winRatePercent, profitFactor, largestWin, largestLoss, mostTradedSymbol, highestRiskSymbol }.
- `resolvedRulebook`: ALWAYS populate when account.phase == "LIVE". Confirms the applicable rulebook AFTER inheritance:
  { programLabel, originatingProgram (ONE_STEP|TWO_STEP|INSTANT|NONE), isLiveFunded, dailyLossPercent, maxLossPercent, maxTrailingLossPercent, maxOpenRiskPercent, profitTargetPercent, minTradingDays, minProfitableDaysIn30, profitableDayMinPercent, consistencyRulePercent, maxRiskPerTradeIdeaPercent, maxExposurePerSymbolPercent, newsRestrictionEnabled, newsMinutesBefore, newsMinutesAfter, tradeIdeaWindowMinutes, maxLeverage, inheritanceExplanation, rulesAdded: [], rulesInherited: [] }.
  Use `inheritanceExplanation` to explain in one paragraph WHICH rules came from the originating evaluation program (e.g. "Max Daily Loss 3% inherited from 1-Step") and which were ADDED by the Live Stage document (e.g. "Max Risk per Trade Idea 3%, Max Exposure 4%, News restriction 5/5 min — added by Live Stage").
- `humanReportMarkdown`: a complete professional markdown report with the sections: Executive Summary, Account Overview, Final Decision, Rule-by-Rule Review (with limits, observed, status, calculation, affected tickets), Critical Findings, Trade-Level Evidence, Max Risk per Trade Idea Analysis, Prohibited Trading Policy Review, Data Limitations, Recommended Action.

# Email output (emailSubject + emailBody) — must be ready for the admin to copy and send WITHOUT EDITS to the trader.

The email is the trader-facing artifact. It must be DETAILED, EVIDENCE-BASED, and explain — for every flagged item — exactly which trades, at what time, in what way, and why something happened. This is not a "you got rejected" letter; it is the firm's professional record of the review.

Required structure for emailBody (in this exact order):

1. Greeting: "Dear {{traderName}},"
2. Opening paragraph: state that the Risk & Compliance team has completed the review of the payout request for account {{accountNumber}} on the {{program}} {{phase}} program. Mention the review period (first trade open time → last trade close time) and the number of closed trades evaluated.
3. Result paragraph: state the outcome (passed / warning / breach / manual review required) and the recommended action plainly. One short paragraph.
4. Account overview block (5–7 lines, "label: value" format, no table syntax):
   - Initial balance
   - Final balance (or last observed)
   - Net realized P&L (and % of initial)
   - Trading days
   - Symbols traded
   - Best day P&L (date, amount)
   - Worst day P&L (date, amount)
5. "Findings" section — one block per rule that is NOT PASSED. For each block include:
   - Rule name (e.g. "Maximum Loss Limit")
   - Configured limit (e.g. "10.00% / $1,000.00")
   - Observed value (e.g. "11.42% / $1,142.00 closed-trade drawdown peaking on 2026-05-14")
   - Calculation method (one sentence — e.g. "max_loss_limit = initial_balance × maxLossPercent / 100")
   - Why it was breached (one paragraph in plain English referencing the trader's actual trades — date, time, size, sequence — not abstract reasoning)
   - Evidence (5–10 most relevant tickets max, ONE PER LINE in this format):
       Ticket {ticket} — {symbol} {direction} {volume} lots | opened {YYYY-MM-DD HH:MM:SS} @ {entryPrice} (SL {stopLoss}, TP {takeProfit}) | closed {YYYY-MM-DD HH:MM:SS} @ {closePrice} | commission {commission} | swap {swap} | P&L {profit}
     Use "N/A" for missing fields. NEVER fabricate values.
6. Optional "Trade idea risk" section if any trade-idea breach exists: for each breaching idea list the symbol, direction, time window, total lots, estimated risk amount + percent, allowed percent, and the cluster of tickets involved.
7. "Data limitations" — one short paragraph listing what could NOT be confirmed from the HTML alone (intra-trade equity, server-side news data, IP / device, EA usage, etc.) if any rule was marked MANUAL_REVIEW or ESTIMATED.
8. "Next step" paragraph — explicit and unambiguous:
   - For APPROVED: "Your payout request of {{requestedPayout}} has been approved and will be processed within {standard schedule}."
   - For PARTIAL APPROVAL / payout reduction: state the original amount, the deduction (with the offending trade(s)), and the approved amount.
   - For REJECT PAYOUT / BREACH: state that the payout has been declined and the consequence on the account (closure / disqualification, per firm policy).
   - For MANUAL REVIEW: state that the team will follow up within {1–2} business days with the next steps; no action required from the trader.
9. Closing: "If you have any questions about this review, you may reply to this email. Best regards, Risk & Compliance Team".

Format rules:
- Plain text only. NO markdown headings (`#`), NO markdown tables, NO `|` characters used as table separators, NO triple backticks, NO `**bold**` markers. Use uppercase short labels for section titles (e.g. "FINDINGS", "EVIDENCE", "NEXT STEP") on their own line if section separation helps readability.
- Numeric formatting: dollars with comma separators and 2 decimals ("$1,142.00"), percentages with 2 decimals ("11.42%"), times in YYYY-MM-DD HH:MM:SS (account.serverTimezone if available).
- Stay under ~800 words but never sacrifice required evidence detail.

Email tone:
- Professional, firm, respectful, compliance-oriented.
- No accusations unless proven by uploaded data. For estimated or pattern findings, use language such as "The review identified a pattern consistent with..." or "This requires manual review...".
- No aggressive, emotional, or apologetic wording.
- Prefer phrasings like: "The review identified...", "Based on the available account data...", "The account exceeded...", "This requires manual review...".

Hard prohibitions in emailBody (case-insensitive — NEVER include any of these substrings):
"claude", "anthropic", "openai", "gpt", "llm", "ai-generated", "ai generated", "language model", "automated by", "this is an automated", "generated by ai". Do not reference any AI or automation tooling, internal reasoning, confidence scores, or provider names. Do not promise outcomes outside the stated next step.

emailSubject must be concise and reference the account number and outcome, e.g. "Payout Review — Account 12345 — Breach Confirmed" or "Payout Review — Account 12345 — Approved" or "Payout Review — Account 12345 — Manual Review Required".

# Output contract

You respond ONLY by calling the submit_audit_result tool with valid structured output. No prose outside the tool call. Use the exact UPPER_SNAKE_CASE enum values where the schema requires them (finalDecision, severity, per-rule status / dataQuality). Populate the extended optional fields (humanReportMarkdown, tradeIdeaGroups, dailyAnalysis, prohibitedPolicyReview, accountSummaryExtras) whenever the data supports it.
"""


RULE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "rule": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["PASSED", "WARNING", "BREACH", "MANUAL_REVIEW", "NOT_ENOUGH_DATA"],
        },
        "severity": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        },
        "dataQuality": {
            "type": "string",
            "enum": ["CONFIRMED", "ESTIMATED", "NOT_ENOUGH_DATA"],
        },
        "explanation": {"type": "string"},
        "calculation": {"type": "string"},
        "relatedTickets": {"type": "array", "items": {"type": "string"}},
        "allowed": {
            "type": "string",
            "description": "Allowed threshold for this rule (e.g. '5% / $5,000').",
        },
        "actual": {
            "type": "string",
            "description": "Actual measured result for this rule (e.g. '6.2% / $6,200').",
        },
    },
    "required": ["rule", "status", "severity", "dataQuality", "explanation"],
}

SUBMIT_AUDIT_TOOL = {
    "name": "submit_audit_result",
    "description": "Submit the structured payout audit result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "finalDecision": {
                "type": "string",
                "enum": ["PASSED", "WARNING", "BREACH", "MANUAL_REVIEW"],
            },
            "severity": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            },
            "executiveSummary": {"type": "string"},
            "confirmedBreaches": {
                "type": "array",
                "items": RULE_ANALYSIS_SCHEMA,
            },
            "estimatedBreaches": {
                "type": "array",
                "items": RULE_ANALYSIS_SCHEMA,
            },
            "warnings": {"type": "array", "items": RULE_ANALYSIS_SCHEMA},
            "notEnoughData": {"type": "array", "items": RULE_ANALYSIS_SCHEMA},
            "ruleByRuleAnalysis": {
                "type": "array",
                "items": RULE_ANALYSIS_SCHEMA,
            },
            "evidenceTable": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticket": {"type": "string"},
                        "symbol": {"type": "string"},
                        "time": {"type": "string"},
                        "observation": {"type": "string"},
                        "ruleRef": {"type": "string"},
                    },
                    # Only observation + ruleRef are mandatory — some rules
                    # (e.g. Minimum Trading Days) have no specific ticket.
                    "required": ["observation", "ruleRef"],
                },
            },
            "internalRecommendation": {
                "type": "string",
                "enum": [
                    "APPROVE PAYOUT",
                    "REJECT PAYOUT",
                    "PARTIAL APPROVAL",
                    "MANUAL REVIEW REQUIRED",
                ],
            },
            "emailSubject": {"type": "string"},
            "emailBody": {"type": "string"},

            # Extended optional fields (populated when the data supports them).
            # The UI consumes these progressively — absence is fine.
            "humanReportMarkdown": {
                "type": "string",
                "description": "Full professional markdown report (Executive Summary, Account Overview, Rule-by-Rule Review, Critical Findings, Trade-Level Evidence, Max Risk per Trade Idea Analysis, Prohibited Trading Policy Review, Data Limitations, Recommended Action).",
            },
            "tradeIdeaGroups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ideaId": {"type": "string"},
                        "symbol": {"type": "string"},
                        "direction": {"type": "string", "enum": ["BUY", "SELL", "MIXED"]},
                        "startTime": {"type": "string"},
                        "endTime": {"type": "string"},
                        "tickets": {"type": "array", "items": {"type": "string"}},
                        "totalLots": {"type": "number"},
                        "estimatedTotalRiskAmount": {"type": "number"},
                        "estimatedTotalRiskPercentage": {"type": "number"},
                        "allowedRiskPercentage": {"type": "number"},
                        "status": {
                            "type": "string",
                            "enum": ["PASSED", "WARNING", "BREACH", "MANUAL_REVIEW"],
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["ideaId", "symbol", "tickets", "status", "explanation"],
                },
            },
            "dailyAnalysis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "dayStartReference": {"type": "number"},
                        "dailyPnL": {"type": "number"},
                        "lowestObserved": {"type": "number"},
                        "dailyLossLimitAmount": {"type": "number"},
                        "dailyLossLimitPercent": {"type": "number"},
                        "status": {
                            "type": "string",
                            "enum": ["PASSED", "WARNING", "BREACH", "MANUAL_REVIEW", "NOT_ENOUGH_DATA"],
                        },
                        "affectedTickets": {"type": "array", "items": {"type": "string"}},
                        "explanation": {"type": "string"},
                    },
                    "required": ["date", "status", "explanation"],
                },
            },
            "prohibitedPolicyReview": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "policyName": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": [
                                "NO_EVIDENCE",
                                "SUSPICIOUS",
                                "BREACHED",
                                "MANUAL_REVIEW_REQUIRED",
                                "NOT_APPLICABLE",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
                        },
                        "evidence": {"type": "string"},
                        "affectedTickets": {"type": "array", "items": {"type": "string"}},
                        "explanation": {"type": "string"},
                        "dataNeededForConfirmation": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["policyName", "status", "explanation"],
                },
            },
            "accountSummaryExtras": {
                "type": "object",
                "properties": {
                    "netProfit": {"type": "number"},
                    "grossProfit": {"type": "number"},
                    "grossLoss": {"type": "number"},
                    "winningTrades": {"type": "integer"},
                    "losingTrades": {"type": "integer"},
                    "winRatePercent": {"type": "number"},
                    "profitFactor": {"type": "number"},
                    "largestWin": {"type": "number"},
                    "largestLoss": {"type": "number"},
                    "mostTradedSymbol": {"type": "string"},
                    "highestRiskSymbol": {"type": "string"},
                },
            },
            "resolvedRulebook": {
                "type": "object",
                "description": "The applicable rulebook after resolving LiveFunded inheritance. Confirms which limits were enforced and where they came from.",
                "properties": {
                    "programLabel": {"type": "string"},
                    "originatingProgram": {
                        "type": "string",
                        "enum": ["ONE_STEP", "TWO_STEP", "INSTANT", "NONE"],
                    },
                    "isLiveFunded": {"type": "boolean"},
                    "dailyLossPercent": {"type": "number"},
                    "maxLossPercent": {"type": "number"},
                    "maxTrailingLossPercent": {"type": "number"},
                    "maxOpenRiskPercent": {"type": "number"},
                    "profitTargetPercent": {"type": "number"},
                    "minTradingDays": {"type": "integer"},
                    "minProfitableDaysIn30": {"type": "integer"},
                    "profitableDayMinPercent": {"type": "number"},
                    "consistencyRulePercent": {"type": "number"},
                    "maxRiskPerTradeIdeaPercent": {"type": "number"},
                    "maxExposurePerSymbolPercent": {"type": "number"},
                    "newsRestrictionEnabled": {"type": "boolean"},
                    "newsMinutesBefore": {"type": "integer"},
                    "newsMinutesAfter": {"type": "integer"},
                    "tradeIdeaWindowMinutes": {"type": "integer"},
                    "maxLeverage": {"type": "integer"},
                    "inheritanceExplanation": {
                        "type": "string",
                        "description": "One paragraph explaining which rules were inherited from the originating program and which were added by the Live Stage document.",
                    },
                    "rulesAdded": {"type": "array", "items": {"type": "string"}},
                    "rulesInherited": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "required": [
            "finalDecision",
            "severity",
            "executiveSummary",
            "confirmedBreaches",
            "estimatedBreaches",
            "warnings",
            "notEnoughData",
            "ruleByRuleAnalysis",
            "evidenceTable",
            "internalRecommendation",
            "emailSubject",
            "emailBody",
        ],
    },
}


# Pre-computed enum sets for output validation
_FINAL_DECISIONS = {"PASSED", "WARNING", "BREACH", "MANUAL_REVIEW"}
_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_RULE_STATUSES = {"PASSED", "WARNING", "BREACH", "MANUAL_REVIEW", "NOT_ENOUGH_DATA"}
_DATA_QUALITIES = {"CONFIRMED", "ESTIMATED", "NOT_ENOUGH_DATA"}
_RECOMMENDATIONS = {
    "APPROVE PAYOUT",
    "REJECT PAYOUT",
    "PARTIAL APPROVAL",
    "MANUAL REVIEW REQUIRED",
}


# ────────────────────────────────────────────────────────────────────────────
# Request / response models
# ────────────────────────────────────────────────────────────────────────────


class AuditRequest(BaseModel):
    """Frontend envía este shape. Aceptamos cualquier campo extra para forward-compat."""

    account: dict[str, Any] = Field(..., description="AccountMetadata")
    preset: dict[str, Any] = Field(..., description="RulePreset")
    trades: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# ────────────────────────────────────────────────────────────────────────────
# Smart trade sampling — preserve flagged trades, fill with uniform sample
# ────────────────────────────────────────────────────────────────────────────


def _truncate_trades(
    trades: list[dict], metrics: dict[str, Any]
) -> tuple[list[dict], int]:
    """Select up to MAX_TRADES_IN_PROMPT trades, preserving ones the engine flagged.

    Strategy:
      1. Build a "must keep" set from metrics: tradesWithoutSL, lotSizeSpikes,
         largestLosingTradeTicket, shortDurationTrades, and any ticket inside
         tradeIdeaRiskAnalysis / stackedTrades / prohibitedPatterns.
      2. Always keep the first and last trade (boundaries).
      3. Fill the remainder with a uniform-by-index sample from the unflagged
         population, to preserve temporal distribution.

    Returns (sampled_list, dropped_count).
    """
    n = len(trades)
    if n <= MAX_TRADES_IN_PROMPT:
        return trades, 0

    must_keep: set[str] = set()

    def _add(values: Any) -> None:
        if values is None:
            return
        if isinstance(values, str):
            must_keep.add(values)
            return
        if isinstance(values, (list, tuple, set)):
            for v in values:
                if isinstance(v, str):
                    must_keep.add(v)
                elif isinstance(v, dict):
                    t = v.get("ticket") or v.get("orderId")
                    if isinstance(t, str):
                        must_keep.add(t)
                    tickets = v.get("tickets") or v.get("relatedTickets")
                    if isinstance(tickets, list):
                        for tt in tickets:
                            if isinstance(tt, str):
                                must_keep.add(tt)

    _add(metrics.get("tradesWithoutSL"))
    _add(metrics.get("shortDurationTrades"))
    _add(metrics.get("lotSizeSpikes"))
    _add(metrics.get("stackedTrades"))
    _add(metrics.get("tradeIdeaRiskAnalysis"))
    _add(metrics.get("prohibitedPatterns"))
    _add(metrics.get("largestLosingTradeTicket"))

    kept_indices: set[int] = set()
    for i, t in enumerate(trades):
        ticket = str(t.get("ticket") or t.get("orderId") or "")
        if ticket and ticket in must_keep:
            kept_indices.add(i)

    # Always keep boundaries.
    kept_indices.add(0)
    kept_indices.add(n - 1)

    remaining = MAX_TRADES_IN_PROMPT - len(kept_indices)
    if remaining > 0:
        unflagged_indices = [i for i in range(n) if i not in kept_indices]
        if unflagged_indices:
            step = max(1, len(unflagged_indices) / remaining)
            for j in range(remaining):
                idx = int(j * step)
                if idx < len(unflagged_indices):
                    kept_indices.add(unflagged_indices[idx])

    sorted_idx = sorted(kept_indices)
    sampled = [trades[i] for i in sorted_idx]
    return sampled, n - len(sampled)


# ────────────────────────────────────────────────────────────────────────────
# Anthropic call with timeout + retries
# ────────────────────────────────────────────────────────────────────────────


def _build_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=httpx.Timeout(ANTHROPIC_TIMEOUT_SECONDS, connect=10.0),
        max_retries=0,  # we implement retry ourselves to log + backoff cleanly
    )


def _call_anthropic_with_retry(client: anthropic.Anthropic, **kwargs: Any):
    """Call client.messages.create with exponential backoff on 429/5xx/overload.

    Raises the last error if all attempts fail. 4xx other than 429 are not
    retried.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(ANTHROPIC_MAX_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_err = e
            wait = (2**attempt) + random.random()
            log.warning(
                "Anthropic 429 rate limit (attempt %d/%d), retrying in %.1fs",
                attempt + 1,
                ANTHROPIC_MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            last_err = e
            if e.status_code in (500, 502, 503, 504, 529):
                wait = (2**attempt) + random.random()
                log.warning(
                    "Anthropic %d (attempt %d/%d), retrying in %.1fs",
                    e.status_code,
                    attempt + 1,
                    ANTHROPIC_MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            raise
        except anthropic.APIConnectionError as e:
            last_err = e
            wait = (2**attempt) + random.random()
            log.warning(
                "Anthropic connection error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                ANTHROPIC_MAX_RETRIES,
                wait,
                type(e).__name__,
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


# ────────────────────────────────────────────────────────────────────────────
# Output validation
# ────────────────────────────────────────────────────────────────────────────


def _validate_rule_item(item: Any, path: str) -> Optional[str]:
    if not isinstance(item, dict):
        return f"{path}: not an object"
    for required in ("rule", "status", "severity", "dataQuality", "explanation"):
        if required not in item:
            return f"{path}: missing field '{required}'"
    if item["status"] not in _RULE_STATUSES:
        return f"{path}.status: invalid value '{item['status']}'"
    if item["severity"] not in _SEVERITIES:
        return f"{path}.severity: invalid value '{item['severity']}'"
    if item["dataQuality"] not in _DATA_QUALITIES:
        return f"{path}.dataQuality: invalid value '{item['dataQuality']}'"
    return None


def validate_audit_output(data: dict) -> Optional[str]:
    """Return None if valid, otherwise a short error string."""
    if not isinstance(data, dict):
        return "audit result is not an object"

    required_top = [
        "finalDecision",
        "severity",
        "executiveSummary",
        "confirmedBreaches",
        "estimatedBreaches",
        "warnings",
        "notEnoughData",
        "ruleByRuleAnalysis",
        "evidenceTable",
        "internalRecommendation",
        "emailSubject",
        "emailBody",
    ]
    for f in required_top:
        if f not in data:
            return f"missing top-level field '{f}'"

    if data["finalDecision"] not in _FINAL_DECISIONS:
        return f"finalDecision: invalid value '{data['finalDecision']}'"
    if data["severity"] not in _SEVERITIES:
        return f"severity: invalid value '{data['severity']}'"
    if data["internalRecommendation"] not in _RECOMMENDATIONS:
        return (
            "internalRecommendation: invalid value "
            f"'{data['internalRecommendation']}'"
        )

    for arr in (
        "confirmedBreaches",
        "estimatedBreaches",
        "warnings",
        "notEnoughData",
        "ruleByRuleAnalysis",
    ):
        items = data.get(arr)
        if not isinstance(items, list):
            return f"{arr}: not an array"
        for i, item in enumerate(items):
            err = _validate_rule_item(item, f"{arr}[{i}]")
            if err:
                return err

    if not isinstance(data["evidenceTable"], list):
        return "evidenceTable: not an array"
    for i, row in enumerate(data["evidenceTable"]):
        if not isinstance(row, dict):
            return f"evidenceTable[{i}]: not an object"
        if "observation" not in row or "ruleRef" not in row:
            return f"evidenceTable[{i}]: must include 'observation' and 'ruleRef'"

    # Anti-leak: never let provider/automation references reach the trader.
    body_lower = str(data.get("emailBody") or "").lower()
    for forbidden in EMAIL_FORBIDDEN_SUBSTRINGS:
        if forbidden in body_lower:
            return f"emailBody contains forbidden phrase '{forbidden.strip()}'"
    if "|---" in body_lower or "| ---" in body_lower:
        return "emailBody contains markdown table syntax"

    return None


# ────────────────────────────────────────────────────────────────────────────
# Audit handler
# ────────────────────────────────────────────────────────────────────────────


def run_audit(req: AuditRequest) -> dict:
    """Llama a Claude con el contexto, fuerza tool call, retorna structured output.

    Always real — never a mock or stub. If ANTHROPIC_API_KEY is missing or
    the call fails, this function raises an HTTPException; the frontend
    surfaces it as ai_status="failed" and a visible error banner. No fake
    audit is ever substituted.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured on server")

    client = _build_client()

    trades_for_prompt, dropped = _truncate_trades(req.trades, req.metrics)
    sampling_note = ""
    if dropped:
        sampling_note = (
            f"\n\nNOTE: trades_sample was reduced from {len(req.trades)} total "
            f"down to {len(trades_for_prompt)} entries while preserving every "
            "ticket flagged by metrics (no-SL, lot spikes, prohibited-pattern "
            "clusters, largest losing trade, short-duration trades). Counts "
            "and metrics are computed on the FULL set; use `metrics` as ground "
            "truth for totals."
        )

    user_payload = {
        "account": req.account,
        "preset": req.preset,
        "metrics": req.metrics,
        "trades_sample": trades_for_prompt,
        "trades_total_count": len(req.trades),
    }

    user_content = (
        "Analyze the following payout review request and submit the structured "
        f"result by calling submit_audit_result.{sampling_note}\n\n"
        f"```json\n{json.dumps(user_payload, indent=2, default=str)}\n```"
    )

    # Cache the static SYSTEM_PROMPT — cuts input cost ~80% on repeat calls.
    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    started = time.time()
    try:
        message = _call_anthropic_with_retry(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            tools=[SUBMIT_AUDIT_TOOL],
            tool_choice={"type": "tool", "name": "submit_audit_result"},
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.RateLimitError:
        log.exception("Anthropic rate limit exceeded after retries")
        raise HTTPException(503, "Risk Manager service is overloaded. Try again shortly.")
    except anthropic.APIStatusError as e:
        log.exception(
            "Anthropic API error after retries: status=%s",
            getattr(e, "status_code", "?"),
        )
        raise HTTPException(
            502, "Risk Manager upstream is currently unavailable. Try again shortly."
        )
    except anthropic.APIConnectionError:
        log.exception("Could not reach Anthropic API after retries")
        raise HTTPException(504, "Risk Manager upstream connection failed.")
    except Exception:
        log.exception("Unexpected error calling Anthropic")
        raise HTTPException(500, "Risk Manager internal error.")

    latency_ms = int((time.time() - started) * 1000)

    # Validate stop_reason — anything other than tool_use means the output is
    # likely truncated or malformed.
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        log.warning("Anthropic stop_reason=max_tokens — output likely truncated")
        raise HTTPException(
            502,
            "Risk Manager produced a truncated response. Increase ANTHROPIC_MAX_TOKENS "
            "or reduce trade volume.",
        )

    tool_block = next(
        (b for b in message.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        log.warning(
            "No tool_use block in response. stop_reason=%s", stop_reason
        )
        raise HTTPException(
            502,
            "Risk Manager did not return a structured audit. Please retry.",
        )

    audit = tool_block.input
    if not isinstance(audit, dict):
        raise HTTPException(502, "Risk Manager returned a non-object audit payload.")

    err = validate_audit_output(audit)
    if err:
        log.warning("Audit output validation failed: %s", err)
        raise HTTPException(
            502, f"Risk Manager produced an invalid audit payload: {err}"
        )

    usage = {
        "input_tokens": getattr(message.usage, "input_tokens", 0),
        "output_tokens": getattr(message.usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(
            message.usage, "cache_creation_input_tokens", 0
        ),
        "cache_read_input_tokens": getattr(
            message.usage, "cache_read_input_tokens", 0
        ),
    }
    cost_usd = _estimate_cost_usd(MODEL, usage)
    response_id = getattr(message, "id", None)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    log.info(
        "AI_CALL ok provider=anthropic model=%s response_id=%s latency_ms=%d "
        "input=%d output=%d cache_read=%d cache_write=%d cost_usd=%.6f",
        MODEL,
        response_id,
        latency_ms,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_input_tokens"],
        usage["cache_creation_input_tokens"],
        cost_usd,
    )

    return {
        "ok": True,
        "data": audit,
        "ai": {
            "ai_status": "ok",
            "provider": "anthropic",
            "model": MODEL,
            "response_id": response_id,
            "generated_at": generated_at,
            "latency_ms": latency_ms,
            "usage": usage,
            "cost_usd_estimate": cost_usd,
            "fallback_used": False,
            "mock": False,
        },
        # Legacy fields preserved for backward compatibility with older
        # frontend clients still reading res.data.{model,usage,latencyMs}.
        "latencyMs": latency_ms,
        "model": MODEL,
        "usage": usage,
        "trades_dropped": dropped,
    }




# ────────────────────────────────────────────────────────────────────────────
# External data helpers
# ────────────────────────────────────────────────────────────────────────────


async def get_economic_events(symbols: list[str]) -> dict:
    """Fetch economic calendar events from Apify for the last 7 and next 7 days.

    Filters for US events with high or medium importance. Returns a dict with
    an ``events`` list and ``source`` metadata. Errors are caught and logged so
    they never block the main audit response.
    """
    if not APIFY_API_KEY or not APIFY_ECONOMIC_CALENDAR_TOKEN:
        log.info("Apify credentials not configured — skipping economic events fetch")
        return {"events": [], "source": "apify", "error": "credentials_not_configured"}

    now = datetime.utcnow()
    date_from = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    date_to = (now + timedelta(days=7)).strftime("%Y-%m-%d")

    url = (
        f"https://api.apify.com/v2/acts/{APIFY_ECONOMIC_CALENDAR_TOKEN}/runs/last/dataset/items"
        f"?token={APIFY_API_KEY}"
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params={"dateFrom": date_from, "dateTo": date_to})
            resp.raise_for_status()
            raw_events: list[dict] = resp.json()

        filtered = [
            e for e in raw_events
            if (
                e.get("country", "").upper() == "US"
                and e.get("importance", "").upper() in ("HIGH", "MEDIUM")
            )
        ]

        log.info(
            "Economic events fetched: %d total, %d US high/medium after filter",
            len(raw_events),
            len(filtered),
        )
        return {
            "events": filtered,
            "source": "apify",
            "date_from": date_from,
            "date_to": date_to,
            "total_fetched": len(raw_events),
        }
    except Exception:
        log.exception("Failed to fetch economic events from Apify")
        return {"events": [], "source": "apify", "error": "fetch_failed"}


async def get_price_data(symbols: list[str]) -> dict:
    """Fetch real-time and historical price data from Massive API for each symbol.

    Processes symbols concurrently and returns a mapping of symbol -> price data.
    Per-symbol errors are caught individually so a single bad symbol never
    fails the entire request.
    """
    if not MASSIVE_API_KEY:
        log.info("MASSIVE_API_KEY not configured — skipping price data fetch")
        return {"error": "credentials_not_configured"}

    if not symbols:
        return {}

    results: dict[str, Any] = {}

    async with httpx.AsyncClient(
        timeout=10.0,
        headers={"Authorization": f"Bearer {MASSIVE_API_KEY}"},
    ) as client:
        async def _fetch_symbol(symbol: str) -> None:
            try:
                # Real-time quote
                rt_resp = await client.get(
                    "https://api.massiveapi.com/v1/quotes",
                    params={"symbol": symbol},
                )
                rt_resp.raise_for_status()
                rt_data = rt_resp.json()

                # Historical OHLCV (last 30 days, daily)
                hist_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
                hist_resp = await client.get(
                    "https://api.massiveapi.com/v1/historical",
                    params={"symbol": symbol, "from": hist_from, "interval": "1d"},
                )
                hist_resp.raise_for_status()
                hist_data = hist_resp.json()

                results[symbol] = {
                    "realtime": rt_data,
                    "historical": hist_data,
                }
                log.info("Price data fetched for symbol: %s", symbol)
            except Exception:
                log.exception("Failed to fetch price data for symbol: %s", symbol)
                results[symbol] = {"error": "fetch_failed"}

        await asyncio.gather(*[_fetch_symbol(s) for s in symbols])

    return results


def parse_mt_html(html_content: str) -> dict:
    """Parse an MT4/MT5 HTML statement and extract account info and trades.

    Uses BeautifulSoup to locate the account summary and the closed-trades
    table. Returns a dict with ``account_info`` and ``trades`` keys. Parsing
    errors are caught and surfaced in the returned dict rather than raised, so
    callers can decide how to handle partial data.
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception:
        log.exception("BeautifulSoup failed to parse HTML")
        return {"account_info": {}, "trades": [], "parse_error": "html_parse_failed"}

    account_info: dict[str, Any] = {}
    trades: list[dict[str, Any]] = []

    # ── Account summary ──────────────────────────────────────────────────────
    # MT4/MT5 statements embed account details in a table near the top of the
    # document. We look for cells whose text contains known labels.
    try:
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                for i, cell in enumerate(cells):
                    cell_lower = cell.lower()
                    if "account" in cell_lower and i + 1 < len(cells):
                        account_info.setdefault("accountNumber", cells[i + 1])
                    elif "balance" in cell_lower and i + 1 < len(cells):
                        account_info.setdefault("balance", cells[i + 1])
                    elif "equity" in cell_lower and i + 1 < len(cells):
                        account_info.setdefault("equity", cells[i + 1])
                    elif "name" in cell_lower and i + 1 < len(cells):
                        account_info.setdefault("name", cells[i + 1])
    except Exception:
        log.exception("Error extracting account info from MT HTML")

    # ── Trades table ─────────────────────────────────────────────────────────
    # MT4/MT5 statements list closed trades in a table whose header row
    # contains columns like Ticket, Open Time, Type, Size/Volume, Symbol,
    # Price (open), S/L, T/P, Close Time, Price (close), Commission, Swap,
    # Profit. Column order varies slightly between MT4 and MT5 exports.
    TRADE_COLUMN_ALIASES: dict[str, str] = {
        "ticket": "ticket",
        "order": "ticket",
        "open time": "openTime",
        "opentime": "openTime",
        "type": "type",
        "size": "volume",
        "volume": "volume",
        "lots": "volume",
        "item": "symbol",
        "symbol": "symbol",
        "price": "openPrice",
        "open price": "openPrice",
        "s/l": "stopLoss",
        "sl": "stopLoss",
        "stop loss": "stopLoss",
        "t/p": "takeProfit",
        "tp": "takeProfit",
        "take profit": "takeProfit",
        "close time": "closeTime",
        "closetime": "closeTime",
        "close price": "closePrice",
        "commission": "commission",
        "swap": "swap",
        "profit": "profit",
        "taxes": "taxes",
    }

    try:
        for table in soup.find_all("table"):
            headers_row = table.find("tr")
            if headers_row is None:
                continue

            raw_headers = [
                th.get_text(strip=True).lower()
                for th in headers_row.find_all(["th", "td"])
            ]

            # Identify this as a trades table by requiring at least "ticket"
            # (or "order") and "profit" columns.
            col_map: dict[int, str] = {}
            for idx, raw_h in enumerate(raw_headers):
                canonical = TRADE_COLUMN_ALIASES.get(raw_h)
                if canonical:
                    col_map[idx] = canonical

            canonical_names = set(col_map.values())
            if "ticket" not in canonical_names or "profit" not in canonical_names:
                continue

            # Parse each data row.
            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if not cells:
                    continue

                trade: dict[str, Any] = {}
                for idx, field_name in col_map.items():
                    if idx < len(cells):
                        trade[field_name] = cells[idx]

                # Skip summary / balance / deposit rows that lack a numeric ticket.
                ticket_val = trade.get("ticket", "")
                if not ticket_val or not any(ch.isdigit() for ch in str(ticket_val)):
                    continue

                # Skip non-trade rows (balance, credit, etc.) identified by type.
                trade_type = trade.get("type", "").lower()
                if trade_type in ("balance", "credit", "deposit", "withdrawal", ""):
                    continue

                trades.append(trade)

            # Stop after the first valid trades table.
            if trades:
                break
    except Exception:
        log.exception("Error extracting trades from MT HTML")

    log.info(
        "MT HTML parsed: account_info keys=%s, trades=%d",
        list(account_info.keys()),
        len(trades),
    )
    return {"account_info": account_info, "trades": trades}


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "risk-manager-api",
        "version": "3.4.0",
        "configured": {
            "anthropic_key": bool(ANTHROPIC_API_KEY),
            "api_secret": bool(API_SECRET),
            "model": MODEL,
            "ai_provider": "anthropic",
            "mock_ai": False,
        },
    }


@app.post("/audit", dependencies=[Depends(require_api_key)])
def audit(req: AuditRequest) -> dict:
    return run_audit(req)


@app.post("/analyze", dependencies=[Depends(require_api_key)])
def analyze(req: AuditRequest) -> dict:
    return run_audit(req)


@app.post("/full", dependencies=[Depends(require_api_key)])
def full(req: AuditRequest) -> dict:
    return run_audit(req)


@app.post("/narrate", dependencies=[Depends(require_api_key)])
def narrate(req: AuditRequest) -> dict:
    return run_audit(req)


@app.post("/audit-html", dependencies=[Depends(require_api_key)])
async def audit_html(file: UploadFile = File(...)) -> dict:
    """Accept an MT4/MT5 HTML statement, parse it, and enrich with market data.

    Workflow:
      1. Read and parse the uploaded HTML file.
      2. Extract unique symbols from the parsed trades.
      3. Fetch price data (Massive API) and economic events (Apify) in parallel.
      4. Return the enriched payload — ready for a subsequent /audit call or
         for direct consumption by the frontend.

    Authentication: X-API-Key header (same secret as all other endpoints).
    Content-Type: multipart/form-data with a single ``file`` field.
    """


    started = time.time()

    # ── Read uploaded file ────────────────────────────────────────────────────

    try:
        raw_bytes = await file.read()
        html_content = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        log.exception("Failed to read uploaded HTML file")
        raise HTTPException(400, "Could not read uploaded file. Ensure it is a valid HTML file.")

    if not html_content.strip():
        raise HTTPException(400, "Uploaded file is empty.")

    # ── Parse MT HTML ─────────────────────────────────────────────────────────
    parsed = parse_mt_html(html_content)
    trades: list[dict[str, Any]] = parsed.get("trades", [])
    account_info: dict[str, Any] = parsed.get("account_info", {})

    if parsed.get("parse_error"):
        log.warning("MT HTML parse error: %s", parsed["parse_error"])

    # ── Extract unique symbols ────────────────────────────────────────────────
    symbols: list[str] = list(
        {str(t["symbol"]) for t in trades if t.get("symbol")}
    )

    # ── Parallel enrichment ───────────────────────────────────────────────────
    price_task = asyncio.create_task(get_price_data(symbols))
    events_task = asyncio.create_task(get_economic_events(symbols))
    price_data, economic_events = await asyncio.gather(price_task, events_task)

    latency_ms = int((time.time() - started) * 1000)

    # ── Basic audit summary ───────────────────────────────────────────────────
    total_profit = 0.0
    for t in trades:
        try:
            total_profit += float(str(t.get("profit", "0")).replace(",", "") or 0)
        except (ValueError, TypeError):
            pass

    audit_summary = {
        "tradesFound": len(trades),
        "symbolsFound": symbols,
        "totalProfit": round(total_profit, 2),
        "parseWarning": parsed.get("parse_error"),
        "note": (
            "Basic summary only. Submit trades to /audit with a full preset "
            "and metrics payload for a complete compliance audit."
        ),
    }

    log.info(
        "audit-html ok trades=%d symbols=%d latency_ms=%d",
        len(trades),
        len(symbols),
        latency_ms,
    )

    return {
        "ok": True,
        "data": {
            "accountInfo": account_info,
            "trades": trades,
            "priceData": price_data,
            "economicEvents": economic_events,
            "auditSummary": audit_summary,
        },
        "latencyMs": latency_ms,
    }


# Body size cap — reject oversized payloads before parsing.
@app.middleware("http")
async def body_size_guard(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            n = int(cl)
            if n > MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "ok": False,
                        "error": (
                            f"Request body too large ({n} bytes, limit "
                            f"{MAX_REQUEST_BYTES})."
                        ),
                        "status": 413,
                    },
                )
        except ValueError:
            pass
    return await call_next(request)


# Logging middleware ligero — útil en Railway para ver latencias
@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - started) * 1000)
    log.info(
        "%s %s → %s (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format errors to match the riskmanager.functions.ts client expectations."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": exc.detail,
            "status": exc.status_code,
        },
    )
