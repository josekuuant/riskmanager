"""Risk Manager API — Anthropic Messages API con structured output via tool calling.

Esta API recibe trades + metrics ya calculados por el frontend Lovable
(TanStack Start) y delega el AUDIT a Claude usando tool calling — Claude
no recalcula, solo analiza los findings y produce el email + recomendación.

Contract (compatible con riskmanager.functions.ts del proyecto Lovable):

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
      finalDecision: "PASSED" | "WARNING" | "BREACH" | "MANUAL_REVIEW",
      severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
      executiveSummary: str,
      confirmedBreaches: [{ rule, status, severity, dataQuality,
                            explanation, calculation, relatedTickets,
                            allowed, actual }, ...],
      estimatedBreaches: [...],
      warnings: [...],
      notEnoughData: [...],
      ruleByRuleAnalysis: [...],
      evidenceTable: [{ ticket, symbol, time, observation, ruleRef }, ...],
      internalRecommendation: "APPROVE PAYOUT" | "REJECT PAYOUT" |
                              "PARTIAL APPROVAL" | "MANUAL REVIEW REQUIRED",
      emailSubject: str,
      emailBody: str
    }

  GET /health  — sin auth, para healthchecks
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, Security
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
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8000"))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("riskmanager")

app = FastAPI(
    title="NYS Risk Manager API",
    description="Payout audit con Claude (Anthropic Messages API + tool calling)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def require_api_key(api_key: str = Security(api_key_header)) -> str:
    if not API_SECRET:
        # Hard fail si no configuraron el secret — no queremos open API
        raise HTTPException(503, "API_SECRET not configured on server")
    if api_key != API_SECRET:
        raise HTTPException(403, "Invalid API key")
    return api_key


# ────────────────────────────────────────────────────────────────────────────
# System prompt + tool schema  (idénticos a analyze.functions.ts del Lovable
# para que el agente se comporte consistentemente sin importar el provider)
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior Prop Firm Risk Manager and Trading Compliance Analyst.

Your job is to analyze MetaTrader account reports and determine whether a trader appears to have breached payout review rules.

You receive:
- Account metadata
- Selected rule preset
- Parsed trades (may be capped to a representative sample if very large)
- Deterministic metrics (authoritative pre-computed calculations)
- SL/TP data
- Balance, equity and floating P&L if available

Critical rules:
1. Do not invent data.
2. Do not fabricate prices, tickets, timestamps, equity, SL, TP or calculations.
3. Clearly separate confirmed breaches from estimated breaches.
4. If data is insufficient, mark the rule as NOT_ENOUGH_DATA or MANUAL_REVIEW.
5. Uploaded MetaTrader data is the primary source of evidence.
6. Use exact tickets, symbols, timestamps, volumes, entry prices, close prices, SL, TP and P&L from the provided data — never approximate or invent these values.
7. Generate a professional email that the admin can copy and send manually.
8. Do not send emails automatically. The emailBody is a draft only.
9. Do not recommend unsupported or unfair actions.
10. Use firm, professional, compliance-oriented language.

Treat the deterministic metrics as ground truth — your job is to explain, contextualize and present them, not recompute them.

Analyze every applicable rule:
- Profit Target
- Daily Loss Limit
- Maximum Loss
- Max Risk Per Trade Idea
- Max Exposure Per Symbol
- Consistency Rule
- Minimum Trading Days
- Trades without SL
- Prohibited strategy patterns (martingale, grid, stacking, tick scalping, no-SL high risk, lot size spikes, hedging/mirroring)
- Payout eligibility
- Manual review items

Email output (emailSubject + emailBody) — must be ready for the admin to copy and send to the trader without edits.

Required structure for emailBody:
1. Greeting using the trader's name (e.g. "Dear {{traderName}},").
2. State that the Risk & Compliance team has completed a review of the payout request.
3. Reference the account number explicitly.
4. State the result clearly (passed / warning / breach / manual review required).
5. Explain the breached rule(s) or manual-review reason in plain language.
6. Include an Evidence section listing each relevant trade with: ticket, symbol, direction, volume, open time, close time, entry price, close price, SL, TP, P&L, allowed threshold, and actual result. Use only values present in the input data — write "N/A" when a field is missing. Format as a readable list or aligned text block (no markdown tables).
7. Briefly explain in simple language why the rule was breached or why review is required.
8. State the next step clearly: payout approved, payout rejected, partial approval, or manual review required.
9. Professional closing signed by "Risk & Compliance Team".

Email tone:
- Professional, firm, respectful, compliance-oriented.
- No accusations unless proven by uploaded data. For estimated or pattern findings, use language such as "The review identified a pattern consistent with..." or "This requires manual review...".
- No aggressive, emotional, or apologetic wording.
- Prefer phrasings like: "The review identified...", "Based on the available account data...", "The account exceeded...", "This requires manual review...".
- Do not include internal notes, reasoning, confidence scores, provider names, or references to AI/automation.
- Do not promise outcomes outside the stated next step.
- emailSubject must be concise and reference the account number and outcome (e.g. "Payout Review – Account 12345 – Manual Review Required").

Respond ONLY by calling the submit_audit_result tool. Use UPPER_SNAKE_CASE enum values exactly. Return valid structured output only — no prose outside the tool call."""


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
                    "required": ["ticket", "symbol", "time", "observation", "ruleRef"],
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


# ────────────────────────────────────────────────────────────────────────────
# Request / response models
# ────────────────────────────────────────────────────────────────────────────


class AuditRequest(BaseModel):
    """Lovable envía este shape. Aceptamos cualquier campo extra para forward-compat."""

    account: dict[str, Any] = Field(..., description="AccountMetadata")
    preset: dict[str, Any] = Field(..., description="RulePreset")
    trades: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# ────────────────────────────────────────────────────────────────────────────
# Anthropic call
# ────────────────────────────────────────────────────────────────────────────


def _truncate_trades(trades: list[dict]) -> tuple[list[dict], int]:
    """Si hay muchos trades, mandamos un sample y el agente lo sabe.

    Estrategia: keepamos el primero y el último siempre (boundaries), y
    samplemos uniformemente en el medio para preservar distribución temporal.
    """
    n = len(trades)
    if n <= MAX_TRADES_IN_PROMPT:
        return trades, 0
    step = n / MAX_TRADES_IN_PROMPT
    indices = sorted({int(i * step) for i in range(MAX_TRADES_IN_PROMPT)} | {0, n - 1})
    sampled = [trades[i] for i in indices]
    return sampled, n - len(sampled)


def run_audit(req: AuditRequest) -> dict:
    """Llama a Claude con el contexto, fuerza tool call, retorna structured output."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured on server")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    trades_for_prompt, dropped = _truncate_trades(req.trades)
    note = ""
    if dropped:
        note = (
            f"\n\nNOTE: trades array was sampled from {len(req.trades)} total "
            f"down to {len(trades_for_prompt)} representative entries to fit "
            "context window. Counts and metrics are computed on the FULL set; "
            "use `metrics` as ground truth for totals."
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
        f"result by calling submit_audit_result.{note}\n\n"
        f"```json\n{json.dumps(user_payload, indent=2, default=str)}\n```"
    )

    started = time.time()
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[SUBMIT_AUDIT_TOOL],
            tool_choice={"type": "tool", "name": "submit_audit_result"},
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIStatusError as e:
        log.error("Anthropic API error %s: %s", e.status_code, e.message)
        raise HTTPException(
            502,
            f"Anthropic API error ({e.status_code}): {e.message[:300]}",
        )
    except anthropic.APIConnectionError:
        raise HTTPException(504, "Could not reach Anthropic API")
    except Exception as e:
        log.exception("Unexpected error calling Anthropic")
        raise HTTPException(500, f"Internal error: {type(e).__name__}")

    latency_ms = int((time.time() - started) * 1000)

    # Extraer el tool_use block — debería ser el primero por tool_choice forzado
    tool_block = next(
        (b for b in message.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        log.warning("No tool_use block in response. stop_reason=%s", message.stop_reason)
        # Mostrar texto crudo si lo hubo, para debug
        text_blocks = [
            getattr(b, "text", "") for b in message.content if getattr(b, "type", None) == "text"
        ]
        raise HTTPException(
            502,
            f"Model did not call the tool. stop_reason={message.stop_reason}. "
            f"Text: {' '.join(text_blocks)[:400]}",
        )

    return {
        "ok": True,
        "data": tool_block.input,
        "latencyMs": latency_ms,
        "model": MODEL,
        "usage": {
            "input_tokens": getattr(message.usage, "input_tokens", 0),
            "output_tokens": getattr(message.usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(
                message.usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(
                message.usage, "cache_read_input_tokens", 0
            ),
        },
        "trades_dropped": dropped,
    }


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "risk-manager-api",
        "version": "2.0.0",
        "configured": {
            "anthropic_key": bool(ANTHROPIC_API_KEY),
            "api_secret": bool(API_SECRET),
            "model": MODEL,
        },
    }


@app.post("/audit", dependencies=[Depends(require_api_key)])
def audit(req: AuditRequest) -> dict:
    return run_audit(req)


# Aliases — el riskmanager.functions.ts del frontend Lovable acepta
# endpoint = "analyze" | "full" | "narrate". Los 3 son el mismo audit ahora.
@app.post("/analyze", dependencies=[Depends(require_api_key)])
def analyze(req: AuditRequest) -> dict:
    return run_audit(req)


@app.post("/full", dependencies=[Depends(require_api_key)])
def full(req: AuditRequest) -> dict:
    return run_audit(req)


@app.post("/narrate", dependencies=[Depends(require_api_key)])
def narrate(req: AuditRequest) -> dict:
    return run_audit(req)


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
