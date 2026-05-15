"""FastAPI wrapper para el motor de reglas — expone HTTP endpoints.

Endpoints:
  POST /analyze   — Sube un reporte MT5 HTML + parámetros, recibe JSON del engine.
                    100% determinístico, no llama a Anthropic. Más barato y rápido.
  POST /narrate   — Recibe el JSON del análisis y devuelve reporte + email.
                    Llama a Anthropic. Use cuando necesitás los textos generados.
  POST /full      — Combinación: sube reporte + recibe análisis + reporte + email
                    en una sola llamada. Más caro pero atómico.
  GET  /health    — Healthcheck para deploys.

Auth: API key en header `X-API-Key` (configurar API_SECRET en env).
CORS: ajustar ALLOWED_ORIGINS para tu dominio Lovable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import anthropic
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from mt5_parser import parse, read_html
from rules_engine import RULES, evaluate

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

API_SECRET = os.environ.get("API_SECRET", "change-me-in-production")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")
    if o.strip()
]

app = FastAPI(
    title="NYS Risk Manager API",
    description="Engine determinístico de reglas + agente narrador para revisión de cuentas",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def require_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def normalize_phase(model: str, phase: Optional[str]) -> Optional[str]:
    if model == "instant":
        return None
    if model == "1step":
        return phase or "evaluation"
    if model == "2step":
        if phase not in ("phase1", "phase2", "funded"):
            raise HTTPException(400, "Phase required for 2step: phase1|phase2|funded")
        return phase
    raise HTTPException(400, f"Unknown model: {model}")


def parse_report(content_bytes: bytes) -> tuple[float, list, dict]:
    """Guarda bytes a archivo temp, llama al parser, limpia."""
    with NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(content_bytes)
        path = Path(f.name)
    try:
        html = read_html(path)
        return parse(html)
    finally:
        path.unlink(missing_ok=True)


def run_engine(
    report_bytes: bytes,
    model: str,
    phase: Optional[str],
    size: Optional[int],
    server_utc_offset: int,
    previous_payouts: int,
) -> dict:
    """Lee reporte + corre engine. Devuelve dict JSON-ready."""
    phase_norm = normalize_phase(model, phase)
    key = (model, phase_norm if model != "instant" else None)
    if key not in RULES:
        raise HTTPException(400, f"Combinación no soportada: model={model} phase={phase_norm}")

    initial_balance, trades, summary = parse_report(report_bytes)
    if size:
        initial_balance = float(size)

    result = evaluate(
        model=model,
        phase=phase_norm,
        initial_balance=initial_balance,
        trades=trades,
        server_utc_offset_hours=server_utc_offset,
        n_previous_payouts=previous_payouts,
    )
    out = result.to_dict()
    out["mt5_summary"] = summary
    return out


def call_agent(analysis: dict) -> str:
    """Llama al agente narrador (Anthropic Managed Agents).

    Requiere env vars: ANTHROPIC_API_KEY, AGENT_ID, ENV_ID.
    """
    agent_id = os.environ.get("AGENT_ID")
    env_id = os.environ.get("ENV_ID")
    if not (agent_id and env_id):
        raise HTTPException(500, "AGENT_ID y ENV_ID deben estar configurados en el server")

    client = anthropic.Anthropic()

    trader_name = (analysis.get("mt5_summary") or {}).get("Name")
    kickoff = (
        "Hola — necesito que revises esta cuenta y produzcas los TRES outputs "
        "(veredicto, reporte interno, email al cliente). El análisis "
        "determinístico ya está hecho — usá los números del JSON tal cual.\n\n"
    )
    if trader_name:
        kickoff += f"Nombre del trader: {trader_name}\n\n"
    kickoff += f"```json\n{json.dumps(analysis, indent=2, default=str)}\n```"

    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        title=f"API session — {analysis.get('model','?')} {analysis.get('phase') or ''}",
    )

    full_response: list[str] = []
    with client.beta.sessions.events.stream(session_id=session.id) as stream:
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": kickoff}]}],
        )
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        full_response.append(block.text)
            elif event.type == "session.status_terminated":
                break
            elif event.type == "session.status_idle":
                if event.stop_reason.type == "requires_action":
                    continue
                break

    return "".join(full_response)


# ────────────────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────────────────


class NarrateRequest(BaseModel):
    analysis: dict


class FullResponse(BaseModel):
    analysis: dict
    narration: str


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "risk-manager-api"}


@app.post("/analyze", dependencies=[Depends(require_api_key)])
def analyze(
    report: UploadFile = File(..., description="MT5 ReportHistory.html"),
    model: str = Form(..., description="instant | 1step | 2step"),
    phase: Optional[str] = Form(None, description="evaluation | phase1 | phase2 | funded"),
    size: Optional[int] = Form(None, description="Account size en USD (override)"),
    server_utc_offset: int = Form(0),
    previous_payouts: int = Form(0),
) -> dict:
    """Análisis determinístico: corre el engine sobre el reporte."""
    content = report.file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    return run_engine(content, model, phase, size, server_utc_offset, previous_payouts)


@app.post("/narrate", dependencies=[Depends(require_api_key)])
def narrate(req: NarrateRequest) -> dict:
    """Llama al agente para narrar un análisis ya hecho."""
    narration = call_agent(req.analysis)
    return {"narration": narration}


@app.post("/full", dependencies=[Depends(require_api_key)], response_model=FullResponse)
def full(
    report: UploadFile = File(...),
    model: str = Form(...),
    phase: Optional[str] = Form(None),
    size: Optional[int] = Form(None),
    server_utc_offset: int = Form(0),
    previous_payouts: int = Form(0),
) -> FullResponse:
    """One-shot: análisis + narración. Más caro pero atómico."""
    content = report.file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    analysis = run_engine(content, model, phase, size, server_utc_offset, previous_payouts)
    narration = call_agent(analysis)
    return FullResponse(analysis=analysis, narration=narration)
