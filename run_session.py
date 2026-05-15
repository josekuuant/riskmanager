"""Revisa una cuenta NYS combinando engine determinístico + agente narrador.

Flujo:
  1. Local: corre rules_engine.evaluate() sobre el reporte → JSON estructurado
  2. Anthropic: pasa ese JSON al agente. El agente NO recalcula — solo escribe
     reporte interno + email al cliente a partir de los findings del engine.

Uso:
    export ANTHROPIC_API_KEY=...
    export AGENT_ID=agent_...
    export ENV_ID=env_...
    # MEMORY_STORE_ID YA NO ES NECESARIO — las reglas están en el código.

    python run_session.py <modelo> <reporte> [--phase ...] [--size N] [--server-utc-offset N]

Ejemplos:
    python run_session.py instant trades/ReportHistory9708.html --size 10000
    python run_session.py 2step trades/cuenta.xlsx --phase phase1
    python run_session.py 1step trades/funded.html --phase funded --size 25000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import anthropic

from mt5_parser import parse, read_html
from rules_engine import RULES, evaluate, to_json

VALID_MODELS = {"instant", "1step", "2step"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(VALID_MODELS))
    p.add_argument("trades", type=Path)
    p.add_argument(
        "--phase",
        choices=["evaluation", "phase1", "phase2", "funded"],
        help="Para 1-step: evaluation|funded. Para 2-step: phase1|phase2|funded.",
    )
    p.add_argument("--size", type=int, help="Account size USD (override del balance detectado)")
    p.add_argument("--server-utc-offset", type=int, default=0,
                   help="Offset horario del server MT5 vs UTC (default 0)")
    p.add_argument("--save-outputs", type=Path, default=Path("./outputs"))
    return p.parse_args()


def normalize_phase(model: str, phase: str | None) -> str | None:
    if model == "instant":
        return None
    if model == "1step":
        return phase or "evaluation"
    if model == "2step":
        if phase is None:
            sys.exit("--phase requerido para 2-step (phase1 | phase2 | funded)")
        return phase
    return phase


def main() -> None:
    args = parse_args()
    if not args.trades.is_file():
        sys.exit(f"No existe: {args.trades}")

    phase = normalize_phase(args.model, args.phase)
    key = (args.model, phase if args.model != "instant" else None)
    if key not in RULES:
        sys.exit(f"Combinación no soportada: model={args.model} phase={phase}")

    agent_id = os.environ["AGENT_ID"]
    env_id = os.environ["ENV_ID"]

    # ────────────────────────────────────────────────────────────────────
    # STEP 1 — Análisis determinístico LOCAL
    # ────────────────────────────────────────────────────────────────────
    print("→ Corriendo motor de reglas (determinístico)...", file=sys.stderr)

    # Para HTML usamos el parser MT5; para CSV/XLSX habría que armar parsers similares.
    if args.trades.suffix.lower() in {".html", ".htm"}:
        html = read_html(args.trades)
        initial_balance, trades, mt5_summary = parse(html)
    else:
        sys.exit(
            f"Formato {args.trades.suffix} no implementado todavía en mt5_parser. "
            "Conviértelo a HTML de MT5 o ampliá el parser."
        )

    if args.size:
        initial_balance = float(args.size)

    result = evaluate(
        model=args.model,
        phase=phase,
        initial_balance=initial_balance,
        trades=trades,
        server_utc_offset_hours=args.server_utc_offset,
    )

    print(f"  → Veredicto del motor: {result.verdict}", file=sys.stderr)
    for f in result.findings:
        print(f"     [{f.severity:7s}] {f.rule}", file=sys.stderr)

    # ────────────────────────────────────────────────────────────────────
    # STEP 2 — Agente NARRA (reporte + email)
    # ────────────────────────────────────────────────────────────────────
    print("→ Llamando al agente para narrar el resultado...", file=sys.stderr)

    analysis_json = result.to_dict()
    analysis_json["mt5_summary"] = mt5_summary
    # Trader name si está en el report (MT5 lo incluye en el header)
    trader_name = mt5_summary.get("Name") or None

    client = anthropic.Anthropic()

    kickoff_text = (
        "Hola — necesito que revises esta cuenta y produzcas los TRES outputs "
        "(veredicto, reporte interno, email al cliente) según tu instrucción.\n\n"
        "El análisis determinístico ya está hecho. Acá va el JSON con todas las "
        "métricas, findings y veredicto del motor. No recalcules — usá estos "
        "números tal cual.\n\n"
    )
    if trader_name:
        kickoff_text += f"Nombre del trader: {trader_name}\n\n"
    kickoff_text += f"```json\n{json.dumps(analysis_json, indent=2, default=str)}\n```"

    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        title=f"Revisión {args.model}{f' {phase}' if phase else ''} — {args.trades.name}",
    )
    print(f"  Sesión: {session.id}", file=sys.stderr)
    print(
        f"  Console: https://platform.claude.com/workspaces/default/sessions/{session.id}",
        file=sys.stderr,
        flush=True,
    )

    full_response: list[str] = []
    with client.beta.sessions.events.stream(session_id=session.id) as stream:
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": kickoff_text}],
            }],
        )

        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        sys.stdout.write(block.text)
                        sys.stdout.flush()
                        full_response.append(block.text)
            elif event.type == "session.status_terminated":
                print("\n[sesión terminada]", file=sys.stderr)
                break
            elif event.type == "session.status_idle":
                if event.stop_reason.type == "requires_action":
                    continue
                print(f"\n[idle: {event.stop_reason.type}]", file=sys.stderr)
                break

    # ────────────────────────────────────────────────────────────────────
    # STEP 3 — Guardar outputs
    # ────────────────────────────────────────────────────────────────────
    args.save_outputs.mkdir(parents=True, exist_ok=True)
    stem = args.trades.stem
    full_text = "".join(full_response)

    # Análisis crudo del motor
    (args.save_outputs / f"{stem}_engine.json").write_text(
        json.dumps(analysis_json, indent=2, default=str), encoding="utf-8"
    )
    # Respuesta completa del agente
    (args.save_outputs / f"{stem}_agent.md").write_text(full_text, encoding="utf-8")

    # Email aislado (si el agente lo emitió en un bloque ```)
    email_match = re.search(r"Asunto:.*?(?=```|\Z)", full_text, re.DOTALL)
    if email_match:
        (args.save_outputs / f"{stem}_email.txt").write_text(
            email_match.group(0).strip(), encoding="utf-8"
        )

    print(f"\n→ Engine output: {args.save_outputs / f'{stem}_engine.json'}", file=sys.stderr)
    print(f"→ Agent output:  {args.save_outputs / f'{stem}_agent.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
