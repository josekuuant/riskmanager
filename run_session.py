"""Revisa una cuenta de prop firm contra las reglas del modelo.

Uso:
    export ANTHROPIC_API_KEY=...
    # Variables de setup (ver setup en chat / .env.example):
    export AGENT_ID=agent_...
    export ENV_ID=env_...
    export MEMORY_STORE_ID=memstore_...

    python run_session.py <modelo> <ruta-al-csv-de-trades>

Ejemplo:
    python run_session.py 2step ./trades/account_98423.xlsx

<modelo> debe ser uno de: 1step, 2step, instant
"""

import argparse
import os
import re
import sys
from pathlib import Path

import anthropic

VALID_MODELS = {"1step", "2step", "instant"}
MODEL_LABEL = {"1step": "1-step", "2step": "2-step", "instant": "instant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(VALID_MODELS))
    parser.add_argument("trades", type=Path, help="CSV o XLSX con el historial de trades")
    parser.add_argument(
        "--save-outputs",
        type=Path,
        default=Path("./outputs"),
        help="Directorio donde guardar el reporte y el email (default: ./outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trades.is_file():
        sys.exit(f"No existe el archivo de trades: {args.trades}")

    agent_id = os.environ["AGENT_ID"]
    env_id = os.environ["ENV_ID"]
    memory_store_id = os.environ["MEMORY_STORE_ID"]

    client = anthropic.Anthropic()

    # 1. Subir el CSV/XLSX de trades
    with args.trades.open("rb") as f:
        uploaded = client.beta.files.upload(file=f)
    print(f"✓ Trades subidos: {uploaded.id}")

    # 2. Crear la sesión con memory store (reglas, read-only) + archivo de trades
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        title=f"Revisión {MODEL_LABEL[args.model]} — {args.trades.name}",
        resources=[
            {
                "type": "memory_store",
                "memory_store_id": memory_store_id,
                "access": "read_only",
                "instructions": (
                    f"Reglas del prop firm. Lee TODOS los archivos en "
                    f"/{args.model}/ ANTES de evaluar la cuenta."
                ),
            },
            {
                "type": "file",
                "file_id": uploaded.id,
                "mount_path": f"/workspace/{args.trades.name}",
            },
        ],
    )
    print(f"✓ Sesión: {session.id}")
    print(
        f"  Console: https://platform.claude.com/workspaces/default/sessions/{session.id}",
        flush=True,
    )

    # 3. Stream-first: abrir stream ANTES de enviar el kickoff
    full_response: list[str] = []
    kickoff = (
        f"hello necesito revisar esta cuenta, es {MODEL_LABEL[args.model]} model. "
        f"Los trades están en /workspace/{args.trades.name}."
    )

    with client.beta.sessions.events.stream(session_id=session.id) as stream:
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": kickoff}],
                }
            ],
        )

        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        sys.stdout.write(block.text)
                        sys.stdout.flush()
                        full_response.append(block.text)
            elif event.type == "agent.tool_use":
                print(f"\n[tool] {event.name}", flush=True)
            elif event.type == "session.status_terminated":
                print("\n[sesión terminada]", flush=True)
                break
            elif event.type == "session.status_idle":
                # Romper solo si el stop_reason es terminal. requires_action
                # significa que está esperando algo del cliente — seguir.
                if event.stop_reason.type == "requires_action":
                    continue
                print(f"\n[idle: {event.stop_reason.type}]", flush=True)
                break

    # 4. Guardar outputs separados (veredicto, reporte interno, email)
    args.save_outputs.mkdir(parents=True, exist_ok=True)
    full_text = "".join(full_response)
    stem = args.trades.stem

    (args.save_outputs / f"{stem}_full.md").write_text(full_text, encoding="utf-8")

    verdict = extract_verdict(full_text)
    if verdict:
        (args.save_outputs / f"{stem}_verdict.txt").write_text(verdict, encoding="utf-8")
        print(f"\n→ Veredicto guardado: {args.save_outputs / f'{stem}_verdict.txt'}")
    print(f"→ Respuesta completa: {args.save_outputs / f'{stem}_full.md'}")


def extract_verdict(text: str) -> str | None:
    """Extrae el bloque de veredicto del output del agente."""
    match = re.search(
        r"VEREDICTO:\s*\w+.*?PAYOUT_RECOMENDADO:\s*[^\n]+",
        text,
        re.DOTALL,
    )
    return match.group(0).strip() if match else None


if __name__ == "__main__":
    main()
