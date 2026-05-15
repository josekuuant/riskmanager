"""Crear el memory store de reglas y poblarlo desde ./rules/.

Estructura esperada de ./rules/:

    rules/
      1step/
        general.md
        drawdown.pdf
      2step/
        phase1.md
        phase2.md
        funded.md
      instant/
        rules.txt

Cada archivo se sube como una memoria en el path equivalente dentro del store:
  ./rules/2step/phase1.md  →  /2step/phase1.md

Archivos .pdf se extraen a texto con pypdf. Archivos .md / .txt se suben tal
cual. Otros formatos se ignoran con un warning.

Uso:
  pip install anthropic pypdf
  export ANTHROPIC_API_KEY=...
  python seed_memory.py            # crea un store nuevo
  python seed_memory.py --reseed   # borra y recrea memorias en un store existente
                                   #   (requiere MEMORY_STORE_ID en el env)
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic

RULES_DIR = Path("./rules")
SUPPORTED_TEXT = {".md", ".txt"}
SUPPORTED_PDF = {".pdf"}
MAX_MEMORY_BYTES = 100_000  # tope por memoria


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("pypdf no instalado. `pip install pypdf` para procesar PDFs.")
    reader = PdfReader(str(path))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()


def collect_rule_files() -> list[tuple[Path, str]]:
    """Devuelve (local_path, memory_path) por cada archivo soportado en ./rules/."""
    if not RULES_DIR.is_dir():
        sys.exit(
            f"No existe {RULES_DIR}/. Crea la estructura:\n"
            "  mkdir -p rules/{1step,2step,instant}\n"
            "  # y deja tus archivos de reglas dentro"
        )
    entries: list[tuple[Path, str]] = []
    for path in sorted(RULES_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT:
            content = path.read_text(encoding="utf-8")
        elif suffix in SUPPORTED_PDF:
            content = extract_pdf(path)
            if not content:
                print(f"⚠  {path}: PDF sin texto extraíble, omitido")
                continue
        else:
            print(f"⚠  {path}: tipo no soportado ({suffix}), omitido")
            continue
        if len(content.encode("utf-8")) > MAX_MEMORY_BYTES:
            print(
                f"⚠  {path}: excede {MAX_MEMORY_BYTES} bytes — pártelo en archivos más chicos"
            )
            continue
        rel = path.relative_to(RULES_DIR).with_suffix(".md")
        mem_path = "/" + rel.as_posix()
        entries.append((path, mem_path))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="Borrar y recrear memorias en un store existente (MEMORY_STORE_ID en env)",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()

    # 1. Crear o reutilizar el memory store
    store_id = os.environ.get("MEMORY_STORE_ID")
    if args.reseed:
        if not store_id:
            sys.exit("--reseed requiere MEMORY_STORE_ID en el env")
        print(f"Reseed sobre store existente: {store_id}")
    else:
        store = client.beta.memory_stores.create(
            name="Prop Firm Rules",
            description=(
                "Reglas del prop firm organizadas por modelo de cuenta. "
                "Subdirectorios: /1step/, /2step/, /instant/. SIEMPRE lee "
                "TODOS los archivos del subdirectorio que corresponde al "
                "modelo de la cuenta que se está revisando ANTES de evaluar."
            ),
        )
        store_id = store.id
        print(f"✓ Memory store creado: {store_id}")

    # 2. Si --reseed, borrar memorias existentes primero
    if args.reseed:
        existing = list(client.beta.memory_stores.memories.list(store_id))
        for mem in existing:
            if getattr(mem, "type", "memory") != "memory":
                continue
            client.beta.memory_stores.memories.delete(
                mem.id, memory_store_id=store_id
            )
            print(f"  - borrado: {mem.path}")

    # 3. Subir cada archivo de reglas
    rule_files = collect_rule_files()
    if not rule_files:
        sys.exit(f"No se encontraron archivos de reglas en {RULES_DIR}/")

    for local_path, mem_path in rule_files:
        content = (
            extract_pdf(local_path)
            if local_path.suffix.lower() in SUPPORTED_PDF
            else local_path.read_text(encoding="utf-8")
        )
        client.beta.memory_stores.memories.create(
            store_id,
            path=mem_path,
            content=content,
        )
        print(f"  + {local_path}  →  {mem_path}")

    print()
    print(f"Listo. Guarda este ID en tu .env:")
    print(f"  MEMORY_STORE_ID={store_id}")


if __name__ == "__main__":
    main()
