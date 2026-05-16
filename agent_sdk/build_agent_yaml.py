#!/usr/bin/env python3
"""
Splice the SYSTEM_PROMPT from riskmanager/api.py into agent.yaml.

The CMA YAML file checked into git carries a placeholder for the system
prompt — we don't want a 16K block-literal making the YAML hard to read or
diff. This script reads the live SYSTEM_PROMPT constant from api.py and
emits a ready-to-apply agent.yaml.

Usage:
    python build_agent_yaml.py > agent.built.yaml
    ant beta:agents create < agent.built.yaml --transform id -r
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # riskmanager/
TEMPLATE = HERE / "agent.yaml"
API_PY = REPO_ROOT / "api.py"


def load_system_prompt() -> str:
    """Pull the SYSTEM_PROMPT string literal out of api.py without importing."""
    spec_text = API_PY.read_text(encoding="utf-8")
    marker = 'SYSTEM_PROMPT = """'
    start = spec_text.find(marker)
    if start == -1:
        sys.exit("Could not find SYSTEM_PROMPT in api.py")
    body_start = start + len(marker)
    end = spec_text.find('"""', body_start)
    if end == -1:
        sys.exit("SYSTEM_PROMPT triple-quote not terminated in api.py")
    return spec_text[body_start:end]


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    if "# SYSTEM_PROMPT placeholder." not in template:
        sys.exit("agent.yaml is missing the SYSTEM_PROMPT placeholder block.")

    prompt = load_system_prompt()
    # YAML block-literal: every prompt line indented 2 spaces under `system: |`.
    indented = "\n".join("  " + line if line else "" for line in prompt.split("\n"))

    out = template.replace(
        "system: |\n"
        "  # SYSTEM_PROMPT placeholder.\n"
        "  # Replace this block with the contents of riskmanager/api.py SYSTEM_PROMPT\n"
        "  # (use scripts/build_agent_yaml.py to splice it in automatically).",
        f"system: |\n{indented}",
    )
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
