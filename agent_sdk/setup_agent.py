"""
One-time setup: create the Managed Agent + Environment.

Run this ONCE, store the printed AGENT_ID and ENVIRONMENT_ID, then use them
in audit_runner.run_audit() via the AGENT_ID / ENVIRONMENT_ID env vars.

To UPDATE the agent later (new prompt / new tool schema), call
client.beta.agents.update(agent_id=..., version=...) — that creates a new
immutable version. Running sessions keep their pinned version; new sessions
get the latest.

This script uses the Anthropic SDK directly so you can do it from Python.
The equivalent terminal flow with the Anthropic CLI is:

    AGENT_ID=$(ant beta:agents create < agent.yaml --transform id -r)
    ENV_ID=$(ant beta:environments create < environment.yaml --transform id -r)

(Use scripts/build_agent_yaml.py first to splice the SYSTEM_PROMPT in.)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
API_PY = REPO_ROOT / "api.py"


def _load_api_module():
    """Load riskmanager/api.py without running its FastAPI server-side code paths."""
    # Stub the env vars api.py reads at import time so the module loads
    # cleanly even on a dev machine that has no Anthropic key set.
    os.environ.setdefault("API_SECRET", "stub_for_import")
    os.environ.setdefault("ANTHROPIC_API_KEY", "stub_for_import")
    spec = importlib.util.spec_from_file_location("riskmanager_api", str(API_PY))
    if spec is None or spec.loader is None:
        sys.exit(f"Could not load {API_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    api = _load_api_module()
    system_prompt: str = api.SYSTEM_PROMPT
    tool_input_schema: dict = api.SUBMIT_AUDIT_TOOL["input_schema"]

    client = anthropic.Anthropic()

    # 1. Environment — minimal, no networking, no tool execution expected.
    env = client.beta.environments.create(
        name="nys-risk-manager-env",
        description="Sandbox for the NYS Risk Manager — agent only emits a custom-tool call.",
        config={
            "type": "cloud",
            "networking": {
                "type": "package_managers_and_custom",
                "allowed_hosts": [],
            },
        },
    )
    print(f"ENVIRONMENT_ID={env.id}")

    # 2. Agent — single custom tool, no agent_toolset, no MCP servers.
    agent = client.beta.agents.create(
        name="NYS Risk Manager",
        model="claude-opus-4-7",
        description=(
            "Senior NYS Markets Risk & Compliance Analyst. Audits a trader's "
            "MetaTrader account for payout eligibility. Every response is a "
            "structured submit_audit_result tool call — no prose."
        ),
        system=system_prompt,
        tools=[
            {
                "type": "custom",
                "name": "submit_audit_result",
                "description": (
                    "Submit the structured payout audit result. Every audit MUST "
                    "end with exactly one call to this tool."
                ),
                "input_schema": tool_input_schema,
            }
        ],
        metadata={
            "source": "riskmanager",
            "version": getattr(api.app, "version", "3.4.0"),
        },
    )
    print(f"AGENT_ID={agent.id}")
    print(f"AGENT_VERSION={agent.version}")
    print()
    print("Save these to your secret store / .env. Then run an audit with:")
    print("  AGENT_ID=... ENVIRONMENT_ID=... python -m agent_sdk.audit_runner < payload.json")


if __name__ == "__main__":
    main()
