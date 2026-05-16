"""
Risk Manager — Managed Agents runner.

Port of riskmanager/api.py:run_audit() onto Anthropic Managed Agents instead
of a single Messages API call. Same input/output contract:

    payload = {"account": ..., "preset": ..., "trades": [...], "metrics": {...}}
    audit = run_audit(payload)
    # audit["finalDecision"], audit["emailBody"], audit["ruleByRuleAnalysis"], ...

How the port works:
  1. The agent is pre-created (one-time setup, see setup_agent.py) with a
     single custom tool `submit_audit_result`. The agent's SYSTEM_PROMPT
     forces every response to be exactly one call to that tool.
  2. Per audit, we create a fresh session, send the payload as a user
     message, and stream events.
  3. When the agent emits `agent.custom_tool_use` with name=submit_audit_result,
     we capture its input as the audit result.
  4. We send back a trivial `user.custom_tool_result` so the session can
     terminate cleanly, then archive the session.

Trade-offs vs the direct Messages API call in api.py:
  + Hosted: no FastAPI / Railway / uvicorn — Anthropic runs the loop.
  + Persisted, versioned agent — can roll back a prompt without redeploy.
  - Extra latency: session creation + event stream round-trip.
  - Extra cost: per-session container (counts against the 1,550 free
    container-hours/month, then ~$0.05/h). For high-volume single-shot
    audits this adds up.
  - Less control over `tool_choice` — can't *force* the tool call at the
    request level, you rely on the SYSTEM_PROMPT to be airtight.

Requires:
  pip install anthropic>=0.92.0
  ANTHROPIC_API_KEY env var
  AGENT_ID env var (from setup_agent.py)
  ENVIRONMENT_ID env var (from setup_agent.py)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import anthropic

log = logging.getLogger(__name__)

# Tool name MUST match agent.yaml.
_OUTPUT_TOOL_NAME = "submit_audit_result"

# Per-audit timeout. Managed Agents sessions don't have a hard wall-clock,
# so we enforce one client-side.
_AUDIT_TIMEOUT_SECONDS = float(os.environ.get("AUDIT_TIMEOUT_SECONDS", "180"))


class AuditError(RuntimeError):
    """Raised when the agent fails to produce a valid audit."""


@dataclass
class AuditCall:
    """One Managed Agents audit call — output + AI verification metadata."""

    audit: dict[str, Any]
    session_id: str
    agent_id: str
    agent_version: Optional[int]
    model: str
    latency_ms: int
    # Anthropic doesn't surface per-session token usage on the SSE stream the
    # way Messages does; pull it from sessions.retrieve() after the run.
    usage: dict[str, int]


def _get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise AuditError(f"Required env var {name} is not set")
    return val


def run_audit(
    payload: dict[str, Any],
    *,
    client: Optional[anthropic.Anthropic] = None,
    agent_id: Optional[str] = None,
    environment_id: Optional[str] = None,
    agent_version: Optional[int] = None,
) -> AuditCall:
    """Run one audit on the Managed Agents-hosted Risk Manager.

    `payload` shape matches the existing FastAPI request body:
        {"account": {...}, "preset": {...}, "trades": [...], "metrics": {...}}
    """
    client = client or anthropic.Anthropic()
    agent_id = agent_id or _get_env("AGENT_ID")
    environment_id = environment_id or _get_env("ENVIRONMENT_ID")

    # Compose the agent reference. String shorthand = latest version;
    # pin explicitly when reproducibility matters (e.g. an old reviewer
    # case file being re-rendered).
    agent_ref: Any = (
        {"type": "agent", "id": agent_id, "version": agent_version}
        if agent_version is not None
        else agent_id
    )

    started = time.time()

    # 1. Create the session. Blocks until the container is provisioned.
    session = client.beta.sessions.create(
        agent=agent_ref,
        environment_id=environment_id,
        title=f"Audit {payload.get('account', {}).get('accountNumber', 'unknown')}",
    )
    log.info(
        "AUDIT_SESSION_CREATED session_id=%s account=%s",
        session.id,
        payload.get("account", {}).get("accountNumber"),
    )

    try:
        audit = _drive_session(client, session.id, payload, started)
    finally:
        # Always archive — the session is one-shot, no reason to keep it.
        # archive is idempotent so this is safe even if the loop already
        # archived on error.
        try:
            client.beta.sessions.archive(session_id=session.id)
        except Exception:
            log.exception("Failed to archive session %s", session.id)

    latency_ms = int((time.time() - started) * 1000)

    # Pull usage after the fact — the stream doesn't expose it.
    try:
        session_full = client.beta.sessions.retrieve(session_id=session.id)
        usage = _extract_usage(session_full)
    except Exception:
        log.exception("Failed to retrieve usage for session %s", session.id)
        usage = {}

    return AuditCall(
        audit=audit,
        session_id=session.id,
        agent_id=agent_id,
        agent_version=agent_version,
        model="claude-opus-4-7",  # mirrors agent.yaml; could read from session_full.agent.model
        latency_ms=latency_ms,
        usage=usage,
    )


def _drive_session(
    client: anthropic.Anthropic,
    session_id: str,
    payload: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Open the SSE stream, send the payload, capture the tool call output."""
    user_text = (
        "Analyze the following payout review request and submit the "
        f"structured result by calling {_OUTPUT_TOOL_NAME}.\n\n"
        "```json\n" + json.dumps(payload, indent=2, default=str) + "\n```"
    )

    # Stream-FIRST, then send. Open the stream before posting the user
    # message — otherwise early events arrive buffered and we risk missing
    # the tool_use the agent emits within the first few hundred ms.
    audit_result: Optional[dict[str, Any]] = None
    custom_tool_use_id: Optional[str] = None

    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": user_text}],
                }
            ],
        )

        for event in stream:
            if (time.time() - started) > _AUDIT_TIMEOUT_SECONDS:
                raise AuditError(
                    f"Audit exceeded {_AUDIT_TIMEOUT_SECONDS}s wall-clock budget."
                )

            etype = getattr(event, "type", None)

            # The agent invoked our custom tool — this IS the audit output.
            if etype == "agent.custom_tool_use" and getattr(event, "name", None) == _OUTPUT_TOOL_NAME:
                audit_result = dict(event.input)  # type: ignore[arg-type]
                custom_tool_use_id = event.id
                log.info(
                    "AUDIT_TOOL_CAPTURED session_id=%s decision=%s severity=%s",
                    session_id,
                    audit_result.get("finalDecision"),
                    audit_result.get("severity"),
                )
                # Respond so the session can settle gracefully.
                client.beta.sessions.events.send(
                    session_id=session_id,
                    events=[
                        {
                            "type": "user.custom_tool_result",
                            "custom_tool_use_id": custom_tool_use_id,
                            "content": [
                                {"type": "text", "text": "Audit captured. Session complete."}
                            ],
                        }
                    ],
                )
                # We already have what we wanted — break out of the stream.
                break

            if etype == "session.status_terminated":
                raise AuditError("Session terminated before emitting audit.")

            if etype == "session.error":
                raise AuditError(
                    f"Session error: {getattr(event, 'error', None) or event}"
                )

    if audit_result is None:
        raise AuditError(
            "Agent finished without calling submit_audit_result. "
            "Check the SYSTEM_PROMPT — it must force the tool call."
        )

    return audit_result


def _extract_usage(session_full: Any) -> dict[str, int]:
    """Best-effort token usage extraction from the session resource."""
    usage = getattr(session_full, "usage", None)
    if usage is None:
        return {}
    # The exact shape varies by SDK version. Try common attribute names.
    out = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        val = getattr(usage, key, None)
        if isinstance(val, int):
            out[key] = val
    return out


# ──────────────────────────────────────────────────────────────────────────
# CLI entrypoint — `python audit_runner.py < payload.json > audit.json`
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = sys.stdin.read()
    payload = json.loads(raw)
    call = run_audit(payload)
    sys.stdout.write(
        json.dumps(
            {
                "audit": call.audit,
                "ai": {
                    "session_id": call.session_id,
                    "agent_id": call.agent_id,
                    "agent_version": call.agent_version,
                    "model": call.model,
                    "latency_ms": call.latency_ms,
                    "usage": call.usage,
                },
            },
            indent=2,
            default=str,
        )
    )
    sys.stdout.write("\n")
