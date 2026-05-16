# NYS Risk Manager — Managed Agents port

Port of `riskmanager/api.py` from a FastAPI single-shot Messages-API call to
Anthropic **Managed Agents** (`/v1/agents` + `/v1/sessions`). Same input/output
contract; same SYSTEM_PROMPT; same tool schema.

## Is this the right tool for the job?

Probably not — managed agents are optimized for **stateful** work (container
workspaces, multi-turn loops, file mounts). The Risk Manager is a one-shot
"payload in → audit JSON out" call. The existing FastAPI version is simpler,
faster, and cheaper.

Use this port when you want:
- Hosted on Anthropic infra (no Railway / FastAPI / uvicorn).
- Persisted, versioned agent config — `agents.update()` creates a new version
  without redeploying anything, sessions can pin to specific versions for
  reproducibility.
- The audit JSON addressable by a `session_id` for compliance trail.

Skip it if:
- You're cost-sensitive at high audit volume — every audit provisions a
  container (counts against your free hours, then ~$0.05/h).
- You need < 5 s latency — session creation adds 1–3 s before the model
  even starts working.

## Layout

```
agent_sdk/
├── agent.yaml             ← Agent definition (YAML, for `ant beta:agents create`)
├── environment.yaml       ← Environment definition (YAML)
├── build_agent_yaml.py    ← Splices the live SYSTEM_PROMPT from api.py into agent.yaml
├── setup_agent.py         ← One-time agent + environment creation (Python alternative to `ant`)
├── audit_runner.py        ← run_audit(payload) → AuditCall — the actual hot path
└── README.md              ← This file
```

## Setup (one-time)

### Option A — Anthropic CLI (recommended; YAML in git)

```bash
# 1. Splice SYSTEM_PROMPT from api.py into a build artifact
python build_agent_yaml.py > agent.built.yaml

# 2. Create the agent and environment
AGENT_ID=$(ant beta:agents create < agent.built.yaml --transform id -r)
ENV_ID=$(ant beta:environments create < environment.yaml --transform id -r)

echo "AGENT_ID=$AGENT_ID"
echo "ENVIRONMENT_ID=$ENV_ID"
```

Store both IDs in your secret manager / `.env`.

### Option B — Python (one-shot script)

```bash
ANTHROPIC_API_KEY=sk-ant-... python setup_agent.py
```

Prints `AGENT_ID=…` and `ENVIRONMENT_ID=…` to stdout.

## Run an audit

```python
import os, json
from agent_sdk.audit_runner import run_audit

os.environ["AGENT_ID"] = "agent_..."
os.environ["ENVIRONMENT_ID"] = "env_..."

payload = {
    "account": {"accountNumber": "12345", "accountType": "INSTANT", "phase": "LIVE",
                "accountSize": 100000, "initialBalance": 100000,
                "serverTimezone": "UTC", "dailyResetTime": "21:00"},
    "preset": {...},     # from getPreset()
    "trades": [...],     # parsed trades
    "metrics": {...},    # deterministic metrics
}

call = run_audit(payload)
print(call.audit["finalDecision"])           # PASSED | WARNING | BREACH | MANUAL_REVIEW
print(call.audit["internalRecommendation"])  # APPROVE PAYOUT | ...
print(call.audit["emailBody"])
print(f"latency={call.latency_ms}ms session={call.session_id}")
```

Or from the shell:

```bash
AGENT_ID=... ENVIRONMENT_ID=... python audit_runner.py < payload.json > audit.json
```

## How the port maps onto Managed Agents

| FastAPI version (api.py)                                | Managed Agents port                                          |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| `client.messages.create(tools=[...], tool_choice=...)`  | Custom tool on the agent + system-prompt-forced invocation   |
| Tool schema in `SUBMIT_AUDIT_TOOL`                      | `tools: [{type: "custom", name: "submit_audit_result", ...}]`|
| `tool_block.input` from response                        | `agent.custom_tool_use` event on the session SSE stream      |
| Per-request retry / 429 / 502 handling                  | Implicit — Anthropic's orchestration layer handles it        |
| Per-request `_estimate_cost_usd`                        | Pull tokens from `sessions.retrieve()` after the run         |
| `mock_ai` / `ai_status: "failed"`                       | Same shape — wrap `run_audit()` in your own try/except       |
| Self-hosted FastAPI on Railway                          | Hosted on Anthropic — no infra to operate                    |

## Why the SYSTEM_PROMPT must be airtight

In the Messages API version we force the tool call with `tool_choice={"type": "tool", "name": "submit_audit_result"}`. Managed Agents has no equivalent
"force this tool" lever — the agent decides when to call tools based on the
SYSTEM_PROMPT and user messages.

The existing prompt already opens with:

> You respond ONLY by calling the submit_audit_result tool with valid
> structured output. No prose outside the tool call.

That's enough for Opus 4.7 in practice. If you ever see the agent reply with
prose instead of calling the tool, `run_audit()` raises `AuditError`. The
fix is to strengthen the prompt, not to retry.

## Updating the agent

Don't recreate — update. New version, same ID, sessions in flight keep
their pinned version:

```bash
# CLI
python build_agent_yaml.py > agent.built.yaml
ant beta:agents update --agent-id "$AGENT_ID" --version <current_version> < agent.built.yaml

# or Python
python -c "
import anthropic
c = anthropic.Anthropic()
c.beta.agents.update(agent_id='$AGENT_ID', version=<current>, system=open('new_prompt.txt').read())
"
```

Pin sessions to a specific version when reproducibility matters:

```python
run_audit(payload, agent_version=1772585501101368014)
```
