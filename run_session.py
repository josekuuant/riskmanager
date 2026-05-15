"""Run one session against the Risk & Compliance Agent.

Setup (one-time, run via ant CLI — see README in chat):
  export AGENT_ID=$(ant beta:agents create < agent.yaml --transform id -r)
  export ENV_ID=$(ant beta:environments create < environment.yaml --transform id -r)

Runtime: this script loads AGENT_ID + ENV_ID from env, optionally uploads files,
opens an SSE stream, sends the kickoff message, and drains until idle/terminated.
"""

import os
import sys
from pathlib import Path

import anthropic

AGENT_ID = os.environ["AGENT_ID"]
ENV_ID = os.environ["ENV_ID"]

# Files to mount into the session container at /workspace/<name>.
# Add paths here (or wire this up to whatever drives a real review).
FILES_TO_MOUNT: list[Path] = [
    # Path("./inputs/control_matrix.xlsx"),
    # Path("./inputs/policy.pdf"),
]

KICKOFF_MESSAGE = "Hello — await further instructions."


def main() -> None:
    client = anthropic.Anthropic()

    # 1. Upload each file via the Files API; collect resource entries.
    resources = []
    for path in FILES_TO_MOUNT:
        with path.open("rb") as f:
            uploaded = client.beta.files.upload(file=f)
        resources.append(
            {
                "type": "file",
                "file_id": uploaded.id,
                "mount_path": f"/workspace/{path.name}",
            }
        )

    # 2. Create the session (references the pre-created agent by ID).
    session = client.beta.sessions.create(
        agent=AGENT_ID,
        environment_id=ENV_ID,
        title="Compliance review run",
        resources=resources,
    )
    print(f"Session: {session.id}")
    print(
        f"Watch in Console: https://platform.claude.com/workspaces/default/sessions/{session.id}",
        flush=True,
    )

    # 3. Stream-first: open the stream BEFORE sending the kickoff so we don't
    #    miss early events. Then send the user message while the stream is live.
    with client.beta.sessions.events.stream(session_id=session.id) as stream:
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": KICKOFF_MESSAGE}],
                }
            ],
        )

        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        sys.stdout.write(block.text)
                        sys.stdout.flush()
            elif event.type == "agent.tool_use":
                print(f"\n[tool] {event.name}", flush=True)
            elif event.type == "session.status_terminated":
                print("\n[session terminated]", flush=True)
                break
            elif event.type == "session.status_idle":
                # Break only on terminal stop_reason. requires_action means the
                # agent is waiting on a tool confirmation / custom tool result.
                if event.stop_reason.type == "requires_action":
                    continue
                print(
                    f"\n[idle: {event.stop_reason.type}]",
                    flush=True,
                )
                break

    # 4. Optionally pull files the agent wrote to /mnt/session/outputs/.
    outputs_dir = Path("./outputs")
    outputs_dir.mkdir(exist_ok=True)
    for f in client.beta.files.list(
        scope_id=session.id,
        betas=["managed-agents-2026-04-01"],
    ):
        print(f"Output: {f.filename} ({f.size_bytes} bytes)")
        content = client.beta.files.download(f.id)
        content.write_to_file(outputs_dir / f.filename)


if __name__ == "__main__":
    main()
