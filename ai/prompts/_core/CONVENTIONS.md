# Core Conventions

- Keep prompts short and task-specific.
- Use JSON-only outputs when the caller expects machine parsing.
- Prefer deterministic rules before invoking a model.
- Do not expand context unless the task needs it.
- Preserve execution authority in code, not in prompt text.

## Current Runtime Facts

- Live bot launcher: [`run_live.ps1`](../../run_live.ps1)
- Demo bot launcher: [`run_demo.ps1`](../../run_demo.ps1)
- Live dashboard port: `8080`
- Demo dashboard port: `8766`
- Live `BOT_INSTANCE_PORT`: `45678`
- Demo `BOT_INSTANCE_PORT`: `45679`
- Live and demo must remain separate processes.
- The PowerShell launchers execute `main.py` directly; do not suggest a bare `py` shell that opens the REPL.
