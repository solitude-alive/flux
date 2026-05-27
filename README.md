# flux

Personal Python library bundling reusable building blocks.

## Modules

- **`my_tool.logger`** — production-grade logger for 24h online services.
  Built on top of the stdlib `logging` module with day + size based file
  rotation, thread-safety, KV metric aggregation, optional JSON output,
  and ANSI color for TTY stdout. See [`examples/logger_demo.py`](examples/logger_demo.py).
- **`my_tool.googlevertex`** — thin client wrapper around the Google
  Gemini / Vertex AI SDK (`google-genai`).

## Install

This project uses [`uv`](https://github.com/astral-sh/uv) and the
[hatchling](https://hatch.pypa.io/) build backend.

```bash
uv sync
```

## Run the logger demo

```bash
uv run python examples/logger_demo.py
```

The script writes to `./log/<YYYY-MM-DD>/` with a small `max_bytes` so you
can see the day + size rotation in action.

## Development

Lint and format are wired up via pre-commit (ruff + ruff-format + hygiene
hooks). Install once:

```bash
pre-commit install
```
