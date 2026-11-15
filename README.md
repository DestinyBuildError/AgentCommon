# AgentCommon

English | [简体中文](README.zh-CN.md)

A general-purpose LLM agent backend with tool calling (Tools / MCP / Skills),
streaming interaction (SSE), approval gates, and a self-extending tool
ecosystem — built with Python 3.12 + FastAPI + SQLAlchemy (async).

> Companion H5 frontend lives in a separate repository; this backend exposes
> everything a chat UI needs over REST + SSE.

## Features

- **Agent loop engine** — iterative tool-use loop with per-tool circuit
  breakers, iteration limits, and orphan-session repair on startup.
- **Streaming protocol** — Server-Sent Events with monotonic `seq` numbers,
  automatic replay (`after_seq` / `Last-Event-ID`) so clients never miss or
  duplicate events across reconnects.
- **Tool system** — built-in tools, MCP servers (stdio + streamable HTTP),
  and Markdown-defined Skills with progressive loading.
- **Self-extending agent** — meta tools let the model *discover* tools from a
  curated catalog, *install/uninstall* MCP servers, and fall back to a
  sandboxed Python runner — all behind a human approval gate.
- **Human-in-the-loop** — dangerous tool calls emit `approval_request` events
  and wait for an explicit allow/deny decision (with auto-deny timeout);
  planner-proposed execution plans surface as user-selectable options.
- **Permission modes** — global `read` / `write` / `full` modes gate tool
  execution, persisted across restarts.
- **Pluggable storage** — async SQLAlchemy: SQLite out of the box, PostgreSQL
  for production (one config line).
- **Token & cost accounting** — per-session input/output/cache tokens and
  cost tracking.

## Use Cases

| Scenario | What the backend provides |
|---|---|
| AI chat assistants | Session management, streaming text/thinking deltas, history replay |
| Coding / file agents | Workspace-scoped `read_file` / `write_file` / `search_workspace`, multi-root workspaces |
| Enterprise tool integration | Plug any MCP-compatible server (internal APIs, databases, SaaS) via stdio or HTTP |
| Plugin marketplaces | Curated `tool_catalog.yaml` + `discover_tools` / `install_mcp_server` flow |
| Regulated environments | Approval gates, permission modes, full audit trail of tool calls in DB |
| Offline development & CI | `dev_mock_server.py` runs the full stack with a scripted LLM — no API key needed |
| Multi-tenant products | Per-session isolation, idle-timeout recycling, interrupt support |

## Architecture

```
app/
  main.py            FastAPI entrypoint, lifespan wiring, unified error format
  api/               REST endpoints + SSE event stream (9 resources)
  core/
    agent_loop.py    iterative tool-use loop
    llm_client.py    Anthropic SDK wrapper (retries, streaming)
    dispatcher.py    tool dispatch, circuit breaker, permission gating
    planner.py       task planner -> task_options events
    session_manager.py  session lifecycle, idle recycling, interrupt
    event_bus.py     per-session bus with seq numbering + replay buffer
    context.py       context compaction
    pricing.py       token cost accounting
    workspace_registry.py  multi-root workspace directories
    repair.py        orphan tool_use repair
  db/                async models + repository (SQLite -> PostgreSQL)
  schemas/           Pydantic request/response + SSE event schemas
  tools/             registry, built-in tools, meta tools, tool catalog
  mcp_loader.py      MCP integration (namespacing, lazy reconnect, degradation)
  skills_loader.py   SKILL.md progressive loading
skills/              sample skills (code_review, data_analysis)
catalog_servers/     3 local demo MCP servers (text_tools, unit_convert, password_gen)
demo_mcp_server.py   bundled demo MCP server (current_time / echo_text)
dev_mock_server.py   scripted-LLM server for keyless demo / integration testing
contract_capture/    captured API contract fixtures (regression reference)
tests/               287 pytest cases
```

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=...
```

No API key? Run the mock server instead — it drives the whole pipeline
(planner, tool calls, SSE frames) with a scripted LLM:

```bash
python dev_mock_server.py --port 8000
```

### 3. Run

```bash
python -m uvicorn app.main:app --port 8000
# or background mode on Windows:
LaunchBackground.bat
```

Check `GET /health`, then create a session and chat (see API below).
The bundled demo MCP server answers "what time is it" via
`mcp__demo__current_time` — a full MCP round-trip out of the box.

### 4. Test

```bash
python -m pytest            # 287 tests, no API key required
```

## Configuration

Copy `config.yaml.example` to `config.yaml` and edit. Highlights:

| Key | Meaning |
|---|---|
| `model` / `effort` / `max_tokens` | LLM selection and cost/quality knob |
| `tool_breaker_limit` | consecutive-failure circuit breaker per tool |
| `max_iterations` | agent-loop iteration cap per turn |
| `approval_timeout_s` | auto-deny timeout for pending approvals |
| `planner_enabled` / `planner_*` | task planner (plan-first workflow) |
| `database_url` | SQLite or PostgreSQL connection string |
| `workspace_root` | seed workspace directory (multi-root at runtime) |
| `mcp_servers` | MCP servers to attach at startup |
| `enable_tool_discovery` | enable the four self-extension meta tools |

Secrets come from environment variables only (`ANTHROPIC_API_KEY`);
env vars always override `config.yaml`.

## REST API

Base URL: `/api`. Errors are always
`{"ok": false, "error": {"code": "...", "message": "..."}}` (400/404/405/409/422/500).

### Sessions

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions` | Create session (201). Body: `{"title": "...", "model": "...", "effort": "high"}` (all optional) |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/{session_id}` | Session detail incl. token usage & cost |
| `DELETE` | `/api/sessions/{session_id}` | Delete session (204) |
| `POST` | `/api/sessions/{session_id}/interrupt` | Interrupt a running turn |

Session object: `id`, `user_id`, `title`, `status` (`idle`/`running`/`waiting_approval`/`waiting_choice`), `model`, `effort`, `total_input_tokens`, `total_output_tokens`, `cache_read_tokens`, `total_cost_usd`, `created_at`, `updated_at`.

### Messages

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/messages` | Queue a user message (202). Body: `{"content": "..."}` → `{"ok": true, "session_id": "...", "queued": true}` |
| `GET` | `/api/sessions/{session_id}/messages?after_seq=N` | History / incremental fetch. `after_seq` returns only messages past the cursor |

### Streaming (SSE)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/sessions/{session_id}/events` | Event stream. Send `after_seq` query param or `Last-Event-ID` header to replay missed events |

Every frame is `{"seq": <int>, "type": "<event>", "data": {...}}`. Event types:

| Event | Meaning |
|---|---|
| `text_delta` | Streaming assistant text chunk |
| `thinking_delta` | Streaming reasoning chunk |
| `tool_call_start` | Tool invocation started (name + input preview) |
| `tool_call_result` | Tool invocation finished (result / error) |
| `approval_request` | Dangerous tool call needs a human decision |
| `task_options` | Planner proposes candidate execution plans |
| `task_progress` | Long-running task progress push |
| `workspace_changed` | Workspace directory set changed (global broadcast) |
| `tools_changed` | MCP toolset changed after install/uninstall |
| `turn_complete` | Turn finished |
| `error` | Turn aborted with an error |

Reconnect semantics: the `seq` cursor is monotonic per session; on reconnect
the bus replays everything after `after_seq` from its buffer (and falls back
to DB-derived history), so clients dedupe by `seq` and never lose events.

### Approvals & Choices

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/approvals/{tool_use_id}` | Resolve approval. Body: `{"decision": "allow"}` or `{"decision": "deny"}`. Unresolved requests auto-deny after `approval_timeout_s` |
| `POST` | `/api/sessions/{session_id}/choices/{choice_id}` | Resolve planner choice. Body: `{"option_index": 0}`. Timeout auto-selects the recommended option |

### Permissions

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/permissions` | Current global permission mode |
| `PUT` | `/api/permissions` | Set mode. Body: `{"mode": "read"}` / `"write"` / `"full"`. Persisted in DB |

### Workspaces

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/workspaces` | List workspace root directories |
| `POST` | `/api/workspaces` | Add a directory (201). Body: `{"path": "..."}` |
| `DELETE` | `/api/workspaces/{dir_id}` | Remove a directory (204) |

Changes broadcast `workspace_changed` to every live session immediately;
agent-side tools (`add_workspace_dir` / `remove_workspace_dir`) hit the same
registry and are blocked when `agent_manage_dirs: false`.

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "version": "...", "model": "...", "effort": "..."}` |

## Tool System

- **Built-in tools** — workspace-scoped file operations (`read_file`,
  `write_file`, `search_workspace`, ...), workspace management, `run_python`
  (sandboxed, hard-capped timeout).
- **MCP** — configure servers in `config.yaml` (stdio subprocess or
  streamable HTTP). Tools register under `mcp__<server>__<tool>`; a broken
  server degrades to a warning without blocking startup.
- **Skills** — a `skills/<name>/SKILL.md` file with progressive loading: the
  system prompt carries a one-line description per skill; the model loads the
  full body on demand via the `load_skill` tool.
- **Meta tools (self-extension)** — `discover_tools` (search the curated
  catalog), `install_mcp_server` / `uninstall_mcp_server` (live install,
  broadcasts `tools_changed`), `run_python`. The mutating ones are gated by
  the approval flow.

## License

[MIT](LICENSE)
