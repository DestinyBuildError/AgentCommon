# AgentCommon

[English](README.md) | 简体中文

一个通用型 LLM Agent 后端：带工具调用（Tools / MCP / Skills）、流式交互（SSE）、人工审批门、可自扩展的工具生态。基于 Python 3.12 + FastAPI + SQLAlchemy (async)。

> 配套的 H5 前端在独立仓库中维护；本仓库的后端通过 REST + SSE 为聊天界面提供全部能力。

## 特性

- **Agent 循环引擎** —— 迭代式工具调用循环，内置按工具熔断、迭代上限、启动时孤儿会话修复。
- **流式协议** —— SSE 事件带单调递增 `seq` 编号，支持 `after_seq` / `Last-Event-ID` 自动重放，客户端断线重连不丢事件、不重复。
- **工具体系** —— 内置工具 + MCP server（stdio / streamable HTTP）+ Markdown 定义的 Skills（渐进加载）。
- **自扩展 Agent** —— 元工具让模型*发现*精选工具目录中的工具、*安装/卸载* MCP server，并可降级使用沙箱化 Python 执行器——全部经过人工审批门。
- **人在回路** —— 危险工具调用发出 `approval_request` 事件等待显式 allow/deny 决议（超时自动 deny）；规划器产出的执行方案以选项形式交由用户选择。
- **权限模式** —— 全局 `read` / `write` / `full` 三档权限门控工具执行，持久化存储、重启不丢。
- **存储可插拔** —— 异步 SQLAlchemy：开箱 SQLite，改一行配置切 PostgreSQL。
- **Token 与成本核算** —— 按会话统计输入/输出/缓存 token 与成本。

## 应用场景

| 场景 | 后端提供的能力 |
|---|---|
| AI 聊天助手 | 会话管理、text/thinking 流式增量、历史重放 |
| 编程 / 文件代理 | 工作区范围内的 `read_file` / `write_file` / `search_workspace`，多根目录工作区 |
| 企业工具集成 | 通过 stdio 或 HTTP 接入任意 MCP 兼容服务（内部 API、数据库、SaaS） |
| 插件市场 | 精选 `tool_catalog.yaml` + `discover_tools` / `install_mcp_server` 安装链路 |
| 合规敏感环境 | 审批门、权限模式、DB 中完整的工具调用审计链 |
| 离线开发与 CI | `dev_mock_server.py` 以脚本化 LLM 跑通全栈——无需 API key |
| 多租户产品 | 会话级隔离、空闲超时回收、打断支持 |

## 架构

```
app/
  main.py            FastAPI 入口、lifespan 装配、统一错误格式
  api/               REST 端点 + SSE 事件流
  core/
    agent_loop.py    迭代式工具调用循环
    llm_client.py    Anthropic SDK 封装（重试、流式）
    dispatcher.py    工具分发、熔断器、权限门控
    planner.py       任务规划器 → task_options 事件
    session_manager.py  会话生命周期、空闲回收、打断
    event_bus.py     会话级事件总线（seq 编号 + 重放缓冲）
    context.py       上下文压缩
    pricing.py       token 成本核算
    workspace_registry.py  多根目录工作区
    repair.py        孤儿 tool_use 修复
  db/                异步模型 + 仓储（SQLite → PostgreSQL）
  schemas/           Pydantic 请求/响应 + SSE 事件模式
  tools/             注册表、内置工具、元工具、工具目录
  mcp_loader.py      MCP 接入（命名空间、惰性重连、失败降级）
  skills_loader.py   SKILL.md 渐进加载
skills/              示例 Skills（code_review / data_analysis）
catalog_servers/     3 个本地演示 MCP server（text_tools / unit_convert / password_gen）
demo_mcp_server.py   内置演示 MCP server（current_time / echo_text）
dev_mock_server.py   脚本化 LLM 服务器（无 key 演示 / 联调）
contract_capture/    捕获的 API 契约样例（回归参照）
tests/               287 个 pytest 用例
```

## 快速开始

### 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.yaml.example config.yaml
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=...
```

没有 API key？直接跑 mock 服务器——用脚本化 LLM 驱动全链路（规划器、工具调用、SSE 帧）：

```bash
python dev_mock_server.py --port 8000
```

### 3. 运行

```bash
python -m uvicorn app.main:app --port 8000
# Windows 后台模式：
LaunchBackground.bat
```

`GET /health` 检查就绪后即可建会话开聊（接口见下文）。内置演示 MCP server 让「现在几点」走通 `mcp__demo__current_time`——开箱即可看到 MCP 全链路。

### 4. 测试

```bash
python -m pytest            # 287 个用例，无需 API key
```

## 配置

复制 `config.yaml.example` 为 `config.yaml` 后按需修改。重点项：

| 配置项 | 说明 |
|---|---|
| `model` / `effort` / `max_tokens` | 模型选择与成本/质量档位 |
| `tool_breaker_limit` | 同一工具连续失败 N 次后熔断 |
| `max_iterations` | 单轮 agent loop 最大迭代次数 |
| `approval_timeout_s` | 待审批超时自动 deny |
| `planner_enabled` / `planner_*` | 任务规划器（先出方案再执行的工作流） |
| `database_url` | SQLite 或 PostgreSQL 连接串 |
| `workspace_root` | 工作区种子目录（运行期可多目录管理） |
| `mcp_servers` | 启动时接入的 MCP server |
| `enable_tool_discovery` | 四个自扩展元工具的开关 |

密钥只走环境变量（`ANTHROPIC_API_KEY`）；环境变量优先级高于 `config.yaml`。

## REST API

Base URL：`/api`。错误响应统一为
`{"ok": false, "error": {"code": "...", "message": "..."}}`（400/404/405/409/422/500）。

### 会话

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/sessions` | 创建会话（201）。Body：`{"title": "...", "model": "...", "effort": "high"}`（均可选） |
| `GET` | `/api/sessions` | 会话列表 |
| `GET` | `/api/sessions/{session_id}` | 会话详情，含 token 用量与成本 |
| `DELETE` | `/api/sessions/{session_id}` | 删除会话（204） |
| `POST` | `/api/sessions/{session_id}/interrupt` | 打断进行中的一轮对话 |

会话对象：`id`、`user_id`、`title`、`status`（`idle`/`running`/`waiting_approval`/`waiting_choice`）、`model`、`effort`、`total_input_tokens`、`total_output_tokens`、`cache_read_tokens`、`total_cost_usd`、`created_at`、`updated_at`。

### 消息

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/messages` | 投递用户消息（202）。Body：`{"content": "..."}` → `{"ok": true, "session_id": "...", "queued": true}` |
| `GET` | `/api/sessions/{session_id}/messages?after_seq=N` | 历史 / 增量拉取。`after_seq` 游标之后的消息 |

### 流式（SSE）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/sessions/{session_id}/events` | 事件流。携带 `after_seq` 参数或 `Last-Event-ID` 头可重放错过的帧 |

每帧格式：`{"seq": <int>, "type": "<event>", "data": {...}}`。事件类型：

| 事件 | 含义 |
|---|---|
| `text_delta` | 助手文本流式增量 |
| `thinking_delta` | 推理过程流式增量 |
| `tool_call_start` | 工具调用开始（名称 + 输入预览） |
| `tool_call_result` | 工具调用结束（结果 / 错误） |
| `approval_request` | 危险工具调用等待人工决议 |
| `task_options` | 规划器给出候选执行方案 |
| `task_progress` | 长时任务进度推送 |
| `workspace_changed` | 工作区目录集合变更（全局广播） |
| `tools_changed` | 安装/卸载后的工具集变更 |
| `turn_complete` | 一轮对话结束 |
| `error` | 本轮异常中止 |

重连语义：`seq` 游标按会话单调递增；重连时总线从 `after_seq` 起重放缓冲中的全部事件（缓冲不足时回退 DB 推导历史），客户端按 `seq` 去重即可做到不丢不重。

### 审批与选择

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/approvals/{tool_use_id}` | 审批决议。Body：`{"decision": "allow"}` 或 `{"decision": "deny"}`。超时未决议自动 deny |
| `POST` | `/api/sessions/{session_id}/choices/{choice_id}` | 规划器选项决议。Body：`{"option_index": 0}`。超时自动采用推荐项 |

### 权限

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/permissions` | 当前全局权限模式 |
| `PUT` | `/api/permissions` | 设置模式。Body：`{"mode": "read"}` / `"write"` / `"full"`。持久化到 DB |

### 工作区

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/workspaces` | 列出工作区根目录 |
| `POST` | `/api/workspaces` | 添加目录（201）。Body：`{"path": "..."}` |
| `DELETE` | `/api/workspaces/{dir_id}` | 移除目录（204） |

目录变更立即向所有活跃会话广播 `workspace_changed`；agent 侧工具（`add_workspace_dir` / `remove_workspace_dir`）走同一注册表，`agent_manage_dirs: false` 时被禁用。

### 健康检查

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "version": "...", "model": "...", "effort": "..."}` |

## 工具体系

- **内置工具** —— 工作区范围内的文件操作（`read_file`、`write_file`、`search_workspace` 等）、工作区管理、`run_python`（沙箱化，超时硬上限）。
- **MCP** —— 在 `config.yaml` 中配置（stdio 子进程或 streamable HTTP）。工具以 `mcp__<server>__<tool>` 前缀注册；单个 server 启动失败仅降级告警，不阻塞整体。
- **Skills** —— `skills/<name>/SKILL.md` 渐进加载：system prompt 只带一行描述；模型按需通过 `load_skill` 工具加载完整内容。
- **元工具（自扩展）** —— `discover_tools`（检索精选目录）、`install_mcp_server` / `uninstall_mcp_server`（运行期安装，广播 `tools_changed`）、`run_python`。变更类操作全部过审批门。

## 许可证

[MIT](LICENSE)
