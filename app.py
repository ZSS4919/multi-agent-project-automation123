"""
Multi-Agent Software Project Automation System
多 Agent 协同软件项目自动化系统

隐私说明：
- 本代码不包含任何个人信息。
- 本代码不包含真实 API Key。
- 默认 Mock 模式，不会访问外部接口。
- 如需接入真实模型，请在本地创建 .env 文件，且不要上传到 GitHub。

运行方式：
    pip install fastapi uvicorn pydantic requests python-dotenv
    python app.py

浏览器打开：
    http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

APP_NAME = "Multi-Agent Software Project Automation System"
DB_PATH = Path("multi_agent_project_history.db")


class ProjectRequest(BaseModel):
    project_name: str = Field(..., description="Project name")
    project_goal: str = Field(..., description="Project goal")
    target_users: str = Field(
        default="Product managers, developers, project managers, startup teams",
        description="Target users",
    )
    core_features: str = Field(..., description="Core features")
    tech_preference: str = Field(
        default="Python FastAPI + SQLite + HTML + JavaScript",
        description="Preferred tech stack",
    )
    scale_requirement: str = Field(
        default="Support large-scale projects, multi-agent workflows, model routing, and task history",
        description="Scale requirement",
    )
    extra_requirement: str = Field(
        default="Generate a complete project plan suitable for GitHub demonstration",
        description="Extra requirement",
    )


@dataclass
class AgentResult:
    agent_name: str
    role: str
    output: str
    duration_ms: int


@dataclass
class WorkflowResult:
    workflow_id: str
    created_at: str
    project_name: str
    agent_results: List[AgentResult]
    final_report: str


class LLMProvider:
    """
    OpenAI-Compatible LLM Provider.

    Safe default:
    - If no environment variables are configured, the system uses local Mock mode.
    - No API key is included in this file.

    Optional local .env example:
        LLM_BASE_URL=https://api.openai.com/v1
        LLM_API_KEY=your_api_key_here
        LLM_MODEL=gpt-4o-mini
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "mock-local-agent")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    @property
    def is_mock(self) -> bool:
        return not self.base_url or not self.api_key

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.is_mock:
            return self._mock_generate(system_prompt, user_prompt)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            return (
                "LLM API call failed. The system has fallen back to local Mock output.\n"
                f"Error: {exc}\n\n"
                + self._mock_generate(system_prompt, user_prompt)
            )

    def _mock_generate(self, system_prompt: str, user_prompt: str) -> str:
        text = system_prompt.lower()

        if "planner" in text:
            return """
## Requirement Breakdown

1. Clarify the project goal, target users, and business boundary.
2. Split the system into web interface, task management, agent orchestration, model access, storage, and result display.
3. Convert the requirement into executable tasks for multiple agents.
4. Start with an MVP workflow, then expand to authentication, queues, knowledge base, and model routing.

## Task List

- Requirement analysis
- System architecture design
- API design
- Database design
- Frontend interaction design
- Test plan generation
- Documentation generation
- Risk and quality review
""".strip()

        if "architect" in text:
            return """
## System Architecture

The system uses a layered architecture:

- Web Layer: FastAPI and a simple HTML interface.
- Orchestrator Layer: Coordinates multiple agents and manages context.
- Agent Layer: Planner, Architect, Backend, Frontend, Test, Documentation, and QA agents.
- Model Layer: Supports OpenAI-compatible APIs.
- Storage Layer: SQLite stores workflow history and agent outputs.

## Scalability

- Add Redis and Celery for asynchronous tasks.
- Add vector database support for knowledge retrieval.
- Add user authentication for team usage.
- Add model routing for different task types.
""".strip()

        if "backend" in text:
            return """
## Backend Design

Core APIs:

- GET /health: Health check.
- POST /api/run: Submit project requirement and start the multi-agent workflow.
- GET /api/history: List workflow history.
- GET /api/history/{workflow_id}: View workflow detail.

Database tables:

- workflows: Stores workflow ID, created time, project name, and final report.
- agent_outputs: Stores each agent's name, role, output, and execution time.
""".strip()

        if "frontend" in text:
            return """
## Frontend Design

The page includes:

1. Project name input.
2. Project goal textarea.
3. Target users input.
4. Core features textarea.
5. Tech preference input.
6. Scale requirement textarea.
7. Generate button.
8. Agent output cards.
9. Final report display area.

Interaction flow:

- User enters project information.
- Frontend calls POST /api/run.
- Backend returns all agent outputs.
- Page displays each agent result and final report.
""".strip()

        if "test" in text:
            return """
## Test Plan

Functional tests:

- Submit a complete project request and verify that all agent outputs are returned.
- Submit incomplete input and verify validation errors.
- Run without API key and verify Mock mode.
- Run with API configuration and verify model API call.

API tests:

- GET /health should return system status.
- POST /api/run should return workflow_id, agent_results, and final_report.
- GET /api/history should return workflow list.

Exception tests:

- Model API timeout.
- Invalid API key.
- Database write failure.
""".strip()

        if "doc" in text:
            return """
## Documentation Draft

# Multi-Agent Software Project Automation System

This project demonstrates how multiple AI agents can collaborate to complete software project planning and automation tasks.

## Highlights

- Multi-agent collaboration workflow
- Local FastAPI web application
- SQLite workflow history
- Mock mode without API key
- OpenAI-compatible API support

## Use Cases

- Software project requirement analysis
- Large-scale project task breakdown
- AI coding assistant workflow demonstration
- Local deployment practice
- Multi-model evaluation
""".strip()

        if "qa" in text:
            return """
## QA Review

Strengths:

- Complete workflow from input to final report.
- Demonstrates multi-agent collaboration clearly.
- Can run locally without API key.
- Can be connected to real models through environment variables.
- No personal information is included in the source code.

Risks and improvements:

- Current workflow is synchronous. For long tasks, add a task queue.
- Current storage is SQLite. For production, use PostgreSQL.
- Add authentication before team usage.
- Add token usage tracking when using real models.
""".strip()

        return "Local Mock Agent output completed."


class BaseAgent:
    def __init__(self, name: str, role: str, system_prompt: str, llm: LLMProvider) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm

    def run(self, context: Dict[str, Any]) -> AgentResult:
        start = time.time()
        user_prompt = json.dumps(context, ensure_ascii=False, indent=2)
        output = self.llm.generate(self.system_prompt, user_prompt)
        duration_ms = int((time.time() - start) * 1000)

        return AgentResult(
            agent_name=self.name,
            role=self.role,
            output=output,
            duration_ms=duration_ms,
        )


class PlannerAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "PlannerAgent",
            "Requirement Planning Agent",
            "You are Planner Agent. You break down software project requirements into executable tasks.",
            llm,
        )


class ArchitectAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "ArchitectAgent",
            "System Architecture Agent",
            "You are Architect Agent. You design system architecture, modules, and scalability plans.",
            llm,
        )


class BackendAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "BackendAgent",
            "Backend Design Agent",
            "You are Backend Agent. You design APIs, database tables, and backend logic.",
            llm,
        )


class FrontendAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "FrontendAgent",
            "Frontend Design Agent",
            "You are Frontend Agent. You design UI structure, user flow, and interaction logic.",
            llm,
        )


class TestAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "TestAgent",
            "Test Planning Agent",
            "You are Test Agent. You create test plans, API tests, exception tests, and acceptance criteria.",
            llm,
        )


class DocAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "DocAgent",
            "Documentation Agent",
            "You are Documentation Agent. You generate README, deployment guide, and project explanation.",
            llm,
        )


class QAAgent(BaseAgent):
    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(
            "QAAgent",
            "Quality Review Agent",
            "You are QA Agent. You check risks, missing parts, privacy issues, security issues, and quality problems.",
            llm,
        )


class MultiAgentOrchestrator:
    def __init__(self) -> None:
        self.llm = LLMProvider()
        self.agents = [
            PlannerAgent(self.llm),
            ArchitectAgent(self.llm),
            BackendAgent(self.llm),
            FrontendAgent(self.llm),
            TestAgent(self.llm),
            DocAgent(self.llm),
            QAAgent(self.llm),
        ]

    def run(self, request: ProjectRequest) -> WorkflowResult:
        workflow_id = str(uuid.uuid4())
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        context: Dict[str, Any] = {
            "project_request": request.model_dump(),
            "previous_agent_outputs": [],
        }

        agent_results: List[AgentResult] = []

        for agent in self.agents:
            result = agent.run(context)
            agent_results.append(result)
            context["previous_agent_outputs"].append(asdict(result))

        final_report = self._build_final_report(request, agent_results)

        workflow = WorkflowResult(
            workflow_id=workflow_id,
            created_at=created_at,
            project_name=request.project_name,
            agent_results=agent_results,
            final_report=final_report,
        )

        save_workflow(workflow)
        return workflow

    @staticmethod
    def _build_final_report(request: ProjectRequest, results: List[AgentResult]) -> str:
        parts = [
            f"# {request.project_name} - Multi-Agent Project Report",
            "",
            "## Project Goal",
            request.project_goal,
            "",
            "## Target Users",
            request.target_users,
            "",
            "## Core Features",
            request.core_features,
            "",
            "## Tech Preference",
            request.tech_preference,
            "",
            "## Agent Output Summary",
        ]

        for item in results:
            parts.extend(
                [
                    "",
                    f"### {item.agent_name} - {item.role}",
                    item.output,
                ]
            )

        parts.extend(
            [
                "",
                "## Project Highlights",
                "- Demonstrates multi-agent collaboration for complex software project planning.",
                "- Supports both local Mock mode and real model API mode.",
                "- Compatible with OpenAI-style model APIs and local model gateways.",
                "- Suitable for local deployment practice and large-scale project automation.",
                "",
                "## Privacy and Security Notice",
                "- No personal email, phone number, address, account, or real API key is included.",
                "- API keys should only be stored in a local .env file.",
                "- The .env file should never be uploaded to GitHub.",
                "- Default Mock mode does not access external services.",
            ]
        )

        return "\n".join(parts)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                project_name TEXT NOT NULL,
                final_report TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                role TEXT NOT NULL,
                output TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
            )
            """
        )
        conn.commit()


def save_workflow(workflow: WorkflowResult) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO workflows (
                workflow_id,
                created_at,
                project_name,
                final_report
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                workflow.workflow_id,
                workflow.created_at,
                workflow.project_name,
                workflow.final_report,
            ),
        )

        for result in workflow.agent_results:
            conn.execute(
                """
                INSERT INTO agent_outputs (
                    workflow_id,
                    agent_name,
                    role,
                    output,
                    duration_ms
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    workflow.workflow_id,
                    result.agent_name,
                    result.role,
                    result.output,
                    result.duration_ms,
                ),
            )

        conn.commit()


def list_workflows() -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT workflow_id, created_at, project_name
            FROM workflows
            ORDER BY created_at DESC
            LIMIT 30
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_workflow(workflow_id: str) -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        workflow = conn.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()

        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        agents = conn.execute(
            """
            SELECT agent_name, role, output, duration_ms
            FROM agent_outputs
            WHERE workflow_id = ?
            ORDER BY id ASC
            """,
            (workflow_id,),
        ).fetchall()

        data = dict(workflow)
        data["agent_results"] = [dict(row) for row in agents]

        return data


app = FastAPI(title=APP_NAME)

init_db()
orchestrator = MultiAgentOrchestrator()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "llm_mode": "mock" if orchestrator.llm.is_mock else "api",
        "model": orchestrator.llm.model,
        "privacy": "No personal information is stored in source code.",
    }


@app.post("/api/run")
def run_workflow(request: ProjectRequest) -> Dict[str, Any]:
    workflow = orchestrator.run(request)

    return {
        "workflow_id": workflow.workflow_id,
        "created_at": workflow.created_at,
        "project_name": workflow.project_name,
        "agent_results": [asdict(item) for item in workflow.agent_results],
        "final_report": workflow.final_report,
    }


@app.get("/api/history")
def history() -> List[Dict[str, Any]]:
    return list_workflows()


@app.get("/api/history/{workflow_id}")
def workflow_detail(workflow_id: str) -> Dict[str, Any]:
    return get_workflow(workflow_id)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Multi-Agent Software Project Automation System</title>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #f6f7f9;
      color: #1f2937;
    }

    .container {
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 80px;
    }

    .hero {
      background: linear-gradient(135deg, #111827, #374151);
      color: white;
      border-radius: 24px;
      padding: 36px;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.15);
      margin-bottom: 24px;
    }

    .hero h1 {
      margin: 0 0 12px;
      font-size: 34px;
    }

    .hero p {
      line-height: 1.7;
      color: #e5e7eb;
      max-width: 850px;
    }

    .grid {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 24px;
    }

    .card {
      background: white;
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
      border: 1px solid #e5e7eb;
    }

    label {
      display: block;
      font-weight: 700;
      margin: 14px 0 8px;
    }

    input,
    textarea {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 14px;
      background: #fbfdff;
    }

    textarea {
      min-height: 92px;
      resize: vertical;
    }

    button {
      margin-top: 18px;
      width: 100%;
      border: none;
      border-radius: 14px;
      padding: 14px;
      background: #111827;
      color: white;
      font-size: 16px;
      font-weight: 800;
      cursor: pointer;
    }

    button:disabled {
      background: #9ca3af;
      cursor: not-allowed;
    }

    .status {
      margin-top: 12px;
      font-size: 14px;
      color: #4b5563;
    }

    .agent-card {
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 16px;
      margin: 14px 0;
      background: #fbfdff;
    }

    .agent-title {
      font-weight: 800;
      margin-bottom: 8px;
    }

    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0b1020;
      color: #e5e7eb;
      padding: 16px;
      border-radius: 14px;
      overflow-x: auto;
      line-height: 1.6;
    }

    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 700;
      margin-right: 8px;
    }

    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>多 Agent 协同软件项目自动化系统</h1>
      <p>
        输入一个软件项目需求，系统会自动调用 Planner、Architect、Backend、Frontend、Test、Doc、QA 等多个 Agent，
        完成需求拆解、架构设计、开发方案、测试计划、文档生成和质量检查。
      </p>
      <span class="badge">Multi-Agent</span>
      <span class="badge">Local Deployment</span>
      <span class="badge">Mock Mode</span>
      <span class="badge">No Personal Data</span>
    </section>

    <div class="grid">
      <div class="card">
        <h2>项目需求输入</h2>

        <label>项目名称</label>
        <input id="project_name" value="Enterprise AI Knowledge Assistant" />

        <label>项目目标</label>
        <textarea id="project_goal">Help teams upload documents, retrieve knowledge, summarize information, and generate project reports.</textarea>

        <label>目标用户</label>
        <input id="target_users" value="Product managers, developers, project managers, operations teams" />

        <label>核心功能</label>
        <textarea id="core_features">1. Document upload and parsing
2. Knowledge retrieval
3. Multi-agent summarization
4. Project report generation
5. Workflow history storage
6. OpenAI-compatible model integration</textarea>

        <label>技术偏好</label>
        <input id="tech_preference" value="Python FastAPI + SQLite + HTML + JavaScript" />

        <label>规模要求</label>
        <textarea id="scale_requirement">Support large-scale projects, task queues, vector databases, user permissions, model routing, and token usage tracking.</textarea>

        <label>额外要求</label>
        <input id="extra_requirement" value="Generate a complete project plan suitable for GitHub demonstration." />

        <button id="runBtn" onclick="runWorkflow()">启动多 Agent 工作流</button>
        <div id="status" class="status">系统默认 Mock 模式，不需要 API Key 即可演示。</div>
      </div>

      <div class="card">
        <h2>Agent 输出结果</h2>
        <div id="result">等待运行...</div>
      </div>
    </div>
  </div>

  <script>
    function escapeHtml(text) {
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    async function runWorkflow() {
      const btn = document.getElementById('runBtn');
      const status = document.getElementById('status');
      const result = document.getElementById('result');

      btn.disabled = true;
      status.innerText = '多 Agent 正在协同执行，请稍等...';
      result.innerHTML = '<p>运行中...</p>';

      const payload = {
        project_name: document.getElementById('project_name').value,
        project_goal: document.getElementById('project_goal').value,
        target_users: document.getElementById('target_users').value,
        core_features: document.getElementById('core_features').value,
        tech_preference: document.getElementById('tech_preference').value,
        scale_requirement: document.getElementById('scale_requirement').value,
        extra_requirement: document.getElementById('extra_requirement').value
      };

      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          throw new Error('Request failed: ' + res.status);
        }

        const data = await res.json();

        let html = `
          <p><strong>Workflow ID：</strong>${escapeHtml(data.workflow_id)}</p>
          <p><strong>创建时间：</strong>${escapeHtml(data.created_at)}</p>
        `;

        for (const item of data.agent_results) {
          html += `
            <div class="agent-card">
              <div class="agent-title">
                ${escapeHtml(item.agent_name)} - ${escapeHtml(item.role)} · ${item.duration_ms}ms
              </div>
              <pre>${escapeHtml(item.output)}</pre>
            </div>
          `;
        }

        html += `
          <h2>最终项目报告</h2>
          <pre>${escapeHtml(data.final_report)}</pre>
        `;

        result.innerHTML = html;
        status.innerText = '运行完成。';
      } catch (err) {
        result.innerHTML = '<pre>' + escapeHtml(String(err)) + '</pre>';
        status.innerText = '运行失败，请检查后端服务。';
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
