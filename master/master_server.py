import os
import asyncio
import logging
import json
import uuid
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
import GPUtil
import uvicorn

# IMPORT THE PROMPTS FROM YOUR PROMPTS.PY FILE
from prompts import SYSTEM_PROMPT_LEADER_SPLIT, SYSTEM_PROMPT_CODER, SYSTEM_PROMPT_AGGREGATOR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("master_server")

MASTER_URL = os.getenv("MASTER_URL", "http://0.0.0.0:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))
LEADER_MODEL = os.getenv("LEADER_MODEL", "codellama:34b")
AGENT_MODEL = os.getenv("AGENT_MODEL", "codellama:13b")


class AgentStatus(str, Enum):
    REGISTERING = "registering"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REASSIGNED = "reassigned"


class SubTask(BaseModel):
    id: str
    title: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=10)
    estimated_tokens: int
    required_capabilities: List[str] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0


class AgentInfo(BaseModel):
    agent_id: str
    gpu_name: str
    vram_gb: float
    tokens_per_sec: float
    endpoint_url: str
    status: AgentStatus = AgentStatus.REGISTERING
    weight: float = 1.0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    current_task: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRegistration(BaseModel):
    agent_id: str
    gpu_name: str
    vram_gb: float
    tokens_per_sec: float
    endpoint_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("vram_gb", "tokens_per_sec")
    def positive_float(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class TaskSubmission(BaseModel):
    prompt: str
    project_name: str = "project"
    leader_model: str = LEADER_MODEL
    agent_model: str = AGENT_MODEL
    max_parallel: int = 4


class TaskResult(BaseModel):
    task_id: str
    agent_id: str
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int


class HealthResponse(BaseModel):
    status: AgentStatus
    checks: Dict[str, bool]
    metrics: Dict[str, float]


class MasterState:
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {}
        self.tasks: Dict[str, Dict[str, SubTask]] = {}
        self.task_results: Dict[str, Dict[str, str]] = {}
        self.active_project: Optional[str] = None
        self.leader_agent_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self.http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(TASK_TIMEOUT))

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

    def calculate_weight(self, agent: AgentRegistration) -> float:
        vram_score = min(agent.vram_gb / 24.0, 2.0)
        token_score = min(agent.tokens_per_sec / 50.0, 2.0)
        return round(vram_score * 0.6 + token_score * 0.4, 2)

    async def register_agent(self, reg: AgentRegistration) -> AgentInfo:
        async with self._lock:
            weight = self.calculate_weight(reg)
            agent = AgentInfo(
                agent_id=reg.agent_id,
                gpu_name=reg.gpu_name,
                vram_gb=reg.vram_gb,
                tokens_per_sec=reg.tokens_per_sec,
                endpoint_url=reg.endpoint_url,
                weight=weight,
                metadata=reg.metadata,
                status=AgentStatus.HEALTHY
            )
            self.agents[reg.agent_id] = agent
            logger.info(f"Registered agent {reg.agent_id} with weight {weight}")
            return agent

    async def update_heartbeat(self, agent_id: str, health: HealthResponse):
        async with self._lock:
            if agent_id not in self.agents:
                return
            agent = self.agents[agent_id]
            agent.last_heartbeat = datetime.utcnow()
            agent.status = health.status
            if health.metrics.get("tokens_per_sec"):
                agent.tokens_per_sec = health.metrics["tokens_per_sec"]
                agent.weight = self._recalculate_weight(agent)

    def _recalculate_weight(self, agent: AgentInfo) -> float:
        vram_score = min(agent.vram_gb / 24.0, 2.0)
        token_score = min(agent.tokens_per_sec / 50.0, 2.0)
        health_modifier = 1.0 if agent.status == AgentStatus.HEALTHY else 0.5
        return round((vram_score * 0.6 + token_score * 0.4) * health_modifier, 2)

    def get_healthy_agents(self) -> List[AgentInfo]:
        now = datetime.utcnow()
        healthy = []
        for agent in self.agents.values():
            if agent.status == AgentStatus.HEALTHY:
                if now - agent.last_heartbeat < timedelta(seconds=HEALTH_CHECK_INTERVAL * 3):
                    healthy.append(agent)
        return sorted(healthy, key=lambda a: a.weight, reverse=True)

    def select_leader(self) -> Optional[AgentInfo]:
        healthy = self.get_healthy_agents()
        return healthy[0] if healthy else None

    def get_available_agents(self, exclude: Set[str] = None) -> List[AgentInfo]:
        exclude = exclude or set()
        healthy = self.get_healthy_agents()
        return [a for a in healthy if a.agent_id not in exclude and a.current_task is None]

    async def submit_project(self, submission: TaskSubmission) -> str:
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        async with self._lock:
            self.active_project = project_id
            self.tasks[project_id] = {}
            self.task_results[project_id] = {}
            self.leader_agent_id = None

        asyncio.create_task(self._execute_project(project_id, submission))
        return project_id

    async def _execute_project(self, project_id: str, submission: TaskSubmission):
        try:
            await self._phase_leader_selection(project_id)
            await self._phase_task_splitting(project_id, submission)
            await self._phase_distribute_tasks(project_id, submission)
            await self._phase_aggregation(project_id, submission)
            logger.info(f"Project {project_id} completed successfully")
        except Exception as e:
            logger.exception(f"Project {project_id} failed: {e}")
        finally:
            async with self._lock:
                self.active_project = None

    async def _phase_leader_selection(self, project_id: str):
        leader = self.select_leader()
        if not leader:
            raise RuntimeError("No healthy agents available for leader role")
        async with self._lock:
            self.leader_agent_id = leader.agent_id
            leader.current_task = f"leader_{project_id}"
        logger.info(f"Selected leader: {leader.agent_id}")

    async def _phase_task_splitting(self, project_id: str, submission: TaskSubmission):
        leader = self.agents[self.leader_agent_id]
        prompt = f"{SYSTEM_PROMPT_LEADER_SPLIT}\n\nPROJECT: {submission.prompt}\n\nConstraints: Max parallel tasks: {submission.max_parallel}. Target model for agents: {submission.agent_model}."
        
        response = await self._call_ollama(leader.endpoint_url, submission.leader_model, prompt, system=SYSTEM_PROMPT_LEADER_SPLIT)
        
        try:
            data = json.loads(response)
            subtasks = data.get("subtasks", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse leader JSON: {e}")
            raise RuntimeError(f"Leader returned invalid JSON: {e}")

        async with self._lock:
            for st_data in subtasks:
                st = SubTask(**st_data)
                self.tasks[project_id][st.id] = st
        logger.info(f"Split project into {len(subtasks)} subtasks")

    async def _phase_distribute_tasks(self, project_id: str, submission: TaskSubmission):
        semaphore = asyncio.Semaphore(submission.max_parallel)
        
        async def process_task(task: SubTask):
            async with semaphore:
                await self._execute_subtask(project_id, task, submission)

        tasks = list(self.tasks[project_id].values())
        await asyncio.gather(*[process_task(t) for t in tasks], return_exceptions=True)

    async def _execute_subtask(self, project_id: str, task: SubTask, submission: TaskSubmission):
        max_retries = 2
        while task.retries <= max_retries:
            agents = self.get_available_agents(exclude={task.assigned_agent} if task.assigned_agent else set())
            if not agents:
                await asyncio.sleep(5)
                continue

            agent = self._select_best_agent(task, agents)
            if not agent:
                await asyncio.sleep(5)
                continue

            async with self._lock:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = agent.agent_id
                agent.current_task = task.id
                task.started_at = datetime.utcnow()

            try:
                # FIXED: Pass project_id correctly into the call
                result = await self._call_agent(agent, task, submission, project_id)
                async with self._lock:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.utcnow()
                    agent.current_task = None
                    agent.completed_tasks += 1
                    self.task_results[project_id][task.id] = result
                logger.info(f"Task {task.id} completed by {agent.agent_id}")
                return
            except Exception as e:
                logger.warning(f"Task {task.id} failed on {agent.agent_id}: {e}")
                async with self._lock:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.retries += 1
                    agent.current_task = None
                    agent.failed_tasks += 1
                if task.retries > max_retries:
                    logger.error(f"Task {task.id} failed permanently after {max_retries} retries")
                    raise

    def _select_best_agent(self, task: SubTask, agents: List[AgentInfo]) -> Optional[AgentInfo]:
        capable = [a for a in agents if all(c in a.metadata.get("capabilities", []) for c in task.required_capabilities)]
        if not capable:
            capable = agents
        return max(capable, key=lambda a: a.weight) if capable else None

    # FIXED: Added project_id parameter to pull context accurately
    async def _call_agent(self, agent: AgentInfo, task: SubTask, submission: TaskSubmission, project_id: str) -> str:
        completed_dependencies = self.task_results.get(project_id, {})
        
        prompt = f"""TASK: {task.title}
DESCRIPTION: {task.description}
FILE PATHS: {', '.join(task.file_paths)}
REQUIRED CAPABILITIES: {', '.join(task.required_capabilities)}
COMPLEXITY: {task.complexity}/10
ESTIMATED TOKENS: {task.estimated_tokens}

DEPENDENCIES COMPLETED:
{json.dumps(completed_dependencies, indent=2)}

PROJECT CONTEXT:
{submission.prompt}

Produce complete, production-ready code for the specified files only."""
        
        return await self._call_ollama(agent.endpoint_url, submission.agent_model, prompt, system=SYSTEM_PROMPT_CODER)

    async def _phase_aggregation(self, project_id: str, submission: TaskSubmission):
        leader = self.agents[self.leader_agent_id]
        all_results = "\n\n".join([
            f"===FILE: {task.file_paths[0] if task.file_paths else f'output_{task.id}.py'}===\n{result}"
            for task, result in self.task_results[project_id].items()
        ])
        
        prompt = f"""ORIGINAL PROJECT REQUEST:
{submission.prompt}

ALL SUBTASK RESULTS:
{all_results}

Synthesize these into a complete, production-ready codebase. Fix any integration issues, resolve imports, add missing files (main.py, requirements.txt, Dockerfile, README.md, tests)."""
        
        final_code = await self._call_ollama(leader.endpoint_url, submission.leader_model, prompt, system=SYSTEM_PROMPT_AGGREGATOR)
        
        async with self._lock:
            self.task_results[project_id]["_FINAL_"] = final_code
        
        async with self._lock:
            leader.current_task = None
        logger.info(f"Project {project_id} aggregation complete")

    async def _call_ollama(self, url: str, model: str, prompt: str, system: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 8192}
        }
        try:
            resp = await self.http_client.post(f"{url}/api/generate", json=payload, timeout=TASK_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama timeout after {TASK_TIMEOUT}s")
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    async def health_check_loop(self):
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            await self._check_all_agents()

    async def _check_all_agents(self):
        async with self._lock:
            agents = list(self.agents.values())
        
        for agent in agents:
            try:
                resp = await self.http_client.get(f"{agent.endpoint_url}/health", timeout=10)
                if resp.status_code == 200:
                    health = HealthResponse(**resp.json())
                    await self.update_heartbeat(agent.agent_id, health)
                else:
                    await self._mark_unhealthy(agent.agent_id)
            except Exception:
                await self._mark_unhealthy(agent.agent_id)

    async def _mark_unhealthy(self, agent_id: str):
        async with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.UNHEALTHY
                if self.agents[agent_id].current_task:
                    task_id = self.agents[agent_id].current_task
                    for proj_tasks in self.tasks.values():
                        if task_id in proj_tasks:
                            proj_tasks[task_id].status = TaskStatus.REASSIGNED
                            proj_tasks[task_id].assigned_agent = None
                self.agents[agent_id].current_task = None

    def get_project_status(self, project_id: str) -> Dict:
        tasks = self.tasks.get(project_id, {})
        return {
            "project_id": project_id,
            "leader": self.leader_agent_id,
            "total_tasks": len(tasks),
            "completed": sum(1 for t in tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks.values() if t.status == TaskStatus.FAILED),
            "in_progress": sum(1 for t in tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "tasks": {tid: {"status": t.status.value, "agent": t.assigned_agent, "error": t.error} 
                     for tid, t in tasks.items()}
        }


state = MasterState()

FRONTEND_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CoLink Master Controller</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 10px 0; }
        .agent { display: inline-block; margin: 5px; padding: 10px 15px; border-radius: 4px; background: #f0f0f0; }
        .agent.healthy { background: #d4edda; } .agent.unhealthy { background: #f8d7da; }
        .task { padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; background: #f8f9fa; }
        .task.completed { border-color: #28a745; } .task.failed { border-color: #dc3545; }
        .task.in_progress { border-color: #ffc107; }
        textarea { width: 100%; height: 200px; font-family: monospace; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:disabled { background: #6c757d; }
        pre { background: #f8f9fa; padding: 15px; overflow: auto; max-height: 400px; }
    </style>
</head>
<body>
    <h1>CoLink Master Controller</h1>
    <div class="card">
        <h2>Submit Project</h2>
        <input type="text" id="projectName" placeholder="Project name" value="my_project">
        <textarea id="prompt" placeholder="Describe your project..."></textarea>
        <br><br>
        <button onclick="submitProject()">Submit Project</button>
        <button onclick="checkHealth()">Check Health</button>
        <div id="result"></div>
    </div>
    <div class="card">
        <h2>Agents</h2>
        <div id="agents"></div>
    </div>
    <div class="card">
        <h2>Project Status</h2>
        <div id="status"></div>
    </div>
    <script>
        async function submitProject() {
            const prompt = document.getElementById('prompt').value;
            const name = document.getElementById('projectName').value;
            const res = await fetch('/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, project_name: name})
            });
            const data = await res.json();
            document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            pollStatus(data.project_id);
        }
        async function pollStatus(projectId) {
            const res = await fetch('/status/' + projectId);
            const data = await res.json();
            document.getElementById('status').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            if (data.completed < data.total_tasks && data.failed === 0) {
                setTimeout(() => pollStatus(projectId), 3000);
            }
        }
        async function checkHealth() {
            const res = await fetch('/agents');
            const data = await res.json();
            document.getElementById('agents').innerHTML = data.map(a => 
                `<span class="agent ${a.status}">${a.agent_id} (${a.gpu_name}, ${a.vram_gb}GB, ${a.tokens_per_sec.toFixed(1)} tok/s, weight: ${a.weight})</span>`
            ).join('');
        }
        setInterval(checkHealth, 10000);
        checkHealth();
    </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.initialize()
    asyncio.create_task(state.health_check_loop())
    yield
    await state.close()


app = FastAPI(title="CoLink Master", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    return FRONTEND_HTML


@app.post("/register")
async def register_agent(reg: AgentRegistration):
    agent = await state.register_agent(reg)
    return {"status": "registered", "agent_id": agent.agent_id, "weight": agent.weight}


@app.post("/heartbeat/{agent_id}")
async def heartbeat(agent_id: str, health: HealthResponse):
    await state.update_heartbeat(agent_id, health)
    return {"status": "ok"}


@app.post("/submit")
async def submit_project(submission: TaskSubmission):
    project_id = await state.submit_project(submission)
    return {"project_id": project_id, "status": "started"}


@app.get("/status/{project_id}")
async def project_status(project_id: str):
    return state.get_project_status(project_id)


@app.get("/agents")
async def list_agents():
    return [asdict(a) for a in state.agents.values()]


@app.get("/health")
async def master_health():
    # FIXED: Added try-except fallback for Jetson Nano embedded GPU environment
    try:
        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
    except Exception:
        gpu_count = 0
        
    return {
        "status": "healthy",
        "agents": len(state.agents),
        "healthy_agents": len(state.get_healthy_agents()),
        "active_project": state.active_project,
        "gpu_count": gpu_count
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)