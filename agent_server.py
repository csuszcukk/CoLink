import os
import asyncio
import logging
import json
import uuid
import time
import platform
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import httpx
from httpx import Timeout
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import GPUtil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from prompts import SYSTEM_PROMPT_CODER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agent_server")

MASTER_URL = os.getenv("MASTER_URL", "http://localhost:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
AGENT_NAME = os.getenv("AGENT_NAME", f"agent_{socket.gethostname()}_{uuid.uuid4().hex[:6]}")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))
REGISTER_RETRIES = int(os.getenv("REGISTER_RETRIES", "5"))
REGISTER_DELAY = int(os.getenv("REGISTER_DELAY", "5"))
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
BENCHMARK_PROMPT = "Write a Python async function that fetches a URL with retries."
BENCHMARK_MODEL = os.getenv("BENCHMARK_MODEL", "codellama:7b")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "codellama:13b")
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))


class TaskRequest(BaseModel):
    task_id: str
    prompt: str
    model: str = DEFAULT_MODEL
    system_prompt: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    file_paths: List[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    task_id: str
    agent_id: str
    result: str
    duration_ms: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, bool]
    metrics: Dict[str, float]


class AgentRegistration(BaseModel):
    agent_id: str
    gpu_name: str
    vram_gb: float
    tokens_per_sec: float
    endpoint_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentState:
    agent_id: str
    gpu_name: str
    vram_gb: float
    tokens_per_sec: float
    endpoint_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered: bool = False
    http_client: Optional[httpx.AsyncClient] = None


def get_gpu_info() -> tuple[str, float]:
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return "CPU Only", 0.0
        gpu = gpus[0]
        name = gpu.name
        vram_total = gpu.memoryTotal / 1024.0
        return name, round(vram_total, 1)
    except Exception as e:
        logger.warning(f"GPUtil failed: {e}")
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return name, round(mem.total / (1024**3), 1)
            except Exception:
                pass
    return "Unknown", 0.0


async def benchmark_tokens_per_sec(client: httpx.AsyncClient, model: str = BENCHMARK_MODEL) -> float:
    payload = {
        "model": model,
        "prompt": BENCHMARK_PROMPT,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 100}
    }
    try:
        start = time.perf_counter()
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
        data = resp.json()
        tokens = len(data.get("response", "").split())
        return round(tokens / elapsed, 2) if elapsed > 0 else 10.0
    except Exception as e:
        logger.warning(f"Benchmark failed: {e}, using fallback")
        return 15.0


async def discover_hardware() -> tuple[str, float, float]:
    gpu_name, vram_gb = get_gpu_info()
    logger.info(f"Detected GPU: {gpu_name}, VRAM: {vram_gb}GB")
    async with httpx.AsyncClient(timeout=10) as client:
        tokens_per_sec = await benchmark_tokens_per_sec(client)
    logger.info(f"Benchmark: {tokens_per_sec} tokens/sec")
    return gpu_name, vram_gb, tokens_per_sec


def build_metadata(gpu_name: str, vram_gb: float, tokens_per_sec: float) -> Dict[str, Any]:
    caps = ["python", "async", "fastapi", "pydantic", "sqlalchemy", "docker"]
    if "cuda" in gpu_name.lower() or "nvidia" in gpu_name.lower():
        caps.extend(["cuda", "pytorch", "tensorflow"])
    if vram_gb >= 16:
        caps.append("large_model")
    if tokens_per_sec >= 30:
        caps.append("high_throughput")
    return {
        "capabilities": caps,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "ollama_url": OLLAMA_URL,
        "benchmark_model": BENCHMARK_MODEL,
        "default_model": DEFAULT_MODEL
    }


state = AgentState(
    agent_id=AGENT_NAME,
    gpu_name="",
    vram_gb=0.0,
    tokens_per_sec=0.0,
    endpoint_url=f"http://{socket.gethostbyname(socket.gethostname())}:{AGENT_PORT}",
    metadata={}
)


async def register_with_master() -> bool:
    reg = AgentRegistration(
        agent_id=state.agent_id,
        gpu_name=state.gpu_name,
        vram_gb=state.vram_gb,
        tokens_per_sec=state.tokens_per_sec,
        endpoint_url=state.endpoint_url,
        metadata=state.metadata
    )
    for attempt in range(REGISTER_RETRIES):
        try:
            resp = await state.http_client.post(
                f"{MASTER_URL}/register",
                json=reg.dict(),
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Registered with master: {data}")
                state.registered = True
                return True
        except Exception as e:
            logger.warning(f"Registration attempt {attempt + 1} failed: {e}")
        await asyncio.sleep(REGISTER_DELAY)
    logger.error("Failed to register with master after all retries")
    return False


async def send_heartbeat():
    try:
        health = await check_health_internal()
        resp = await state.http_client.post(
            f"{MASTER_URL}/heartbeat/{state.agent_id}",
            json=health.dict(),
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Heartbeat failed: {e}")
        return False


async def check_health_internal() -> HealthResponse:
    checks = {"ollama": False, "gpu": False, "memory": True, "disk": True}
    metrics = {
        "tokens_per_sec": state.tokens_per_sec,
        "vram_used_gb": 0.0,
        "vram_total_gb": state.vram_gb,
        "cpu_percent": 0.0,
        "ram_percent": 0.0
    }
    try:
        resp = await state.http_client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        checks["ollama"] = resp.status_code == 200
    except Exception:
        pass
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            checks["gpu"] = True
            metrics["vram_used_gb"] = round(gpus[0].memoryUsed / 1024.0, 1)
    except Exception:
        pass
    status = "healthy" if all(checks.values()) else "degraded" if checks["ollama"] else "unhealthy"
    return HealthResponse(status=status, checks=checks, metrics=metrics)


def strip_markdown_code(text: str) -> str:
    lines = text.strip().split('\n')
    if lines and lines[0].startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].startswith('```'):
        lines = lines[:-1]
    return '\n'.join(lines).strip()


async def call_ollama(client: httpx.AsyncClient, model: str, prompt: str, system: str = "") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 8192}
    }
    resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=TASK_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "")


async def process_task_logic(req: TaskRequest) -> TaskResponse:
    start = time.perf_counter()
    try:
        system = req.system_prompt or SYSTEM_PROMPT_CODER
        full_prompt = req.prompt
        if req.context:
            full_prompt = f"CONTEXT:\n{json.dumps(req.context, indent=2)}\n\nTASK:\n{req.prompt}"
        if req.file_paths:
            full_prompt += f"\n\nFILE PATHS: {', '.join(req.file_paths)}"
        
        result = await call_ollama(state.http_client, req.model, full_prompt, system)
        result = strip_markdown_code(result)
        
        duration = int((time.perf_counter() - start) * 1000)
        return TaskResponse(
            task_id=req.task_id,
            agent_id=state.agent_id,
            result=result,
            duration_ms=duration
        )
    except Exception as e:
        duration = int((time.perf_counter() - start) * 1000)
        logger.exception(f"Task {req.task_id} failed")
        return TaskResponse(
            task_id=req.task_id,
            agent_id=state.agent_id,
            result="",
            duration_ms=duration,
            error=str(e)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(TASK_TIMEOUT))
    gpu_name, vram_gb, tokens_per_sec = await discover_hardware()
    state.gpu_name = gpu_name
    state.vram_gb = vram_gb
    state.tokens_per_sec = tokens_per_sec
    state.metadata = build_metadata(gpu_name, vram_gb, tokens_per_sec)
    await register_with_master()
    asyncio.create_task(heartbeat_loop())
    yield
    await state.http_client.aclose()


async def heartbeat_loop():
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        if state.registered:
            await send_heartbeat()


app = FastAPI(title=f"CoLink Agent - {AGENT_NAME}", lifespan=lifespan)


@app.post("/process_task", response_model=TaskResponse)
async def process_task(req: TaskRequest):
    return await process_task_logic(req)


@app.get("/health", response_model=HealthResponse)
async def health():
    return await check_health_internal()


@app.get("/info")
async def info():
    return {
        "agent_id": state.agent_id,
        "gpu_name": state.gpu_name,
        "vram_gb": state.vram_gb,
        "tokens_per_sec": state.tokens_per_sec,
        "registered": state.registered,
        "metadata": state.metadata
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)