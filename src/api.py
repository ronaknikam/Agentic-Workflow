"""
api.py
FastAPI service exposing the agent. Run with:
    uvicorn src.api:app --reload --port 8000

POST a task to /run and get back the full reasoning trace + final answer.
Uses MockLLM by default (no API key needed to run this demo); set
ANTHROPIC_API_KEY and AGENT_BACKEND=anthropic to use a real Claude backend.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import Agent
from .llm_client import MockLLM
from .memory import AgentMemory

app = FastAPI(
    title="AgentFlow API",
    description="A ReAct-style agent that plans multi-step tool use "
                 "(OCR, object detection, calculator, knowledge base lookup) "
                 "to answer a task, returning the full reasoning trace.",
    version="1.0.0",
)

_memory = AgentMemory()


def _build_agent() -> Agent:
    backend = os.environ.get("AGENT_BACKEND", "mock")
    if backend == "anthropic":
        from .llm_client import AnthropicLLM
        llm = AnthropicLLM()
    else:
        llm = MockLLM()
    return Agent(llm=llm, memory=_memory)


class TaskRequest(BaseModel):
    task: str


@app.get("/health")
def health():
    return {"status": "ok", "backend": os.environ.get("AGENT_BACKEND", "mock")}


@app.post("/run")
def run_task(request: TaskRequest):
    agent = _build_agent()
    result = agent.run(request.task)
    return {
        "task": result.task,
        "final_answer": result.final_answer,
        "num_steps": result.num_steps,
        "completed": result.completed,
        "processing_time_sec": result.processing_time_sec,
        "trace": result.trace,
    }


@app.get("/memory/recent")
def recent_memory(n: int = 5):
    return _memory.recent(n)
