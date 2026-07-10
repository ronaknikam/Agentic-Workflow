"""
agent.py
The ReAct loop: at each step, ask the LLM backend to reason about the task
and the trace so far, then either execute the tool it chose or, if it's
ready, return the final answer. This orchestration logic is identical
whether the backend is MockLLM or a real Claude API call -- only
llm_client.plan_step()'s implementation differs.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional
from .llm_client import LLMClient
from .tools import TOOL_SCHEMAS, run_tool
from .memory import AgentMemory


@dataclass
class AgentResult:
    task: str
    final_answer: Optional[str]
    trace: List[dict] = field(default_factory=list)
    num_steps: int = 0
    completed: bool = False
    processing_time_sec: float = 0.0


class Agent:
    def __init__(self, llm: LLMClient, max_steps: int = 6, memory: AgentMemory = None,
                 tool_schemas: List[dict] = None):
        self.llm = llm
        self.max_steps = max_steps
        self.memory = memory
        self.tool_schemas = tool_schemas or TOOL_SCHEMAS

    def run(self, task: str) -> AgentResult:
        t0 = time.time()
        scratchpad = []

        for step_num in range(1, self.max_steps + 1):
            planned = self.llm.plan_step(task, self.tool_schemas, scratchpad)

            if planned.action is None:
                result = AgentResult(
                    task=task,
                    final_answer=planned.final_answer,
                    trace=scratchpad,
                    num_steps=step_num,
                    completed=True,
                    processing_time_sec=round(time.time() - t0, 3),
                )
                if self.memory:
                    self.memory.record(task, planned.final_answer, step_num)
                return result

            observation = run_tool(planned.action, planned.action_input or {})
            scratchpad.append({
                "step": step_num,
                "thought": planned.thought,
                "action": planned.action,
                "action_input": planned.action_input,
                "observation": observation,
            })

        # Ran out of steps without the LLM signaling completion
        fallback_answer = (
            f"Reached the {self.max_steps}-step limit without a final answer. "
            f"Last observation: {scratchpad[-1]['observation'] if scratchpad else 'none'}"
        )
        result = AgentResult(
            task=task,
            final_answer=fallback_answer,
            trace=scratchpad,
            num_steps=self.max_steps,
            completed=False,
            processing_time_sec=round(time.time() - t0, 3),
        )
        if self.memory:
            self.memory.record(task, fallback_answer, self.max_steps)
        return result
