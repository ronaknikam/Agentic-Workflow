"""
llm_client.py
The agent's reasoning loop (agent.py) is written against this interface,
not against any specific LLM provider. Two implementations are provided:

- MockLLM: a deterministic, rule-based planner used for tests and offline
  demos. No API key, no network call, no cost -- and because it's
  deterministic, the agent's control-flow logic (tool selection, loop
  termination, error handling) can be unit-tested reliably.

- AnthropicLLM: a real backend using the Claude API's native tool-use
  support. Swapping MockLLM for AnthropicLLM in agent construction is the
  only change needed to go from offline demo to a live LLM-driven agent --
  the ReAct loop itself doesn't change.
"""
import os
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class PlannedStep:
    thought: str
    action: Optional[str]         # tool name, or None if this is the final answer
    action_input: Optional[dict]  # tool arguments, or None
    final_answer: Optional[str] = None


class LLMClient(ABC):
    @abstractmethod
    def plan_step(self, task: str, tool_schemas: List[dict], scratchpad: List[dict]) -> PlannedStep:
        """Given the task, available tools, and the trace of (thought, action,
        observation) so far, decide the next step."""
        raise NotImplementedError


class MockLLM(LLMClient):
    """
    A small deterministic policy standing in for a real LLM. It matches
    keywords in the task to decide which tool to call next, and reads
    prior observations off the scratchpad to decide when it has enough
    information to answer -- the same control flow a real LLM-driven
    ReAct loop follows, just without an actual model doing the reasoning.
    """

    def plan_step(self, task: str, tool_schemas: List[dict], scratchpad: List[dict]) -> PlannedStep:
        task_lower = task.lower()
        observations = [s["observation"] for s in scratchpad if "observation" in s]
        used_tools = [s["action"] for s in scratchpad if s.get("action")]

        # --- Step 1: image-based tools (OCR / object counting) ---
        image_path = self._extract_image_path(task)
        if image_path:
            if "how many" in task_lower or "count" in task_lower or "people" in task_lower or "detect" in task_lower:
                if "count_objects" not in used_tools:
                    return PlannedStep(
                        thought="The task asks about counting/detecting objects in an image. I'll run object detection.",
                        action="count_objects",
                        action_input={"image_path": image_path},
                    )
            elif "extract_text" not in used_tools and any(
                w in task_lower for w in ["text", "read", "ocr", "invoice", "total", "amount", "says"]
            ):
                return PlannedStep(
                    thought="The task asks about text content in an image. I'll run OCR.",
                    action="extract_text",
                    action_input={"image_path": image_path},
                )

        # --- Step 2: arithmetic follow-up on a prior observation ---
        needs_math = any(w in task_lower for w in ["total", "sum", "how much", "calculate", "average", "add"])
        if needs_math and observations and "calculator" not in used_tools:
            last_obs = scratchpad[-1]["observation"]
            # Pull numbers from the OCR'd text specifically, not tool metadata
            # like char_count -- otherwise unrelated numbers leak into the sum.
            source_text = last_obs.get("text") if isinstance(last_obs, dict) and last_obs.get("text") else str(last_obs)
            expr = self._extract_math_expression(source_text, task_lower)
            if expr:
                return PlannedStep(
                    thought=f"I have the raw numbers now; I need to compute the result: {expr}",
                    action="calculator",
                    action_input={"expression": expr},
                )

        # --- Step 3: knowledge base lookup ---
        # Only treat this as a pure factual question if it's not already an
        # image-analysis task -- otherwise "...and what is the total" on an
        # OCR task would wrongly trigger a KB lookup after the real answer
        # is already in hand.
        if (not image_path and observations == []
                and any(w in task_lower for w in ["what is", "explain", "define", "who is"])
                and "knowledge_base_lookup" not in used_tools):
            query = re.sub(r"^(what is|explain|define|who is)\s+", "", task_lower).strip(" ?")
            return PlannedStep(
                thought=f"This is a factual question. I'll check the knowledge base for '{query}'.",
                action="knowledge_base_lookup",
                action_input={"query": query},
            )

        # --- Step 4: plain arithmetic task with no image involved ---
        if not image_path and needs_math and "calculator" not in used_tools:
            expr = self._extract_math_expression(task, task_lower)
            if expr:
                return PlannedStep(
                    thought=f"This is a direct calculation: {expr}",
                    action="calculator",
                    action_input={"expression": expr},
                )

        # --- Step 5: enough information gathered -> final answer ---
        if observations:
            return PlannedStep(
                thought="I have enough information from the tool results to answer.",
                action=None,
                action_input=None,
                final_answer=self._synthesize_answer(task, scratchpad),
            )

        return PlannedStep(
            thought="I don't have a matching tool for this task and no prior observations to draw on.",
            action=None,
            action_input=None,
            final_answer="I wasn't able to find a suitable tool to complete this task.",
        )

    @staticmethod
    def _extract_image_path(task: str) -> Optional[str]:
        match = re.search(r"[\w./\-]+\.(?:png|jpg|jpeg)", task, re.IGNORECASE)
        return match.group(0) if match else None

    @staticmethod
    def _extract_math_expression(text: str, task_lower: str) -> Optional[str]:
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if len(numbers) >= 2:
            return " + ".join(numbers)
        if len(numbers) == 1:
            return numbers[0]
        return None

    @staticmethod
    def _synthesize_answer(task: str, scratchpad: List[dict]) -> str:
        parts = []
        for step in scratchpad:
            obs = step["observation"]
            action = step["action"]
            if action == "extract_text" and obs.get("text"):
                parts.append(f"extracted text: \"{obs['text']}\"")
            elif action == "calculator" and obs.get("result") is not None:
                parts.append(f"calculated result: {obs['result']}")
            elif action == "count_objects" and obs.get("all_counts"):
                parts.append(f"object counts: {obs['all_counts']}")
            elif action == "knowledge_base_lookup" and obs.get("found"):
                parts.append(obs["content"])
        return " | ".join(parts) if parts else "No usable result was produced by the tools called."


class AnthropicLLM(LLMClient):
    """
    Real backend using Claude's native tool-use API. Requires the
    `anthropic` package and an ANTHROPIC_API_KEY environment variable.

    Model name is read from ANTHROPIC_MODEL (default below) rather than
    hardcoded, since available model identifiers change over time --
    check https://docs.claude.com for current model strings before
    deploying this.
    """

    def __init__(self, model: str = None, api_key: str = None):
        import anthropic  # imported lazily so MockLLM users don't need the package installed
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def plan_step(self, task: str, tool_schemas: List[dict], scratchpad: List[dict]) -> PlannedStep:
        messages = self._build_messages(task, scratchpad)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=(
                "You are a task-solving agent. Use the available tools to gather "
                "information step by step, then give a final answer. Call at most "
                "one tool per turn."
            ),
            messages=messages,
            tools=tool_schemas,
        )

        text_block = next((b for b in response.content if b.type == "text"), None)
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)

        if tool_block:
            return PlannedStep(
                thought=text_block.text if text_block else "",
                action=tool_block.name,
                action_input=tool_block.input,
            )

        return PlannedStep(
            thought="",
            action=None,
            action_input=None,
            final_answer=text_block.text if text_block else "No response generated.",
        )

    @staticmethod
    def _build_messages(task: str, scratchpad: List[dict]) -> List[dict]:
        messages = [{"role": "user", "content": task}]
        for step in scratchpad:
            if step.get("action"):
                messages.append({
                    "role": "assistant",
                    "content": f"[Called {step['action']} with {json.dumps(step.get('action_input', {}))}]",
                })
                messages.append({
                    "role": "user",
                    "content": f"[Tool result: {step['observation']}]",
                })
        return messages
