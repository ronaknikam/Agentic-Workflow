import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Agent
from src.llm_client import MockLLM
from src.memory import AgentMemory

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def make_agent(max_steps=6, memory=None):
    return Agent(llm=MockLLM(), max_steps=max_steps, memory=memory)


def test_agent_completes_simple_calculation_task():
    agent = make_agent()
    result = agent.run("Calculate 100 + 250")
    assert result.completed is True
    assert "350" in result.final_answer


def test_agent_object_detection_task():
    agent = make_agent()
    task = f"How many people are in {SAMPLES}/street_scene.jpg?"
    result = agent.run(task)
    assert result.completed is True
    assert result.trace[0]["action"] == "count_objects"
    assert "person" in str(result.final_answer)


def test_agent_ocr_plus_math_multi_step_task():
    agent = make_agent()
    task = f"What text does {SAMPLES}/invoice.png contain, and what is the total of the item amounts?"
    result = agent.run(task)
    assert result.completed is True
    actions_used = [s["action"] for s in result.trace]
    assert "extract_text" in actions_used
    assert "calculator" in actions_used
    assert "2450" in result.final_answer


def test_agent_knowledge_base_task():
    agent = make_agent()
    result = agent.run("What is ReAct in the context of agents?")
    assert result.completed is True
    assert "react" in result.final_answer.lower()


def test_agent_respects_max_steps():
    """An agent that's forced to a very low step limit on a multi-tool task
    should stop and report incompletion rather than looping forever."""
    agent = make_agent(max_steps=1)
    task = f"What text does {SAMPLES}/invoice.png contain, and what is the total of the item amounts?"
    result = agent.run(task)
    assert result.num_steps == 1
    assert result.completed is False


def test_agent_records_to_memory():
    memory = AgentMemory(path="/tmp/test_agent_memory.json")
    memory.clear()
    agent = make_agent(memory=memory)
    agent.run("Calculate 5 + 5")
    recent = memory.recent(1)
    assert len(recent) == 1
    assert recent[0]["task"] == "Calculate 5 + 5"


def test_agent_trace_has_thought_action_observation_for_each_step():
    agent = make_agent()
    result = agent.run("Calculate 7 + 8")
    for step in result.trace:
        assert "thought" in step
        assert "action" in step
        assert "observation" in step


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
