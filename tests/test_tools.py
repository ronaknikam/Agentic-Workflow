import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import calculator, extract_text, count_objects, knowledge_base_lookup, run_tool

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_calculator_basic_arithmetic():
    assert calculator("2 + 3")["result"] == 5
    assert calculator("10 / 4")["result"] == 2.5
    assert calculator("2 ** 3")["result"] == 8


def test_calculator_rejects_unsafe_input():
    """The calculator must NOT be a bare eval() -- arbitrary code execution
    via a 'calculate this' tool call is a real vulnerability class for
    agentic systems with code-execution tools."""
    result = calculator("__import__('os').system('echo pwned')")
    assert result["result"] is None
    assert result["error"] is not None


def test_calculator_rejects_non_numeric_names():
    result = calculator("open('/etc/passwd').read()")
    assert result["result"] is None
    assert result["error"] is not None


def test_extract_text_reads_known_content():
    result = extract_text(os.path.join(SAMPLES, "invoice.png"))
    assert result["error"] is None
    assert "450" in result["text"]
    assert "1200" in result["text"]


def test_extract_text_handles_missing_file():
    result = extract_text("does_not_exist.png")
    assert result["error"] is not None
    assert result["text"] is None


def test_count_objects_finds_people_and_bus():
    result = count_objects(os.path.join(SAMPLES, "street_scene.jpg"))
    assert result["error"] is None
    assert "person" in result["all_counts"]
    assert "bus" in result["all_counts"]


def test_count_objects_target_class_filter():
    result = count_objects(os.path.join(SAMPLES, "street_scene.jpg"), target_class="person")
    assert result["target_class"] == "person"
    assert result["count"] >= 1


def test_knowledge_base_lookup_finds_relevant_entry():
    result = knowledge_base_lookup("what is ReAct in agents")
    assert result["found"] is True
    assert "react" in result["topic"].lower()


def test_knowledge_base_lookup_no_match_returns_not_found():
    result = knowledge_base_lookup("what is the airspeed velocity of an unladen swallow")
    assert result["found"] is False


def test_run_tool_unknown_tool_name():
    result = run_tool("nonexistent_tool", {})
    assert "error" in result


def test_run_tool_invalid_arguments():
    result = run_tool("calculator", {"wrong_arg": "1+1"})
    assert "error" in result


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
