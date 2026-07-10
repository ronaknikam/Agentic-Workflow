# AgentFlow

**A ReAct-style agent that plans multi-step tool use, with a pluggable LLM backend — a
deterministic mock for offline testing, and a real Claude API backend for production.**

Most "agent" demos hardcode a single LLM call and a single tool. AgentFlow separates three
concerns that a real agentic system needs to keep separate: the **reasoning loop** (when to
call a tool vs. when to answer), the **LLM backend** (which model actually does the reasoning),
and the **tools** (what the agent can act on). Swapping `MockLLM` for a real Claude API client
is a one-line change — the orchestration logic doesn't move.

The tools pull from my own CV work: OCR (`extract_text`) and object detection (`count_objects`,
YOLOv8), plus a calculator and a small local knowledge-base lookup (a minimal RAG-style
retrieval step). The point isn't the tools themselves — it's the agent correctly deciding
*which* tool to call, *when* it has enough information to stop, and *how* to combine results
from more than one tool into a coherent answer.

## Architecture

```
   Task (natural language)
         │
         ▼
┌──────────────────┐   plan_step(task, tool_schemas, scratchpad) → next
│   ReAct Loop      │   thought + action, or a final answer. Same loop
│   (agent.py)      │   regardless of which LLM backend is plugged in.
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌──────────────┐
│MockLLM │  │ AnthropicLLM │   Swap one line in Agent(llm=...) to go from
│(offline,│  │(real Claude  │   offline/testable to live LLM-driven.
│deter-  │  │ API, native  │
│ministic)│  │ tool-use)    │
└────────┘  └──────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  Tools: calculator · extract_text (OCR) ·     │
│  count_objects (YOLOv8) · knowledge_base_lookup│
└──────────────────────────────────────────────┘
          │
          ▼
   Trace (thought/action/observation per step) + final answer
```

## Example run

```
TASK: What text does samples/invoice.png contain, and what is the total of the item amounts?

Step 1: The task asks about text content in an image. I'll run OCR.
    -> extract_text({'image_path': 'samples/invoice.png'})
    -> observation: {'text': 'ACME SUPPLIES INVOICE\nItem A: 450\nItem B: 1200\nItem C: 800\n...'}
Step 2: I have the raw numbers now; I need to compute the result: 450 + 1200 + 800
    -> calculator({'expression': '450 + 1200 + 800'})
    -> observation: {'result': 2450, 'error': None}

FINAL ANSWER: extracted text: "ACME SUPPLIES INVOICE..." | calculated result: 2450
```

## Quickstart

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr   # OCR tool dependency

# Generate sample assets (invoice image for OCR, real photo for detection)
python3 tests/generate_samples.py

# Run tests (all offline, MockLLM -- no API key needed)
pytest tests/ -v

# Try it from the CLI
python3 -c "
from src.agent import Agent
from src.llm_client import MockLLM
agent = Agent(llm=MockLLM())
result = agent.run('How many people are in samples/street_scene.jpg?')
print(result.final_answer)
"

# Serve as an API (defaults to MockLLM backend)
uvicorn src.api:app --reload --port 8000
curl -X POST http://localhost:8000/run -H "Content-Type: application/json" \
     -d '{"task": "Calculate 40 + 60"}'
```

### Using a real Claude backend instead of MockLLM

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
export AGENT_BACKEND=anthropic
uvicorn src.api:app --reload --port 8000
```

`AnthropicLLM` uses Claude's native tool-use API — the same `TOOL_SCHEMAS` list is passed
directly as the `tools` parameter, no translation layer needed. Check
[docs.claude.com](https://docs.claude.com) for current model identifiers before deploying;
the default is read from `ANTHROPIC_MODEL` rather than hardcoded for this reason.

## Why MockLLM matters (not just a stub)

`MockLLM` is a real, if simple, deterministic policy — it inspects the task and the
scratchpad to decide the next tool call, using the same control-flow contract a real LLM
backend follows. That determinism is what makes `tests/test_agent.py` possible: 18 tests that
verify the *orchestration logic* (correct tool selection, multi-step combination, step-limit
handling, memory persistence) run in ~5 seconds with zero API cost and zero flakiness. A
test suite built directly against a real LLM would be slower, cost money per run, and could
fail nondeterministically even when the orchestration code is correct — bad tests would be
mixed together with the actual bug it's supposed to catch.

## Bugs found and fixed while building this

Two real bugs surfaced when I actually ran multi-step tasks end-to-end (not just eyeballing
the code):

1. **Math extraction was pulling numbers from the whole observation dict**, including tool
   metadata like `char_count`, not just the OCR'd text — so "450 + 1200 + 800" was corrupted
   into "450 + 1200 + 00 + 88". Fixed by extracting numbers specifically from the relevant
   field of the prior observation.
2. **Knowledge-base search matched the wrong entry** for "What is ReAct in the context of
   agents?" — naive keyword overlap with no stopword filtering or plural handling gave a
   false-positive match on an unrelated entry. Fixed with stopword filtering, simple
   singular/plural folding, and topic-weighted scoring.

Both are exactly the class of bug that a demo which only "looks right" would ship with — see
`tests/test_tools.py::test_knowledge_base_lookup_finds_relevant_entry` and the OCR+math
integration test in `test_agent.py` for the regression coverage.

## Design notes

- **Calculator is a whitelisted AST evaluator, not `eval()`.** A "calculate this" tool is a
  classic code-execution vulnerability in agentic systems if it isn't sandboxed — see
  `test_calculator_rejects_unsafe_input`.
- **`max_steps` is a hard circuit breaker.** If the loop doesn't reach a final answer within
  the budget, the agent returns `completed: False` with whatever partial information it has,
  rather than looping indefinitely.
- **Memory is separate from the reasoning loop.** `AgentMemory` persists completed runs to
  disk; the agent doesn't depend on it to function, so it's optional (`Agent(memory=None)`
  works fine) — a common pattern for keeping side effects out of core control flow.

## What I'd add next

- Parallel tool calls where the plan allows independent sub-tasks
- A real vector-store-backed knowledge base instead of keyword overlap (the fix above is a
  patch on a fundamentally simple retrieval method, not a replacement for embeddings)
- Streaming responses from the Anthropic backend
- Tool permissioning / a confirmation step before tools with side effects (this repo's tools
  are all read-only, but a production agent with write access needs this)

## Stack

Python · Claude API (tool use) · YOLOv8 · Tesseract OCR · FastAPI · pytest

---

*Built to demonstrate agent orchestration — reasoning/acting loop, pluggable LLM backend,
tool-use safety, and the debugging process of getting multi-step tool chains actually
correct, not just superficially working.*
