"""
memory.py
Lightweight persistent memory: each completed agent run is appended to a
JSON-backed log, so past tasks/answers can be looked back on across
process restarts. Deliberately simple (no vector store) -- the point is
demonstrating state persistence across runs, not building a production
memory system.
"""
import json
import os
import time


class AgentMemory:
    def __init__(self, path: str = None):
        self.path = path or os.path.join(os.path.dirname(__file__), "..", "data", "memory.json")
        if not os.path.exists(self.path):
            self._write([])

    def _read(self):
        with open(self.path) as f:
            return json.load(f)

    def _write(self, records):
        with open(self.path, "w") as f:
            json.dump(records, f, indent=2)

    def record(self, task: str, final_answer: str, num_steps: int):
        records = self._read()
        records.append({
            "timestamp": time.time(),
            "task": task,
            "final_answer": final_answer,
            "num_steps": num_steps,
        })
        self._write(records)

    def recent(self, n: int = 5):
        return self._read()[-n:]

    def clear(self):
        self._write([])
