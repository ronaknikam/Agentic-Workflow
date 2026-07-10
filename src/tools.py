"""
tools.py
Each tool is a plain Python function plus a schema (name, description,
input_schema) in the same shape Claude's tool-use API expects, so the same
schema list can be handed directly to AnthropicLLM.plan_step without
translation. MockLLM ignores the schemas' fine detail and just matches on
name/keywords, but a real LLM backend relies on the descriptions to decide
when to call each tool.
"""
import ast
import operator
import json
import os
import cv2
import pytesseract

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# Calculator: a SAFE arithmetic evaluator (no eval() on arbitrary input --
# only a whitelisted set of AST node types are permitted).
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculator(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


# ---------------------------------------------------------------------------
# OCR tool
# ---------------------------------------------------------------------------
def extract_text(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}", "text": None}
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}", "text": None}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    return {"error": None, "text": text, "char_count": len(text)}


# ---------------------------------------------------------------------------
# Object counting tool (YOLOv8)
# ---------------------------------------------------------------------------
_detector_cache = {}


def count_objects(image_path: str, target_class: str = None) -> dict:
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    if "model" not in _detector_cache:
        from ultralytics import YOLO
        _detector_cache["model"] = YOLO("yolov8n.pt")
    model = _detector_cache["model"]

    img = cv2.imread(image_path)
    results = model.predict(img, conf=0.35, verbose=False)[0]

    counts = {}
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    if target_class:
        return {"error": None, "target_class": target_class,
                "count": counts.get(target_class, 0), "all_counts": counts}
    return {"error": None, "all_counts": counts, "total_objects": sum(counts.values())}


# ---------------------------------------------------------------------------
# Knowledge base lookup (keyword search over a small local JSON store --
# a minimal stand-in for a RAG retrieval step)
# ---------------------------------------------------------------------------
def knowledge_base_lookup(query: str) -> dict:
    kb_path = os.path.join(DATA_DIR, "knowledge_base.json")
    with open(kb_path) as f:
        kb = json.load(f)

    stopwords = {"what", "is", "the", "a", "an", "of", "in", "to", "and", "for",
                 "context", "does", "do", "explain", "define", "who", "are"}

    def normalize(words):
        # crude singular/plural folding: "agents" -> "agent"
        return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w not in stopwords}

    query_words = normalize(query.lower().split())

    best_match, best_score = None, 0
    for entry in kb:
        topic_words = normalize(entry["topic"].lower().split())
        content_words = normalize(entry["content"].lower().split())
        # topic matches count 3x -- a query mentioning the entry's own topic
        # words is a much stronger signal than incidental overlap in prose
        score = 3 * len(query_words & topic_words) + len(query_words & content_words)
        if score > best_score:
            best_score, best_match = score, entry

    if best_match and best_score > 0:
        return {"found": True, "topic": best_match["topic"], "content": best_match["content"]}
    return {"found": False, "content": None}


# ---------------------------------------------------------------------------
# Tool registry: name -> (callable, schema)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate a numeric arithmetic expression, e.g. '450 + 1200 + 800'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "extract_text",
        "description": "Run OCR on an image file to extract any text it contains.",
        "input_schema": {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"],
        },
    },
    {
        "name": "count_objects",
        "description": "Detect and count objects (people, vehicles, etc.) in an image using YOLOv8.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "target_class": {"type": "string", "description": "Optional: count only this COCO class, e.g. 'person'"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "knowledge_base_lookup",
        "description": "Look up a factual definition or explanation from the local knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_REGISTRY = {
    "calculator": calculator,
    "extract_text": extract_text,
    "count_objects": count_objects,
    "knowledge_base_lookup": knowledge_base_lookup,
}


def run_tool(name: str, arguments: dict) -> dict:
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return TOOL_REGISTRY[name](**arguments)
    except TypeError as e:
        return {"error": f"Invalid arguments for '{name}': {e}"}
    except Exception as e:
        return {"error": f"Tool '{name}' raised an exception: {e}"}
