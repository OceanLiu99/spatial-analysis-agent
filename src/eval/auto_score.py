
# python -m src.spatial_agent.eval.auto_score

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


# =========================
# Config
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DEFAULT_TRUTH_DIR = os.path.join(PROJECT_ROOT, "data", "truth")
DEFAULT_PRED_DIR = os.path.join(PROJECT_ROOT, "data", "pred")
DEFAULT_SCHEMA_PATH = os.path.join(PROJECT_ROOT, "schema.json")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "reports")

ALLOWED_TOOL_KEYS = ["tool", "operation", "name"]

# =========================
# Define commutable tool pairs (add more as needed)
# =========================

COMMUTABLE_PAIRS = {
    frozenset(["buffer", "select_features_using_attributes"]),
    # frozenset(["clip", "select_features_using_attributes"]),
}


# =========================
# Utilities
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([fn for fn in os.listdir(folder) if fn.lower().endswith(".json")])


def safe_get(d: Dict[str, Any], key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def norm_tool_name(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def get_tool(step: Dict[str, Any]) -> str:
    for k in ALLOWED_TOOL_KEYS:
        if k in step:
            return norm_tool_name(step.get(k))
    return ""


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = []
            for h in header:
                v = r.get(h, "")
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                s = str(v).replace('"', '""')
                vals.append(f'"{s}"')
            f.write(",".join(vals) + "\n")


# =========================
# Schema validation
# =========================

@dataclass
class SchemaIssue:
    level: str   # error | warning
    path: str
    message: str


def validate_with_jsonschema(instance: Any, schema: Dict[str, Any]) -> List[SchemaIssue]:
    issues: List[SchemaIssue] = []
    if not HAS_JSONSCHEMA:
        return [SchemaIssue("warning", "$", "jsonschema not installed, fallback validation used")]

    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(instance), key=lambda e: e.path):
        p = "$"
        for part in list(err.absolute_path):
            if isinstance(part, int):
                p += f"[{part}]"
            else:
                p += f".{part}"
        issues.append(SchemaIssue("error", p, err.message))
    return issues


def fallback_validate_minimal(workflow: Any) -> List[SchemaIssue]:
    issues: List[SchemaIssue] = []

    if not isinstance(workflow, dict):
        return [SchemaIssue("error", "$", "workflow root must be an object")]

    if "steps" not in workflow:
        issues.append(SchemaIssue("error", "$.steps", "missing required key: steps"))
        return issues

    steps = workflow.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        issues.append(SchemaIssue("error", "$.steps", "steps must be a non-empty list"))
        return issues

    for i, step in enumerate(steps):
        p = f"$.steps[{i}]"
        if not isinstance(step, dict):
            issues.append(SchemaIssue("error", p, "step must be an object"))
            continue

        tool = get_tool(step)
        if not tool:
            issues.append(SchemaIssue("error", p, "missing tool field"))
        if "inputs" not in step:
            issues.append(SchemaIssue("warning", p, "missing inputs"))
        if "outputs" not in step:
            issues.append(SchemaIssue("warning", p, "missing outputs"))
        if "parameters" not in step:
            issues.append(SchemaIssue("warning", p, "missing parameters"))

    return issues


def validate_schema(workflow: Any, schema: Optional[Dict[str, Any]]) -> List[SchemaIssue]:
    if schema is not None:
        issues = validate_with_jsonschema(workflow, schema)
        if HAS_JSONSCHEMA:
            return issues
        # if jsonschema not available, fall through to minimal
        issues.extend(fallback_validate_minimal(workflow))
        return issues

    return fallback_validate_minimal(workflow)


# =========================
# Semantic metrics
# =========================

def extract_tool_sequence(workflow: Dict[str, Any]) -> List[str]:
    steps = workflow.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [get_tool(s) for s in steps if isinstance(s, dict)]


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def tool_sequence_similarity(truth: List[str], pred: List[str]) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    l = lcs_length(truth, pred)
    denom = max(len(truth), len(pred))
    return l / denom if denom else 0.0


def find_buffer_steps(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    steps = workflow.get("steps", [])
    if not isinstance(steps, list):
        return out
    for s in steps:
        if not isinstance(s, dict):
            continue
        if get_tool(s) == "buffer":
            out.append(s)
    return out


def get_buffer_distance_unit(step: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    params = step.get("parameters", {})
    if not isinstance(params, dict):
        return None, None

    dist = params.get("distance")
    unit = params.get("unit")

    if isinstance(dist, dict):
        v = dist.get("value")
        u = dist.get("unit")
        try:
            v = float(v) if v is not None else None
        except Exception:
            v = None
        u = str(u).strip().lower() if u is not None else None
        return v, u

    try:
        v = float(dist) if dist is not None else None
    except Exception:
        v = None
    u = str(unit).strip().lower() if unit is not None else None
    return v, u


def buffer_param_match_score(truth_wf: Dict[str, Any], pred_wf: Dict[str, Any]) -> Tuple[float, List[str]]:
    t_steps = find_buffer_steps(truth_wf)
    p_steps = find_buffer_steps(pred_wf)

    if not t_steps and not p_steps:
        return 1.0, []
    if not t_steps or not p_steps:
        return 0.0, ["buffer step count mismatch"]

    k = min(len(t_steps), len(p_steps))
    matches = 0
    issues: List[str] = []

    for i in range(k):
        t_v, t_u = get_buffer_distance_unit(t_steps[i])
        p_v, p_u = get_buffer_distance_unit(p_steps[i])

        ok = True
        if t_v is None or p_v is None:
            ok = False
            issues.append(f"buffer[{i}] missing distance value")
        else:
            if abs(t_v - p_v) > 1e-6:
                ok = False
                issues.append(f"buffer[{i}] distance mismatch truth={t_v} pred={p_v}")

        if (t_u is None) != (p_u is None):
            ok = False
            issues.append(f"buffer[{i}] unit missing truth={t_u} pred={p_u}")
        elif t_u is not None and p_u is not None and t_u != p_u:
            ok = False
            issues.append(f"buffer[{i}] unit mismatch truth={t_u} pred={p_u}")

        if ok:
            matches += 1

    if len(t_steps) != len(p_steps):
        issues.append(f"buffer step count mismatch truth={len(t_steps)} pred={len(p_steps)}")

    return matches / max(len(t_steps), len(p_steps)), issues


LAYER_KEYS = {
    "source_layer", "target_layer", "join_layer", "clip_layer", "overlay_layer",
    "input_layer", "layer", "layers"
}

def extract_layers(workflow: Dict[str, Any]) -> List[str]:
    layers: List[str] = []

    # from top-level inputs list if present
    inputs = workflow.get("inputs")
    if isinstance(inputs, list):
        for it in inputs:
            if isinstance(it, dict) and "name" in it:
                layers.append(str(it["name"]).strip())

    steps = workflow.get("steps", [])
    if isinstance(steps, list):
        for s in steps:
            if not isinstance(s, dict):
                continue
            ins = s.get("inputs", {})
            if isinstance(ins, dict):
                for k, v in ins.items():
                    if k in LAYER_KEYS:
                        if isinstance(v, list):
                            for x in v:
                                if x is not None:
                                    layers.append(str(x).strip())
                        elif v is not None:
                            layers.append(str(v).strip())
            # some schemas put layer refs in parameters
            params = s.get("parameters", {})
            if isinstance(params, dict):
                for k, v in params.items():
                    if k in ("in_layer", "in_features", "in_table", "features", "feature_layer"):
                        if v is not None:
                            layers.append(str(v).strip())

    layers = [x for x in layers if x]
    return layers


FIELD_KEY_HINTS = {
    "required_fields", "optional_fields", "fields_used", "field", "target_field", "join_fields"
}

def extract_fields_from_where(where: str) -> List[str]:
    # very simple heuristic
    # remove quoted strings
    tmp = re.sub(r"'[^']*'", " ", where)
    tmp = re.sub(r'"[^"]*"', " ", tmp)

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tmp)
    keywords = {
        "and", "or", "not", "in", "is", "null", "like", "between",
        "true", "false"
    }
    # also filter common operators that might be captured
    return [t for t in tokens if t.lower() not in keywords]


def extract_fields(workflow: Dict[str, Any]) -> List[str]:
    fields: List[str] = []

    inputs = workflow.get("inputs")
    if isinstance(inputs, list):
        for it in inputs:
            if not isinstance(it, dict):
                continue
            for k in ("required_fields", "optional_fields"):
                v = it.get(k)
                if isinstance(v, list):
                    fields.extend([str(x).strip() for x in v if x])

    steps = workflow.get("steps", [])
    if isinstance(steps, list):
        for s in steps:
            if not isinstance(s, dict):
                continue

            params = s.get("parameters", {})
            if isinstance(params, dict):
                for k, v in params.items():
                    if k in FIELD_KEY_HINTS:
                        if isinstance(v, list):
                            fields.extend([str(x).strip() for x in v if x])
                        elif v is not None:
                            fields.append(str(v).strip())

                where = params.get("where")
                if isinstance(where, str) and where.strip():
                    fields.extend(extract_fields_from_where(where))

            # sometimes fields appear in preconditions
            pre = s.get("preconditions")
            if isinstance(pre, list):
                for cond in pre:
                    if isinstance(cond, dict) and "field" in cond and cond["field"]:
                        fields.append(str(cond["field"]).strip())

    fields = [x for x in fields if x]
    # normalize
    return [x.strip() for x in fields if x.strip()]


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set([x.lower() for x in a if x])
    sb = set([x.lower() for x in b if x])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# =========================
# Step-level diff helpers
# =========================

def step_tool_diffs(truth_wf: Dict[str, Any], pred_wf: Dict[str, Any]) -> List[str]:
    t_steps = truth_wf.get("steps", [])
    p_steps = pred_wf.get("steps", [])
    if not isinstance(t_steps, list) or not isinstance(p_steps, list):
        return ["steps missing or not list"]

    diffs: List[str] = []

    def tool_at(steps: List[Any], idx: int) -> str:
        if idx < 0 or idx >= len(steps):
            return ""
        s = steps[idx]
        return get_tool(s) if isinstance(s, dict) else ""

    i = 0
    n = max(len(t_steps), len(p_steps))

    while i < n:
        if i >= len(t_steps):
            diffs.append(f"extra step at index {i}, pred={tool_at(p_steps, i)}")
            i += 1
            continue

        if i >= len(p_steps):
            diffs.append(f"missing step at index {i}, truth={tool_at(t_steps, i)}")
            i += 1
            continue

        tt = tool_at(t_steps, i)
        pt = tool_at(p_steps, i)

        if tt == pt:
            i += 1
            continue

        # ignore adjacent swap if the pair is commutable
        tt_next = tool_at(t_steps, i + 1)
        pt_next = tool_at(p_steps, i + 1)

        if tt_next and pt_next:
            if tt == pt_next and tt_next == pt:
                if frozenset([tt, tt_next]) in COMMUTABLE_PAIRS:
                    i += 2
                    continue

        diffs.append(f"step[{i}] tool mismatch, truth={tt}, pred={pt}")
        i += 1

    return diffs


# =========================
# Scoring
# =========================

@dataclass
class TaskScore:
    task_id: str
    truth_file: str
    pred_file: str

    schema_errors: int
    schema_warnings: int
    schema_issue_samples: List[Dict[str, Any]]

    tool_seq_similarity: float
    buffer_param_score: float
    layer_jaccard: float
    field_jaccard: float

    overall: float

    error_notes: List[str]


def score_one(task_id: str, truth_wf: Dict[str, Any], pred_wf: Dict[str, Any],
              schema: Optional[Dict[str, Any]]) -> TaskScore:

    schema_issues = validate_schema(pred_wf, schema)
    schema_errors = sum(1 for x in schema_issues if x.level == "error")
    schema_warnings = sum(1 for x in schema_issues if x.level == "warning")
    schema_issue_samples = [asdict(x) for x in schema_issues[:10]]

    t_tools = extract_tool_sequence(truth_wf)
    p_tools = extract_tool_sequence(pred_wf)
    tool_sim = tool_sequence_similarity(t_tools, p_tools)

    buf_score, buf_issues = buffer_param_match_score(truth_wf, pred_wf)

    t_layers = extract_layers(truth_wf)
    p_layers = extract_layers(pred_wf)
    layer_score = jaccard(t_layers, p_layers)

    t_fields = extract_fields(truth_wf)
    p_fields = extract_fields(pred_wf)
    field_score = jaccard(t_fields, p_fields)

    # overall score weights
    # schema errors zero out overall because it is not usable
    if schema_errors > 0:
        overall = 0.0
    else:
        overall = (
            0.35 * tool_sim +
            0.25 * buf_score +
            0.20 * layer_score +
            0.20 * field_score
        )

    notes: List[str] = []
    # add concrete diffs
    notes.extend(step_tool_diffs(truth_wf, pred_wf)[:10])
    notes.extend(buf_issues[:10])

    # show missing layers and fields in pred relative to truth
    missing_layers = sorted(set([x.lower() for x in t_layers]) - set([x.lower() for x in p_layers]))
    extra_layers = sorted(set([x.lower() for x in p_layers]) - set([x.lower() for x in t_layers]))
    if missing_layers:
        notes.append("missing layers in pred: " + ", ".join(missing_layers[:10]))
    if extra_layers:
        notes.append("extra layers in pred: " + ", ".join(extra_layers[:10]))

    missing_fields = sorted(set([x.lower() for x in t_fields]) - set([x.lower() for x in p_fields]))
    extra_fields = sorted(set([x.lower() for x in p_fields]) - set([x.lower() for x in t_fields]))
    if missing_fields:
        notes.append("missing fields in pred: " + ", ".join(missing_fields[:10]))
    if extra_fields:
        notes.append("extra fields in pred: " + ", ".join(extra_fields[:10]))

    return TaskScore(
        task_id=task_id,
        truth_file="",
        pred_file="",
        schema_errors=schema_errors,
        schema_warnings=schema_warnings,
        schema_issue_samples=schema_issue_samples,
        tool_seq_similarity=round(tool_sim, 4),
        buffer_param_score=round(buf_score, 4),
        layer_jaccard=round(layer_score, 4),
        field_jaccard=round(field_score, 4),
        overall=round(overall, 4),
        error_notes=notes
    )


def match_pairs(truth_dir: str, pred_dir: str) -> List[Tuple[str, str, str]]:
    truth_files = list_json_files(truth_dir)
    pred_files = list_json_files(pred_dir)

    if not truth_files:
        raise RuntimeError(f"No json files found in truth_dir: {truth_dir}")
    if not pred_files:
        raise RuntimeError(f"No json files found in pred_dir: {pred_dir}")

    def extract_id(fn: str, patterns: List[str]) -> Optional[str]:
        for p in patterns:
            m = re.match(p, fn, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    truth_patterns = [
        r"^task_(\d+)\.json$",
        r"^task(\d+)\.json$",
        r"^truth_task_(\d+)\.json$",
        r"^truth_task(\d+)\.json$"
    ]

    pred_patterns = [
        r"^pred_task_(\d+)\.json$",
        r"^pred_task(\d+)\.json$",
        r"^pred-task-(\d+)\.json$"
    ]

    truth_map: Dict[str, str] = {}
    pred_map: Dict[str, str] = {}

    for fn in truth_files:
        tid = extract_id(fn, truth_patterns)
        if tid is not None:
            truth_map[tid] = fn

    for fn in pred_files:
        tid = extract_id(fn, pred_patterns)
        if tid is not None:
            pred_map[tid] = fn

    common_ids = sorted(set(truth_map.keys()) & set(pred_map.keys()), key=lambda x: int(x))

    pairs: List[Tuple[str, str, str]] = []
    for tid in common_ids:
        task_id = f"task_{tid}"
        t_path = os.path.join(truth_dir, truth_map[tid])
        p_path = os.path.join(pred_dir, pred_map[tid])
        pairs.append((task_id, t_path, p_path))

    if not pairs:
        truth_samples = truth_files[:10]
        pred_samples = pred_files[:10]
        raise RuntimeError(
            "No matching pairs found.\n"
            f"truth_dir={truth_dir}\n"
            f"pred_dir={pred_dir}\n"
            f"truth files sample={truth_samples}\n"
            f"pred files sample={pred_samples}\n"
            "Expected patterns:\n"
            "truth: task_1.json or task1.json or truth_task_1.json\n"
            "pred: pred_task_1.json or pred_task1.json"
        )

    return pairs


def main(truth_dir: str = DEFAULT_TRUTH_DIR,
         pred_dir: str = DEFAULT_PRED_DIR,
         schema_path: str = DEFAULT_SCHEMA_PATH,
         report_dir: str = DEFAULT_REPORT_DIR) -> None:
    
    print(f"Truth directory: {truth_dir}")
    print(f"Prediction directory: {pred_dir}")


    schema = None
    if schema_path and os.path.isfile(schema_path):
        try:
            schema = load_json(schema_path)
        except Exception:
            schema = None

    pairs = match_pairs(truth_dir, pred_dir)
    if not pairs:
        raise RuntimeError("No matching json files found. Truth and pred must share filenames.")

    scores: List[TaskScore] = []
    csv_rows: List[Dict[str, Any]] = []

    for task_id, t_path, p_path in pairs:
        truth_wf = load_json(t_path)
        pred_wf = load_json(p_path)
        s = score_one(task_id, truth_wf, pred_wf, schema)
        s.truth_file = os.path.basename(t_path)
        s.pred_file = os.path.basename(p_path)
        scores.append(s)

        csv_rows.append({
            "task_id": s.task_id,
            "truth_file": s.truth_file,
            "pred_file": s.pred_file,
            "schema_errors": s.schema_errors,
            "schema_warnings": s.schema_warnings,
            "tool_seq_similarity": s.tool_seq_similarity,
            "buffer_param_score": s.buffer_param_score,
            "layer_jaccard": s.layer_jaccard,
            "field_jaccard": s.field_jaccard,
            "overall": s.overall,
            "top_notes": " | ".join(s.error_notes[:3])
        })

        per_task_path = os.path.join(report_dir, "per_task", f"{task_id}.json")
        write_json(per_task_path, asdict(s))

    summary = {
        "count": len(scores),
        "avg_overall": round(sum(x.overall for x in scores) / max(1, len(scores)), 4),
        "avg_tool_seq_similarity": round(sum(x.tool_seq_similarity for x in scores) / max(1, len(scores)), 4),
        "avg_buffer_param_score": round(sum(x.buffer_param_score for x in scores) / max(1, len(scores)), 4),
        "avg_layer_jaccard": round(sum(x.layer_jaccard for x in scores) / max(1, len(scores)), 4),
        "avg_field_jaccard": round(sum(x.field_jaccard for x in scores) / max(1, len(scores)), 4)
    }

    write_json(os.path.join(report_dir, "summary.json"), summary)
    write_csv(
        os.path.join(report_dir, "summary.csv"),
        csv_rows,
        header=[
            "task_id", "truth_file", "pred_file",
            "schema_errors", "schema_warnings",
            "tool_seq_similarity", "buffer_param_score",
            "layer_jaccard", "field_jaccard",
            "overall", "top_notes"
        ]
    )

    print("Done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report folder: {report_dir}")


if __name__ == "__main__":
    main()
