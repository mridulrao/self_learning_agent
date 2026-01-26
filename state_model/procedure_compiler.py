#!/usr/bin/env python3
"""
procedure_compiler.py

LLM-powered ProcedureNode builder (KB doc -> procedure_nodes)

Modes:
1) Single-doc mode (same as before):
   python procedure_compiler.py --doc-id kb-123 --title "..." --content-file doc.txt --out out.json

2) CSV dataset mode (NEW):
   python procedure_compiler.py --csv /path/to/knowledge_items.csv --out out_nodes.jsonl

CSV mode reads each row as a KB document (doc_id/title/content), compiles procedure nodes,
and writes output as JSONL by default (one ProcedureNode per line).

Requirements:
  pip install openai pydantic python-dotenv

Env:
  OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

logger = logging.getLogger("procedure_compiler")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


RiskLevel = Literal["low", "medium", "high"]
StepKind = Literal["instruction", "check", "decision", "escalation"]


# -----------------------------
# Models
# -----------------------------

class ProcedureStep(BaseModel):
    step_id: str = Field(..., description="Stable ID within this procedure, like 's1', 's2'")
    kind: StepKind = Field("instruction", description="instruction/check/decision/escalation")
    instruction: str = Field(..., min_length=1, description="What the user/agent should do")
    expected_observation: Optional[str] = Field(
        None, description="What should be observed after this step (UI, log line, success condition)"
    )
    requires_admin: bool = Field(False, description="True if step likely needs admin privileges")
    optional: bool = Field(False, description="True if step is optional")
    safety_notes: Optional[str] = Field(None, description="Warnings or safety considerations")

    @field_validator("step_id")
    @classmethod
    def _validate_step_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("step_id must be alphanumeric/underscore/dash")
        return v


class Precondition(BaseModel):
    slot: str = Field(..., description="Slot name, e.g. 'os', 'product', 'version'")
    values: List[str] = Field(default_factory=list, description="Allowed values for the slot")

    @field_validator("slot")
    @classmethod
    def _validate_slot(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v.strip()):
            raise ValueError("precondition.slot must be url-safe-ish")
        return v.strip()


class ProcedureNode(BaseModel):
    procedure_id: str = Field(..., description="Stable ID (computed deterministically)")
    doc_id: str = Field(..., description="Source document ID")
    doc_title: str = Field(..., description="Source document title")
    title: str = Field(..., description="Human-readable procedure title")
    summary: str = Field(..., description="1-3 sentence summary of when to use this procedure")

    preconditions: List[Precondition] = Field(
        default_factory=list,
        description="List of slot/value constraints (e.g., [{'slot':'os','values':['windows']}])",
    )
    required_slots: List[str] = Field(
        default_factory=list,
        description="Slots required before acting (e.g., ['os','product','error_signature'])"
    )
    signatures: List[str] = Field(
        default_factory=list,
        description="Error strings/codes/regex patterns OR soft signatures to recognize applicability"
    )
    steps: List[ProcedureStep] = Field(..., min_length=1, description="Ordered steps")
    verify_prompts: List[str] = Field(default_factory=list, description="Short questions/checks to confirm success")
    escalate_requirements: List[str] = Field(
        default_factory=list,
        description="If unresolved, what evidence to collect (logs, screenshots, versions, timestamps, device mgmt status)"
    )
    risk: RiskLevel = Field("low", description="Risk level of the procedure (low/medium/high)")
    tags: List[str] = Field(default_factory=list, description="Optional tags (network/auth/install/etc.)")

    @field_validator("procedure_id")
    @classmethod
    def _validate_procedure_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("procedure_id must be URL/filename-safe")
        return v


class ProcedureNodesResponse(BaseModel):
    procedure_nodes: List[ProcedureNode] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# -----------------------------
# JSON Schema for Structured Outputs
# -----------------------------

def procedure_nodes_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "procedure_nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "procedure_id": {"type": "string"},
                        "doc_id": {"type": "string"},
                        "doc_title": {"type": "string"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "preconditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "slot": {"type": "string"},
                                    "values": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["slot", "values"],
                            },
                        },
                        "required_slots": {"type": "array", "items": {"type": "string"}},
                        "signatures": {"type": "array", "items": {"type": "string"}},
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "step_id": {"type": "string"},
                                    "kind": {"type": "string", "enum": ["instruction", "check", "decision", "escalation"]},
                                    "instruction": {"type": "string"},
                                    "expected_observation": {"type": ["string", "null"]},
                                    "requires_admin": {"type": "boolean"},
                                    "optional": {"type": "boolean"},
                                    "safety_notes": {"type": ["string", "null"]},
                                },
                                "required": [
                                    "step_id",
                                    "kind",
                                    "instruction",
                                    "expected_observation",
                                    "requires_admin",
                                    "optional",
                                    "safety_notes",
                                ],
                            }
                        },
                        "verify_prompts": {"type": "array", "items": {"type": "string"}},
                        "escalate_requirements": {"type": "array", "items": {"type": "string"}},
                        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "procedure_id",
                        "doc_id",
                        "doc_title",
                        "title",
                        "summary",
                        "preconditions",
                        "required_slots",
                        "signatures",
                        "steps",
                        "verify_prompts",
                        "escalate_requirements",
                        "risk",
                        "tags",
                    ],
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["procedure_nodes", "warnings"],
    }


# -----------------------------
# Helpers
# -----------------------------

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def stable_procedure_id(doc_id: str, title: str, summary: str) -> str:
    base = f"{doc_id}::{title.strip().lower()}::{summary.strip().lower()[:120]}"
    return f"proc:{sha1_hex(base)[:16]}"

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "procedure"

def extract_output_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text
    try:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""

def chunk_text(text: str, max_chars: int = 18000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    headings = [m.start() for m in re.finditer(r"(?m)^(#{1,6}\s+.+)$", text)]
    if len(headings) <= 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    idxs = headings + [len(text)]
    chunks: List[str] = []
    cur = ""
    for i in range(len(idxs) - 1):
        seg = text[idxs[i]:idxs[i + 1]]
        if len(cur) + len(seg) <= max_chars:
            cur += seg
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = seg
    if cur.strip():
        chunks.append(cur.strip())

    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            final.extend([c[i:i + max_chars] for i in range(0, len(c), max_chars)])
    return final

def dedupe_procedure_nodes(nodes: List[ProcedureNode]) -> List[ProcedureNode]:
    seen: Dict[str, ProcedureNode] = {}
    for n in nodes:
        sig = "||".join([s.strip().lower() for s in (n.signatures or [])[:6]])
        steps = "||".join([(st.instruction.strip().lower()) for st in (n.steps or [])[:2]])
        key = sha1_hex(f"{n.doc_id}::{n.title.strip().lower()}::{sig}::{steps}")
        if key not in seen:
            seen[key] = n
    return list(seen.values())

def split_multi_action_step(step: ProcedureStep) -> List[ProcedureStep]:
    instr = (step.instruction or "").strip()
    if not instr:
        return [step]
    if len(instr) < 140:
        return [step]

    parts = re.split(r"(?:\.\s+|\n+|;\s+|,\s+then\s+| then\s+)", instr, flags=re.IGNORECASE)
    parts = [p.strip(" -•\t") for p in parts if p and p.strip(" -•\t")]
    if len(parts) < 2 or len(parts) > 6:
        return [step]

    out: List[ProcedureStep] = []
    base_id = slugify(step.step_id)
    for i, p in enumerate(parts, start=1):
        out.append(
            ProcedureStep(
                step_id=f"{base_id}-{i}",
                kind=step.kind,
                instruction=p if p.endswith(".") else p,
                expected_observation=None if i < len(parts) else step.expected_observation,
                requires_admin=step.requires_admin,
                optional=step.optional,
                safety_notes=step.safety_notes,
            )
        )
    return out

def normalize_preconditions(preconditions: List[Precondition], required_slots: List[str]) -> Tuple[List[Precondition], List[str]]:
    drop_slots = {"issue_status"}
    cleaned: List[Precondition] = []
    for pc in preconditions or []:
        slot = (pc.slot or "").strip()
        if slot in drop_slots:
            continue
        values = [v.strip() for v in (pc.values or []) if v and v.strip()]
        if not slot:
            continue
        cleaned.append(Precondition(slot=slot, values=values))

    req = [s for s in (required_slots or []) if s not in drop_slots]
    seen = set()
    req2: List[str] = []
    for s in req:
        if s not in seen:
            seen.add(s)
            req2.append(s)

    seen_pc = set()
    pc2: List[Precondition] = []
    for pc in cleaned:
        key = (pc.slot, tuple(pc.values))
        if key not in seen_pc:
            seen_pc.add(key)
            pc2.append(pc)

    return pc2, req2

def ensure_soft_signatures(node: ProcedureNode) -> None:
    if node.signatures and any(s.strip() for s in node.signatures):
        return

    t = (node.title or "").lower()
    tags = [x.lower() for x in (node.tags or [])]
    text = " ".join([t] + tags)

    soft: List[str] = []
    if "plugin" in text or "add-in" in text or "addins" in text:
        soft.extend(["plugin not working", "add-in not working", "plugin missing or disabled"])

    proper = re.findall(r"[A-Z][a-zA-Z0-9]+", node.title or "")
    if proper:
        uniq: List[str] = []
        for p in proper:
            if p.lower() not in [u.lower() for u in uniq]:
                uniq.append(p)
            if len(uniq) >= 2:
                break
        for p in uniq:
            soft.append(f"{p} issue")

    if "excel" in text:
        soft.append("Excel add-in issue")
    if "powerpoint" in text:
        soft.append("PowerPoint add-in issue")

    if "escalation" in text or "escalate" in text:
        soft.append("escalate to support team")
        soft.append("unresolved after basic troubleshooting")

    out: List[str] = []
    seen = set()
    for s in soft:
        s2 = s.strip()
        if not s2:
            continue
        k = s2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s2)
        if len(out) >= 5:
            break

    node.signatures = out

def ensure_observable_verify_prompts(node: ProcedureNode) -> None:
    v = [x.strip() for x in (node.verify_prompts or []) if x and x.strip()]
    if not v:
        v = ["Did the issue resolve after completing the steps?"]

    text = (node.title + " " + " ".join(node.tags or [])).lower()
    has_observable = any(("confirm" in x.lower() or "verify" in x.lower() or "check" in x.lower()) for x in v)
    if ("plugin" in text or "add-in" in text) and not has_observable:
        v.append("Verify the plugin/add-in is enabled and visible (e.g., the tab/ribbon appears).")

    out: List[str] = []
    seen = set()
    for x in v:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    node.verify_prompts = out

def postprocess_node(node: ProcedureNode, doc_id: str, doc_title: str) -> ProcedureNode:
    node.doc_id = doc_id
    node.doc_title = doc_title
    node.procedure_id = stable_procedure_id(doc_id, node.title, node.summary)

    node.preconditions, node.required_slots = normalize_preconditions(node.preconditions, node.required_slots)

    new_steps: List[ProcedureStep] = []
    for st in node.steps or []:
        new_steps.extend(split_multi_action_step(st))
    node.steps = new_steps or node.steps

    ensure_soft_signatures(node)
    ensure_observable_verify_prompts(node)

    tags: List[str] = []
    seen = set()
    for t in node.tags or []:
        t2 = t.strip()
        if not t2:
            continue
        k = t2.lower()
        if k in seen:
            continue
        seen.add(k)
        tags.append(t2)
    node.tags = tags

    node.signatures = [s.strip() for s in (node.signatures or []) if s and s.strip()]
    return node


# -----------------------------
# LLM Compiler
# -----------------------------

@dataclass
class CompileConfig:
    model: str = "gpt-4.1"
    max_output_tokens: int = 1800
    temperature: float = 0.2
    max_retries: int = 2
    chunk_max_chars: int = 18000


SYSTEM_INSTRUCTIONS = """You are an expert IT support runbook compiler.

Task:
Convert the given knowledge base document into a set of *Procedure Nodes* suitable for an automated troubleshooting agent.

Rules:
- Output MUST be valid JSON that conforms to the provided JSON Schema.
- Create multiple procedure_nodes if the document contains multiple distinct procedures or variants (e.g., different OS, different error codes, different auth modes).
- Prefer smaller, reusable procedures over one giant node.
- Preconditions: represent as an ARRAY of {slot, values} objects. Include only if clearly implied by headings/text (os/product/version/admin/managed_device/network/etc.).
- Avoid meta slots like 'issue_status'. If you need to express sequencing, prefer 'troubleshooting_attempted' = ['true'].
- Required slots: list only the minimum gating slots needed before executing safely.
- Signatures: extract exact error strings/codes when present; if none exist, include 2-4 short "soft signatures" based on the issue (e.g., 'plugin not working', 'add-in missing').
- Steps: keep steps concrete and observable. Use kind=decision when the step branches on an observation.
- Verify prompts: include at least 1, and prefer an observable check (e.g., tab visible, feature works).
- Escalate requirements: list what evidence to gather if not resolved (logs, screenshots, versions, timestamps, device management status).
- Risk: mark 'high' if steps could cause data loss, lockout, or downtime; otherwise low/medium.
- Do not hallucinate tools or company-specific systems not present in the document.
"""


def compile_procedure_nodes(
    *,
    doc_id: str,
    title: str,
    content: str,
    config: Optional[CompileConfig] = None,
    client: Optional[OpenAI] = None,
) -> List[ProcedureNode]:
    cfg = config or CompileConfig()
    cli = client or OpenAI()

    schema = procedure_nodes_json_schema()
    chunks = chunk_text(content, max_chars=cfg.chunk_max_chars)

    all_nodes: List[ProcedureNode] = []
    warnings: List[str] = []

    for chunk_i, chunk in enumerate(chunks):
        user_prompt = f"""DOCUMENT METADATA:
doc_id: {doc_id}
doc_title: {title}
chunk: {chunk_i+1}/{len(chunks)}

DOCUMENT TEXT:
{chunk}
"""

        last_err: Optional[str] = None
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = cli.responses.create(
                    model=cfg.model,
                    input=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "procedure_nodes",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    temperature=cfg.temperature,
                    max_output_tokens=cfg.max_output_tokens,
                )

                raw = extract_output_text(resp)
                if not raw:
                    raise RuntimeError("Empty model output text")

                data = json.loads(raw)
                parsed = ProcedureNodesResponse.model_validate(data)

                for n in parsed.procedure_nodes:
                    all_nodes.append(postprocess_node(n, doc_id=doc_id, doc_title=title))

                warnings.extend(parsed.warnings)
                last_err = None
                break

            except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < cfg.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    user_prompt = (
                        user_prompt
                        + "\n\nREPAIR INSTRUCTIONS:\n"
                          "- Your previous output was invalid or non-conformant.\n"
                          "- Output ONLY JSON matching the schema. No markdown, no extra text.\n"
                          "- Preconditions must be an array of {slot, values}. Avoid 'issue_status'.\n"
                          "- If no explicit error codes, add 2-4 soft signatures.\n"
                          "- Ensure steps array has at least 1 step.\n"
                    )
                    continue
                warnings.append(f"Failed to compile chunk {chunk_i+1}/{len(chunks)}: {last_err}")

    all_nodes = dedupe_procedure_nodes(all_nodes)
    fixed: List[ProcedureNode] = []
    for n in all_nodes:
        fixed.append(postprocess_node(n, doc_id=doc_id, doc_title=title))
    return fixed


# -----------------------------
# CSV ingestion + output (NEW)
# -----------------------------

def _infer_col(headers: List[str], preferred: List[str]) -> Optional[str]:
    hset = {h.strip(): h for h in headers}
    lower_map = {h.lower().strip(): h for h in headers}
    for p in preferred:
        if p in hset:
            return hset[p]
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    return None

def infer_csv_mapping(headers: List[str]) -> Dict[str, str]:
    """
    Tries to guess the id/title/content columns from common names.
    You can override via CLI flags.
    """
    id_col = _infer_col(headers, ["doc_id", "id", "kb_id", "article_id", "ticket_id", "knowledge_id"])
    title_col = _infer_col(headers, ["title", "doc_title", "subject", "headline", "problem", "issue"])
    # content may be spread; we'll pick a primary and optionally compose later
    content_col = _infer_col(headers, ["content", "text", "body", "article", "knowledge", "resolution", "description"])
    return {
        "id_col": id_col or "",
        "title_col": title_col or "",
        "content_col": content_col or "",
    }

def compose_content_from_row(row: Dict[str, str], *, primary: str, extras: List[str]) -> str:
    """
    Builds a 'document text' string from one or more columns.
    """
    parts: List[str] = []
    if primary and row.get(primary):
        parts.append(str(row.get(primary, "")).strip())

    for c in extras:
        v = str(row.get(c, "")).strip()
        if v and v not in parts:
            parts.append(f"{c}:\n{v}")

    return "\n\n".join([p for p in parts if p])

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def compile_csv_dataset(
    *,
    csv_path: str,
    out_path: str,
    config: Optional[CompileConfig] = None,
    model: Optional[str] = None,
    id_col: Optional[str] = None,
    title_col: Optional[str] = None,
    content_col: Optional[str] = None,
    content_extra_cols: Optional[List[str]] = None,
    start_row: int = 0,
    limit: Optional[int] = None,
    output_format: Literal["jsonl", "json"] = "jsonl",
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reads CSV and writes compiled ProcedureNodes.
    Returns a small stats dict.
    """
    cfg = config or CompileConfig()
    if model:
        cfg.model = model

    cli = OpenAI()

    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    progress_p = Path(progress_path) if progress_path else Path(out_path + ".progress.json")
    # if resuming, start_row can be overridden by progress
    if progress_p.exists():
        try:
            prog = json.loads(progress_p.read_text(encoding="utf-8"))
            if isinstance(prog, dict) and isinstance(prog.get("next_row"), int) and prog["next_row"] > start_row:
                start_row = int(prog["next_row"])
                logger.info("Resuming from progress file at row=%d", start_row)
        except Exception:
            pass

    compiled_nodes_total = 0
    docs_processed = 0
    docs_failed = 0
    warnings_total: List[str] = []

    # If output is json (array), we must store in memory => risky for large datasets
    json_accum: List[Dict[str, Any]] = []

    # Ensure clean output for fresh run in jsonl mode when starting at row 0
    if output_format == "jsonl" and start_row == 0 and Path(out_path).exists():
        logger.warning("Overwriting existing output JSONL: %s", out_path)
        Path(out_path).unlink()

    with csv_p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no headers/fieldnames")

        headers = list(reader.fieldnames)
        inferred = infer_csv_mapping(headers)

        use_id = id_col or (inferred["id_col"] or None)
        use_title = title_col or (inferred["title_col"] or None)
        use_content = content_col or (inferred["content_col"] or None)

        if not use_title:
            raise RuntimeError(
                f"Could not infer title column from headers. Pass --title-col. Headers={headers}"
            )
        if not use_content:
            raise RuntimeError(
                f"Could not infer content column from headers. Pass --content-col. Headers={headers}"
            )

        extra_cols = content_extra_cols or []
        logger.info("CSV mapping: id_col=%s title_col=%s content_col=%s extra_cols=%s", use_id, use_title, use_content, extra_cols)

        # Iterate rows
        for i, row in enumerate(reader):
            if i < start_row:
                continue
            if limit is not None and docs_processed >= limit:
                break

            # Construct doc fields
            doc_id = (row.get(use_id, "") if use_id else "").strip()
            if not doc_id:
                doc_id = f"csvrow:{i}"

            title = (row.get(use_title, "") or "").strip() or f"Untitled {doc_id}"
            content = compose_content_from_row(row, primary=use_content, extras=extra_cols)

            if not content.strip():
                docs_failed += 1
                warnings_total.append(f"Row {i}: empty content after composition (doc_id={doc_id})")
                # update progress
                progress_p.write_text(json.dumps({"next_row": i + 1}, indent=2), encoding="utf-8")
                continue

            try:
                nodes = compile_procedure_nodes(
                    doc_id=str(doc_id),
                    title=str(title),
                    content=str(content),
                    config=cfg,
                    client=cli,
                )

                # Write output
                records = []
                for n in nodes:
                    rec = n.model_dump()
                    # attach provenance useful for debugging
                    rec["_source"] = {"csv_path": str(csv_path), "row_index": i}
                    records.append(rec)

                if output_format == "jsonl":
                    write_jsonl(out_path, records)
                else:
                    json_accum.extend(records)

                compiled_nodes_total += len(records)
                docs_processed += 1

                # progress checkpoint
                progress_p.write_text(json.dumps({"next_row": i + 1}, indent=2), encoding="utf-8")

                if docs_processed % 10 == 0:
                    logger.info("Processed=%d nodes=%d last_row=%d", docs_processed, compiled_nodes_total, i)

            except Exception as e:
                docs_failed += 1
                msg = f"Row {i} (doc_id={doc_id}) failed: {type(e).__name__}: {e}"
                warnings_total.append(msg)
                logger.exception(msg)
                progress_p.write_text(json.dumps({"next_row": i + 1}, indent=2), encoding="utf-8")
                continue

    if output_format == "json":
        write_json(out_path, json_accum)

    # Write warnings alongside output
    warn_path = out_path + ".warnings.json"
    write_json(warn_path, {"warnings": warnings_total})

    stats = {
        "csv_path": csv_path,
        "out_path": out_path,
        "output_format": output_format,
        "docs_processed": docs_processed,
        "docs_failed": docs_failed,
        "procedure_nodes_written": compiled_nodes_total if output_format == "jsonl" else len(json_accum),
        "warnings_path": warn_path,
        "progress_path": str(progress_p),
        "model": cfg.model,
    }
    return stats


# -----------------------------
# CLI
# -----------------------------

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser(description="Compile KB docs into ProcedureNodes (single doc or CSV dataset).")

    # Single doc mode args
    ap.add_argument("--doc-id", type=str, default=None, help="Document ID (single-doc mode)")
    ap.add_argument("--title", type=str, default=None, help="Document title (single-doc mode)")
    ap.add_argument("--content", type=str, default=None, help="Document content (single-doc mode)")
    ap.add_argument("--content-file", type=str, default=None, help="Path to a text/markdown file for content (single-doc mode)")

    # CSV mode args
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV file (dataset mode)")
    ap.add_argument("--id-col", type=str, default=None, help="CSV column name for doc_id (optional)")
    ap.add_argument("--title-col", type=str, default=None, help="CSV column name for title")
    ap.add_argument("--content-col", type=str, default=None, help="CSV column name for primary content/body")
    ap.add_argument("--content-extra-col", action="append", default=[], help="Additional columns to append into content (repeatable)")
    ap.add_argument("--start-row", type=int, default=0, help="Start row index for CSV processing (0-based)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    ap.add_argument("--output-format", type=str, default="jsonl", choices=["jsonl", "json"], help="Output format (jsonl recommended)")

    # Shared output/model args
    ap.add_argument("--out", type=str, required=True, help="Output path (.jsonl or .json)")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model override (default from CompileConfig)")
    ap.add_argument("--max-output-tokens", type=int, default=1800)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--chunk-max-chars", type=int, default=18000)
    ap.add_argument("--progress-path", type=str, default=None, help="Progress checkpoint file path (CSV mode)")

    args = ap.parse_args()

    cfg = CompileConfig(
        model=args.model or "gpt-4.1",
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        chunk_max_chars=args.chunk_max_chars,
    )

    # Decide mode
    if args.csv:
        stats = compile_csv_dataset(
            csv_path=args.csv,
            out_path=args.out,
            config=cfg,
            model=args.model,
            id_col=args.id_col,
            title_col=args.title_col,
            content_col=args.content_col,
            content_extra_cols=args.content_extra_col or [],
            start_row=args.start_row,
            limit=args.limit,
            output_format=args.output_format,
            progress_path=args.progress_path,
        )
        print(json.dumps(stats, indent=2))
        return

    # Single doc mode
    if not args.doc_id:
        raise SystemExit("Single-doc mode requires --doc-id (or use --csv for dataset mode).")
    if not args.title:
        raise SystemExit("Single-doc mode requires --title.")
    content = args.content
    if args.content_file:
        content = read_text_file(args.content_file)
    if not content:
        raise SystemExit("Single-doc mode requires --content or --content-file.")

    nodes = compile_procedure_nodes(doc_id=args.doc_id, title=args.title, content=content, config=cfg, client=OpenAI())
    out_records = [n.model_dump() for n in nodes]
    if args.out.endswith(".jsonl"):
        # overwrite
        Path(args.out).write_text("", encoding="utf-8")
        write_jsonl(args.out, out_records)
    else:
        write_json(args.out, out_records)
    print(f"Wrote {len(out_records)} procedure nodes to {args.out}")


if __name__ == "__main__":
    main()



'''
python procedure_compiler.py \
  --csv synthetic_knowledge_items.csv \
  --out ./out/procedure_nodes.jsonl \
  --output-format jsonl \
  --limit 200


python procedure_compiler.py \
  --csv synthetic_knowledge_items.csv \
  --title-col ki_topic \
  --content-col ki_text \
  --out ./out/procedure_nodes.jsonl \
  --output-format jsonl



'''