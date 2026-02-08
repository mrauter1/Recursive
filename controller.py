#!/usr/bin/env python3
"""
controller.py — BVH controller with:
- oracle-failure fix retries (max_agent_fix_retries)
- soft-allowed failing tests when tests were added/updated (partial states allowed)
- optional fuzzy review ("are changes correct/regression-free?") as a risk gate

Requires: python>=3.10, PyYAML (pip install pyyaml)
Note: Oracle commands run via bash/sh. On Windows, run under WSL or ensure bash/sh is available.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

try:
    import yaml  # type: ignore
except Exception:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    raise

# -----------------------------
# Defaults
# -----------------------------

DEFAULT_AGENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["progress", "blocked", "needs_review", "done"]},
        "summary": {"type": "string"},
        "changed_files": {"type": "array", "items": {"type": "string"}},
        "next_actions": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "plan_updates": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mark_done": {"type": "array", "items": {"type": "string"}},
                "mark_blocked": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
        },
    },
    "required": ["status", "summary", "changed_files", "next_actions", "risk_notes"],
}

# Fuzzy review output schema (controller-enforced)
DEFAULT_FUZZY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "syntactic_ok": {"type": "boolean"},
        "logical_ok": {"type": "boolean"},
        "regression_risk": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "notes": {"type": "array", "items": {"type": "string"}},
        "suggested_tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["syntactic_ok", "logical_ok", "regression_risk", "notes", "suggested_tests"],
}

# Oracle steps now support optional keys:
# - kind: "format"|"lint"|"typecheck"|"build"|"tests"|"other"
# - allow_fail_if_tests_changed: bool  (typically true for tests steps)
DEFAULT_ORACLE_YAML = {
    "steps": [
        {"name": "format", "cmd": "true", "required": False, "timeout_sec": 300, "kind": "format"},
        {"name": "lint", "cmd": "true", "required": True, "timeout_sec": 300, "kind": "lint"},
        {"name": "unit", "cmd": "true", "required": True, "timeout_sec": 900, "kind": "tests", "allow_fail_if_tests_changed": True},
    ]
}

DEFAULT_PLAN_JSON = {
    "milestones": [
        {
            "id": "m1",
            "title": "Define milestones",
            "acceptance": ["Replace this stub plan with real milestones."],
            "deps": [],
            "status": "todo",
        }
    ]
}

DEFAULT_STATE_JSON = {
    "max_iterations": 30,
    "max_minutes": 90,
    "max_failed_attempts_per_milestone": 5,
    "no_improve_limit": 3,
    "max_diff_lines": 400,
    "max_changed_files": 25,

    # NEW:
    "max_agent_fix_retries": 3,             # times to ask agent to fix oracle failures before hard failure
    "allow_failing_tests_if_tests_changed": True,
    "tests_path_markers": [                 # heuristic; customize per repo
        "/test/", "/tests/", "\\test\\", "\\tests\\",
        "_test.", ".test.", ".spec."
    ],

    # Guards:
    "forbid_agent_git_commits": True,
    "forbid_agent_branch_change": True,

    # Fuzzy review (optional):
    "enable_fuzzy_review": True,
    "fuzzy_review_blocks_high_risk_commit": True,

    "history": [],
    "started_at": None,   # informational only
    "best_score": None,   # stored as list
}

# -----------------------------
# Data structures
# -----------------------------


@dataclass(frozen=True)
class OracleStepResult:
    name: str
    cmd: str
    required: bool
    ok: bool
    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str
    timeout: bool
    kind: str
    allow_fail_if_tests_changed: bool


@dataclass(frozen=True)
class OracleRunResult:
    ok_required_strict: bool          # strict required: all required steps must pass
    ok_required_with_soft_tests: bool # required passes allowing soft-test failures (if eligible)
    step_results: List[OracleStepResult]
    score: Tuple[int, int, int, int]  # (required_passed, optional_passed, -failures, -runtime_sec)
    summary: str
    soft_failed_required_tests: bool  # True if tests step(s) failed but are soft-allowed


@dataclass
class Milestone:
    id: str
    title: str
    acceptance: List[str]
    deps: List[str]
    status: str  # todo|doing|done|blocked


# -----------------------------
# Utilities
# -----------------------------


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def have_executable(name: str) -> bool:
    try:
        cp = subprocess.run([name, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return cp.returncode == 0 or bool((cp.stdout or cp.stderr).strip())
    except Exception:
        return False


def run_cmd(
    cmd: List[str],
    cwd: Path,
    timeout_sec: Optional[int] = None,
    capture: bool = True,
    check: bool = False,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "text": True,
        "input": input_text,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if timeout_sec is not None:
        kwargs["timeout"] = timeout_sec
    return subprocess.run(cmd, **kwargs, check=check)


def _shell_cmd(cmd: str) -> List[str]:
    if have_executable("bash"):
        return ["bash", "-lc", cmd]
    if have_executable("sh"):
        return ["sh", "-lc", cmd]
    raise RuntimeError("No suitable shell found (bash/sh). On Windows, run under WSL or provide a shell.")


# -----------------------------
# Git helpers
# -----------------------------


def git_is_repo(repo: Path) -> bool:
    try:
        cp = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo, capture=True)
        return cp.returncode == 0 and (cp.stdout or "").strip() == "true"
    except Exception:
        return False


def git_current_branch(repo: Path) -> str:
    cp = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, capture=True)
    return (cp.stdout or "").strip()


def git_head(repo: Path) -> str:
    cp = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo, capture=True)
    return (cp.stdout or "").strip()


def git_has_dirty_tree(repo: Path) -> bool:
    cp = run_cmd(["git", "status", "--porcelain"], cwd=repo, capture=True)
    return bool((cp.stdout or "").strip())


def git_revert_workspace(repo: Path) -> None:
    run_cmd(["git", "reset", "--hard", "HEAD"], cwd=repo, capture=True)
    run_cmd(["git", "clean", "-fd"], cwd=repo, capture=True)


def git_commit(repo: Path, message: str) -> None:
    run_cmd(["git", "add", "-A"], cwd=repo, capture=True)
    cp = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo, capture=True)
    if not (cp.stdout or "").strip():
        return
    run_cmd(["git", "commit", "-m", message], cwd=repo, capture=True)


def git_diff_stats(repo: Path) -> Tuple[int, int]:
    """
    Returns (approx_changed_files, total_changed_lines_estimate) for both staged and unstaged.
    """
    def parse_numstat(txt: str) -> Tuple[int, int]:
        files = 0
        lines = 0
        for line in (txt or "").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            add_s, del_s = parts[0], parts[1]
            files += 1
            add = int(add_s) if add_s.isdigit() else 0
            dele = int(del_s) if del_s.isdigit() else 0
            lines += add + dele
        return files, lines

    wt = run_cmd(["git", "diff", "--numstat"], cwd=repo, capture=True).stdout or ""
    idx = run_cmd(["git", "diff", "--cached", "--numstat"], cwd=repo, capture=True).stdout or ""
    f1, l1 = parse_numstat(wt)
    f2, l2 = parse_numstat(idx)
    return (f1 + f2), (l1 + l2)


def git_changed_files(repo: Path) -> List[str]:
    """
    Both staged and unstaged changed file paths.
    """
    wt = run_cmd(["git", "diff", "--name-only"], cwd=repo, capture=True).stdout or ""
    idx = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo, capture=True).stdout or ""
    files = [f.strip() for f in (wt.splitlines() + idx.splitlines()) if f.strip()]
    # preserve order but unique
    seen = set()
    out: List[str] = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def git_unified_diff(repo: Path, max_chars: int = 120_000) -> str:
    """
    Diff including staged+unstaged. Truncated to max_chars for prompting.
    """
    wt = run_cmd(["git", "diff"], cwd=repo, capture=True).stdout or ""
    idx = run_cmd(["git", "diff", "--cached"], cwd=repo, capture=True).stdout or ""
    combined = ""
    if idx.strip():
        combined += "### STAGED DIFF\n" + idx + "\n"
    if wt.strip():
        combined += "### UNSTAGED DIFF\n" + wt + "\n"
    if len(combined) > max_chars:
        return combined[:max_chars] + "\n\n[diff truncated]\n"
    return combined


# -----------------------------
# .agent scaffolding
# -----------------------------


def ensure_file_defaults(agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "reports").mkdir(parents=True, exist_ok=True)

    prd = agent_dir / "prd.md"
    if not prd.exists():
        write_text(prd, "# Product / Task Definition\n\nDescribe what “done” means.\n")

    plan = agent_dir / "plan.json"
    if not plan.exists():
        write_json(plan, DEFAULT_PLAN_JSON)

    oracle = agent_dir / "oracle.yaml"
    if not oracle.exists():
        write_text(oracle, yaml.safe_dump(DEFAULT_ORACLE_YAML, sort_keys=False))

    state = agent_dir / "state.json"
    if not state.exists():
        write_json(state, DEFAULT_STATE_JSON)

    mem = agent_dir / "memory.md"
    if not mem.exists():
        write_text(mem, "# Memory\n\n(Keep this short: symptom → cause → fix pattern)\n")

    schema = agent_dir / "agent_output.schema.json"
    if not schema.exists():
        write_json(schema, DEFAULT_AGENT_SCHEMA)

    fuzzy_schema = agent_dir / "fuzzy_review.schema.json"
    if not fuzzy_schema.exists():
        write_json(fuzzy_schema, DEFAULT_FUZZY_SCHEMA)


# -----------------------------
# Plan handling
# -----------------------------


def load_milestones(plan_path: Path) -> List[Milestone]:
    plan = read_json(plan_path)
    ms: List[Milestone] = []
    for m in plan.get("milestones", []):
        ms.append(
            Milestone(
                id=str(m.get("id")),
                title=str(m.get("title", "")),
                acceptance=list(m.get("acceptance", [])),
                deps=list(m.get("deps", [])),
                status=str(m.get("status", "todo")),
            )
        )
    return ms


def save_milestones(plan_path: Path, milestones: List[Milestone]) -> None:
    data = {
        "milestones": [
            {
                "id": m.id,
                "title": m.title,
                "acceptance": m.acceptance,
                "deps": m.deps,
                "status": m.status,
            }
            for m in milestones
        ]
    }
    write_json(plan_path, data)


def milestone_map(milestones: List[Milestone]) -> Dict[str, Milestone]:
    return {m.id: m for m in milestones}


def select_next_milestone(milestones: List[Milestone]) -> Optional[Milestone]:
    mm = milestone_map(milestones)

    def deps_done(m: Milestone) -> bool:
        for d in m.deps:
            if d not in mm:
                return False
            if mm[d].status != "done":
                return False
        return True

    for m in milestones:
        if m.status == "doing" and deps_done(m):
            return m
    for m in milestones:
        if m.status == "todo" and deps_done(m):
            return m
    return None


# -----------------------------
# Agent output parsing/validation
# -----------------------------


def _extract_first_json_object(text: str) -> str:
    s = text.strip()
    if not s:
        raise ValueError("Empty agent output")

    # Try direct parse
    if s.startswith("{"):
        try:
            json.loads(s)
            return s
        except Exception:
            pass

    start = s.find("{")
    if start < 0:
        raise ValueError("Could not find '{' in agent output")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    json.loads(candidate)
                    return candidate

    raise ValueError("Could not extract a balanced JSON object from agent output")


def parse_agent_json(text: str) -> Dict[str, Any]:
    obj_text = _extract_first_json_object(text)
    data = json.loads(obj_text)
    if not isinstance(data, dict):
        raise ValueError("Agent output JSON must be an object")
    return data


def _fallback_validate_agent_shape(obj: Dict[str, Any]) -> None:
    def req(k: str, t: Any) -> None:
        if k not in obj:
            raise ValueError(f"Missing required key: {k}")
        if not isinstance(obj[k], t):
            raise ValueError(f"Key {k} must be {t}, got {type(obj[k])}")

    req("status", str)
    req("summary", str)
    req("changed_files", list)
    req("next_actions", list)
    req("risk_notes", list)

    if obj["status"] not in ("progress", "blocked", "needs_review", "done"):
        raise ValueError(f"Invalid status: {obj['status']}")

    for k in ("changed_files", "next_actions", "risk_notes"):
        for i, v in enumerate(obj[k]):
            if not isinstance(v, str):
                raise ValueError(f"{k}[{i}] must be string")

    if "plan_updates" in obj:
        if not isinstance(obj["plan_updates"], dict):
            raise ValueError("plan_updates must be object")
        pu = obj["plan_updates"]
        for opt in ("mark_done", "mark_blocked"):
            if opt in pu and not (isinstance(pu[opt], list) and all(isinstance(x, str) for x in pu[opt])):
                raise ValueError(f"plan_updates.{opt} must be list[str]")
        if "notes" in pu and not isinstance(pu["notes"], str):
            raise ValueError("plan_updates.notes must be string")


def validate_agent_shape(obj: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        print("Warning: 'jsonschema' not installed; using fallback validation.", file=sys.stderr)
        _fallback_validate_agent_shape(obj)
        return

    try:
        jsonschema.validate(instance=obj, schema=DEFAULT_AGENT_SCHEMA)
    except jsonschema.exceptions.ValidationError as exc:
        raise ValueError(f"Agent output does not match schema: {exc.message}") from exc


def _fallback_validate_fuzzy_shape(obj: Dict[str, Any]) -> None:
    def req(k: str, t: Any) -> None:
        if k not in obj:
            raise ValueError(f"Missing required key: {k}")
        if not isinstance(obj[k], t):
            raise ValueError(f"Key {k} must be {t}, got {type(obj[k])}")

    req("syntactic_ok", bool)
    req("logical_ok", bool)
    req("regression_risk", str)
    req("notes", list)
    req("suggested_tests", list)

    if obj["regression_risk"] not in ("low", "medium", "high", "unknown"):
        raise ValueError("regression_risk must be low|medium|high|unknown")

    for k in ("notes", "suggested_tests"):
        for i, v in enumerate(obj[k]):
            if not isinstance(v, str):
                raise ValueError(f"{k}[{i}] must be string")


def validate_fuzzy_shape(obj: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        print("Warning: 'jsonschema' not installed; using fallback validation.", file=sys.stderr)
        _fallback_validate_fuzzy_shape(obj)
        return

    try:
        jsonschema.validate(instance=obj, schema=DEFAULT_FUZZY_SCHEMA)
    except jsonschema.exceptions.ValidationError as exc:
        raise ValueError(f"Fuzzy review output does not match schema: {exc.message}") from exc


# -----------------------------
# Adapters
# -----------------------------


class AgentAdapter(Protocol):
    def run(self, prompt: str, repo: Path, out_dir: Path, schema_path: Path) -> Dict[str, Any]:
        ...


@dataclass
class CodexAdapter:
    sandbox: str = "workspace-write"
    ask_for_approval: str = "on-request"
    model: Optional[str] = None

    def run(self, prompt: str, repo: Path, out_dir: Path, schema_path: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        last_msg_path = out_dir / "agent_last_message.txt"

        base_cmd = ["codex", "exec"]
        if self.model:
            base_cmd += ["-c", f"model={self.model}"]
        base_cmd += ["--sandbox", self.sandbox]

        # Try with approval flag; fallback without if unsupported.
        cmd_try = base_cmd + [
            "--ask-for-approval", self.ask_for_approval,
            "--output-schema", str(schema_path),
            "--output-last-message", str(last_msg_path),
            "-"
        ]
        try:
            cp = run_cmd(cmd_try, cwd=repo, capture=True, input_text=prompt)
            if cp.returncode != 0:
                cmd_fallback = base_cmd + [
                    "--output-schema", str(schema_path),
                    "--output-last-message", str(last_msg_path),
                    "-"
                ]
                cp2 = run_cmd(cmd_fallback, cwd=repo, capture=True, input_text=prompt)
                if cp2.returncode != 0:
                    raise RuntimeError(f"codex exec failed:\n{cp2.stderr}\n{cp2.stdout}")
        except FileNotFoundError:
            raise RuntimeError("codex executable not found on PATH.")

        if not last_msg_path.exists():
            raise RuntimeError("Codex did not produce --output-last-message output.")

        raw = read_text(last_msg_path)
        obj = parse_agent_json(raw)
        return obj


@dataclass
class ClaudeAdapter:
    output_format: str = "json"
    continue_session: bool = False

    def run(self, prompt: str, repo: Path, out_dir: Path, schema_path: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "claude_output.json"

        cmd = ["claude", "-p", prompt, "--output-format", self.output_format]
        if self.continue_session:
            cmd.append("--continue")

        try:
            cp = run_cmd(cmd, cwd=repo, capture=True)
        except FileNotFoundError:
            raise RuntimeError("claude executable not found on PATH.")

        if cp.returncode != 0:
            raise RuntimeError(f"claude failed:\n{cp.stderr}\n{cp.stdout}")

        stdout = (cp.stdout or "").strip()
        if not stdout:
            raise RuntimeError("Claude returned empty output.")

        if self.output_format == "stream-json":
            lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
            last_valid_line = ""
            for line in reversed(lines):
                try:
                    json.loads(line)
                    last_valid_line = line
                    break
                except json.JSONDecodeError:
                    continue
            stdout = last_valid_line

        # Best-effort unwrap
        try:
            data = json.loads(stdout)
            if isinstance(data, dict) and "status" in data and "summary" in data:
                obj = data
            else:
                candidate = None
                if isinstance(data, dict):
                    for key in ("output", "message", "content", "text"):
                        if key in data:
                            candidate = data[key]
                            break
                if isinstance(candidate, str):
                    obj = parse_agent_json(candidate)
                elif isinstance(candidate, dict):
                    obj = candidate
                else:
                    obj = parse_agent_json(stdout)
        except json.JSONDecodeError:
            obj = parse_agent_json(stdout)

        write_text(out_path, json.dumps(obj, indent=2) + "\n")
        return obj


@dataclass
class CustomCommandAdapter:
    """
    Custom command template:
      --custom-cmd 'myagent --input {prompt_file} --output {out_file}'
    """
    template: str

    def run(self, prompt: str, repo: Path, out_dir: Path, schema_path: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = out_dir / "prompt.txt"
        out_file = out_dir / "agent_output.json"
        write_text(prompt_file, prompt)

        cmd_str = self.template.format(
            prompt_file=str(prompt_file),
            out_file=str(out_file),
            schema_file=str(schema_path),
        )

        cp = run_cmd(_shell_cmd(cmd_str), cwd=repo, capture=True)
        if cp.returncode != 0:
            raise RuntimeError(f"Custom command failed:\n{cp.stderr}\n{cp.stdout}")
        if not out_file.exists():
            raise RuntimeError("Custom command did not produce {out_file} output.")

        obj = parse_agent_json(read_text(out_file))
        return obj


# -----------------------------
# Tests-changed heuristic
# -----------------------------


def tests_changed(changed_files: List[str], markers: List[str]) -> bool:
    # simple substring markers (customizable per repo)
    for f in changed_files:
        lf = f.lower()
        for m in markers:
            if m.lower() in lf:
                return True
    return False


# -----------------------------
# Oracle runner with soft-allowed tests failures
# -----------------------------


def run_oracle(repo: Path, oracle_path: Path, allow_soft_tests: bool, tests_were_changed: bool) -> OracleRunResult:
    oracle_obj = yaml.safe_load(read_text(oracle_path)) or {}
    steps = oracle_obj.get("steps", [])
    results: List[OracleStepResult] = []

    required_passed = 0
    optional_passed = 0
    failures = 0
    total_runtime = 0.0

    ok_required_strict = True
    ok_required_with_soft_tests = True
    soft_failed_required_tests = False

    for s in steps:
        name = str(s.get("name", "step"))
        cmd = str(s.get("cmd", "true"))
        required = bool(s.get("required", False))
        kind = str(s.get("kind", "other"))
        allow_fail_if_tests_changed = bool(s.get("allow_fail_if_tests_changed", False))
        timeout_sec = s.get("timeout_sec", None)
        if timeout_sec is not None:
            try:
                timeout_sec = int(timeout_sec)
            except Exception:
                timeout_sec = None

        t0 = time.time()
        timed_out = False
        try:
            cp = run_cmd(_shell_cmd(cmd), cwd=repo, timeout_sec=timeout_sec, capture=True)
            exit_code = cp.returncode
            stdout = cp.stdout or ""
            stderr = cp.stderr or ""
        except subprocess.TimeoutExpired as te:
            timed_out = True
            exit_code = 124
            stdout = (te.stdout or "") if hasattr(te, "stdout") else ""
            stderr = (te.stderr or "") if hasattr(te, "stderr") else ""
        t1 = time.time()

        duration = t1 - t0
        total_runtime += duration
        ok = (exit_code == 0) and (not timed_out)

        # Strict required
        if required and not ok:
            ok_required_strict = False

        # Soft-allowed required tests:
        soft_allowed_here = (
            allow_soft_tests
            and tests_were_changed
            and required
            and kind == "tests"
            and allow_fail_if_tests_changed
        )
        if required and not ok and soft_allowed_here:
            soft_failed_required_tests = True
            # Required-with-soft-tests still considered OK.
        elif required and not ok and not soft_allowed_here:
            ok_required_with_soft_tests = False

        if ok:
            if required:
                required_passed += 1
            else:
                optional_passed += 1
        else:
            failures += 1

        results.append(
            OracleStepResult(
                name=name,
                cmd=cmd,
                required=required,
                ok=ok,
                exit_code=exit_code,
                duration_sec=duration,
                stdout=stdout[-8000:],
                stderr=stderr[-8000:],
                timeout=timed_out,
                kind=kind,
                allow_fail_if_tests_changed=allow_fail_if_tests_changed,
            )
        )

        # Fail-fast only when required-with-soft-tests is definitively failed
        if required and not ok and not soft_allowed_here:
            break

    score = (
        required_passed,
        optional_passed,
        -failures,
        -int(round(total_runtime)),
    )
    summary = (
        f"required_passed={required_passed}, optional_passed={optional_passed}, "
        f"failures={failures}, runtime_sec={int(round(total_runtime))}"
    )
    return OracleRunResult(
        ok_required_strict=ok_required_strict,
        ok_required_with_soft_tests=ok_required_with_soft_tests,
        step_results=results,
        score=score,
        summary=summary,
        soft_failed_required_tests=soft_failed_required_tests,
    )


# -----------------------------
# Prompt builders
# -----------------------------


def build_prompt(
    prd: str,
    milestone: Milestone,
    memory: str,
    last_oracle: Optional[OracleRunResult],
    schema_hint: Dict[str, Any],
    extra_feedback: str = "",
) -> str:
    oracle_section = ""
    if last_oracle is not None:
        lines: List[str] = []
        for r in last_oracle.step_results:
            status = "PASS" if r.ok else "FAIL"
            req = "REQ" if r.required else "OPT"
            lines.append(f"- [{status}] ({req}) kind={r.kind} {r.name}: {r.cmd}")
            if not r.ok:
                if r.stderr.strip():
                    lines.append("  stderr (tail):")
                    lines.append(textwrap.indent(r.stderr.strip()[-1200:], "    "))
                if r.stdout.strip():
                    lines.append("  stdout (tail):")
                    lines.append(textwrap.indent(r.stdout.strip()[-1200:], "    "))
        oracle_section = "\n".join(lines)

    acceptance = "\n".join([f"- {a}" for a in milestone.acceptance]) if milestone.acceptance else "- (none provided)"
    schema_keys = ", ".join(schema_hint.get("properties", {}).keys())
    memory_tail = memory.strip()[-1200:] if memory.strip() else "(none)"

    feedback_block = ""
    if extra_feedback.strip():
        feedback_block = f"\nAdditional feedback / constraints:\n{extra_feedback.strip()}\n"

    prompt = f"""
You are a coding agent operating in a git repository.

Product definition (what “done” means):
{prd.strip()}

Current milestone:
- id: {milestone.id}
- title: {milestone.title}
Acceptance criteria:
{acceptance}

Recent memory (avoid repeating mistakes; keep changes minimal and verifiable):
{memory_tail}

Most recent oracle results:
{oracle_section if oracle_section else "(no prior oracle run)"}
{feedback_block}

Strict output requirement:
Return ONLY a single JSON object with keys: {schema_keys}
Do not include markdown fences. Do not include commentary outside JSON.
Your JSON MUST match the expected types.

Operational constraints:
- Make small, incremental changes.
- Prefer edits that improve verification outcomes.
- If blocked, set status="blocked" and explain why in summary.
- Do NOT run 'git commit' or switch branches.

Now proceed: implement the milestone or fix the failures, then output the required JSON.
""".strip()

    return prompt


def build_fix_prompt(oracle: OracleRunResult, diff_text: str, attempt: int, max_attempts: int) -> str:
    # Keep it focused: show failing step tails + diff (truncated already)
    failing = []
    for r in oracle.step_results:
        if not r.ok:
            failing.append(f"- FAIL ({'REQ' if r.required else 'OPT'}) kind={r.kind} {r.name}: {r.cmd}")
            if r.stderr.strip():
                failing.append("  stderr (tail):")
                failing.append(textwrap.indent(r.stderr.strip()[-1200:], "    "))
            if r.stdout.strip():
                failing.append("  stdout (tail):")
                failing.append(textwrap.indent(r.stdout.strip()[-1200:], "    "))
    failing_txt = "\n".join(failing) if failing else "(no failing steps found?)"

    return f"""
Oracle checks are failing. Fix them.

Retry attempt {attempt}/{max_attempts}.
Rules:
- Do not broaden the diff unnecessarily.
- Fix the failing checks first, cheapest path.
- Do not commit or switch branches.

Failing oracle details:
{failing_txt}

Current diff (includes staged+unstaged, may be truncated):
{diff_text}

After applying fixes, output the required JSON object (same schema as before).
""".strip()


def build_fuzzy_prompt(diff_text: str) -> str:
    return f"""
You are performing a code review. Evaluate the diff for:
1) syntactic correctness (no obvious syntax/type errors),
2) logical correctness (does it implement intended behavior),
3) regression risk.

Diff (may be truncated):
{diff_text}

Return ONLY a single JSON object with keys:
- syntactic_ok: boolean
- logical_ok: boolean
- regression_risk: "low"|"medium"|"high"|"unknown"
- notes: array of strings
- suggested_tests: array of strings

No markdown. No extra commentary.
""".strip()


# -----------------------------
# Controller helpers
# -----------------------------


def score_better(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return a > b  # lexicographic


def apply_plan_updates(milestones: List[Milestone], agent_obj: Dict[str, Any]) -> None:
    pu = agent_obj.get("plan_updates")
    if not isinstance(pu, dict):
        return
    mark_done = pu.get("mark_done", []) if isinstance(pu.get("mark_done"), list) else []
    mark_blocked = pu.get("mark_blocked", []) if isinstance(pu.get("mark_blocked"), list) else []

    mm = milestone_map(milestones)
    for mid in mark_done:
        if mid in mm:
            mm[mid].status = "done"
    for mid in mark_blocked:
        if mid in mm:
            mm[mid].status = "blocked"


def commit_message(m: Milestone, oracle: OracleRunResult, note: str = "") -> str:
    msg = f"agent: {m.id} {m.title} [{oracle.summary}]"
    if note:
        msg += f" {note}"
    return msg[:200]


def append_memory(memory_path: Path, line: str) -> None:
    existing = read_text(memory_path)
    if not existing.endswith("\n"):
        existing += "\n"
    write_text(memory_path, existing + line.rstrip() + "\n")


def run_fuzzy_review(
    reviewer: AgentAdapter,
    repo: Path,
    out_dir: Path,
    fuzzy_schema_path: Path,
    diff_text: str
) -> Dict[str, Any]:
    """
    Runs a fuzzy review using the same adapter interface, but we enforce our own fuzzy schema.
    For Codex, you should pass a CodexAdapter(sandbox="read-only") as reviewer.
    For Claude/custom, it just calls the tool and parses returned JSON.
    """
    prompt = build_fuzzy_prompt(diff_text)

    # We cannot rely on external schema enforcement for non-Codex tools,
    # but CodexAdapter will enforce its schema_path; we pass fuzzy_schema_path.
    obj = reviewer.run(prompt=prompt, repo=repo, out_dir=out_dir, schema_path=fuzzy_schema_path)
    # reviewer.run validates against the agent schema by default; to avoid that,
    # we treat fuzzy review as separate: parse again from the raw output file is tool-specific.
    # Simpler: we accept reviewer output if it matches fuzzy schema keys.
    # Therefore we validate it here.
    validate_fuzzy_shape(obj)
    return obj


def run_agent_with_validation(
    adapter: AgentAdapter,
    prompt: str,
    repo: Path,
    out_dir: Path,
    schema_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    agent_obj = adapter.run(prompt=prompt, repo=repo, out_dir=out_dir, schema_path=schema_path)
    validate_agent_shape(agent_obj)
    if output_path is not None:
        write_json(output_path, agent_obj)
    return agent_obj


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="BVH Controller for long-horizon agentic development")
    ap.add_argument("--repo", type=str, default=".", help="Path to git repository")
    ap.add_argument("--agent", type=str, choices=["codex", "claude", "custom"], default="codex", help="Agent adapter")
    ap.add_argument("--custom-cmd", type=str, default="", help="Custom command template (used when --agent custom)")
    ap.add_argument("--sandbox", type=str, default="workspace-write", help="Codex sandbox: read-only|workspace-write|danger-full-access")
    ap.add_argument("--approval", type=str, default="on-request", help="Codex ask-for-approval policy (if supported)")
    ap.add_argument("--model", type=str, default="", help="Codex model config (optional)")
    ap.add_argument("--dry-run", action="store_true", help="Do not commit; still runs oracle and reports")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"Repo not found: {repo}", file=sys.stderr)
        return 2

    if not git_is_repo(repo):
        print(f"Not a git repository: {repo}", file=sys.stderr)
        return 2

    agent_dir = repo / ".agent"
    ensure_file_defaults(agent_dir)

    prd_path = agent_dir / "prd.md"
    plan_path = agent_dir / "plan.json"
    oracle_path = agent_dir / "oracle.yaml"
    state_path = agent_dir / "state.json"
    memory_path = agent_dir / "memory.md"
    schema_path = agent_dir / "agent_output.schema.json"
    fuzzy_schema_path = agent_dir / "fuzzy_review.schema.json"
    reports_dir = agent_dir / "reports"

    state = read_json(state_path)
    if not state.get("started_at"):
        state["started_at"] = now_iso()
        write_json(state_path, state)

    # Start clean: recommended. If you want to continue from a partial test-failing state,
    # you can manually keep changes; but this controller assumes clean start for safety.
    if git_has_dirty_tree(repo):
        print("Working tree is dirty (including staged changes). Commit/stash before running the controller.", file=sys.stderr)
        return 2

    # Per-run budget
    run_started = dt.datetime.now(dt.timezone.utc)
    max_minutes = int(state.get("max_minutes", 90))
    deadline = run_started + dt.timedelta(minutes=max_minutes)

    max_iter = int(state.get("max_iterations", 30))
    no_improve_limit = int(state.get("no_improve_limit", 3))
    max_failed_per_ms = int(state.get("max_failed_attempts_per_milestone", 5))
    max_diff_lines = int(state.get("max_diff_lines", 400))
    max_changed_files = int(state.get("max_changed_files", 25))

    max_agent_fix_retries = int(state.get("max_agent_fix_retries", 3))
    allow_failing_tests_if_tests_changed = bool(state.get("allow_failing_tests_if_tests_changed", True))
    tests_markers = list(state.get("tests_path_markers", []))

    forbid_agent_git_commits = bool(state.get("forbid_agent_git_commits", True))
    forbid_agent_branch_change = bool(state.get("forbid_agent_branch_change", True))

    enable_fuzzy_review = bool(state.get("enable_fuzzy_review", False))
    fuzzy_blocks_high_risk = bool(state.get("fuzzy_review_blocks_high_risk_commit", True))

    # Adapter selection
    if args.agent == "codex":
        adapter: AgentAdapter = CodexAdapter(
            sandbox=args.sandbox,
            ask_for_approval=args.approval,
            model=(args.model or None),
        )
        # Reviewer: read-only Codex (recommended for fuzzy checks)
        reviewer: AgentAdapter = CodexAdapter(
            sandbox="read-only",
            ask_for_approval=args.approval,
            model=(args.model or None),
        )
    elif args.agent == "claude":
        adapter = ClaudeAdapter(output_format="json", continue_session=False)
        reviewer = ClaudeAdapter(output_format="json", continue_session=False)
    else:
        if not args.custom_cmd.strip():
            print("--agent custom requires --custom-cmd", file=sys.stderr)
            return 2
        adapter = CustomCommandAdapter(template=args.custom_cmd.strip())
        reviewer = adapter  # same adapter for fuzzy checks

    prd = read_text(prd_path)
    schema_hint = read_json(schema_path)
    memory = read_text(memory_path)

    milestones = load_milestones(plan_path)
    if not milestones:
        print("No milestones found in plan.json", file=sys.stderr)
        return 2

    best_score: Optional[Tuple[int, int, int, int]] = None
    if state.get("best_score"):
        bs = state["best_score"]
        if isinstance(bs, list) and len(bs) == 4 and all(isinstance(x, int) for x in bs):
            best_score = (bs[0], bs[1], bs[2], bs[3])

    last_oracle: Optional[OracleRunResult] = None
    fail_counts: Dict[str, int] = {}
    no_improve_streak = 0

    for it in range(1, max_iter + 1):
        if dt.datetime.now(dt.timezone.utc) > deadline:
            print("Budget stop: time limit reached.")
            break

        milestones = load_milestones(plan_path)
        m = select_next_milestone(milestones)
        if m is None:
            print("No eligible milestones (frontier empty). Either all done or blocked.")
            break

        # Mark milestone doing
        mm = milestone_map(milestones)
        if mm[m.id].status == "todo":
            mm[m.id].status = "doing"
            save_milestones(plan_path, milestones)

        iter_dir = reports_dir / f"iter-{it:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_prompt(prd, m, memory, last_oracle, schema_hint)
        write_text(iter_dir / "prompt.txt", prompt)

        print(f"\n=== Iteration {it}/{max_iter} | milestone {m.id}: {m.title} ===")

        # Guardrails: forbid commits/branch changes by agent
        head_before = git_head(repo)
        branch_before = git_current_branch(repo)

        # ---- Phase 1: agent works (initial attempt)
        try:
            agent_obj = run_agent_with_validation(
                adapter=adapter,
                prompt=prompt,
                repo=repo,
                out_dir=iter_dir,
                schema_path=schema_path,
                output_path=iter_dir / "agent_output.json",
            )
        except Exception as e:
            fail_counts[m.id] = fail_counts.get(m.id, 0) + 1
            write_text(iter_dir / "agent_error.txt", str(e))
            print(f"Agent invocation failed: {e}", file=sys.stderr)
            if fail_counts[m.id] >= max_failed_per_ms:
                mm = milestone_map(milestones)
                mm[m.id].status = "blocked"
                save_milestones(plan_path, milestones)
                append_memory(memory_path, f"- Agent invocation failures on {m.id} → check adapter/tooling → verify CLI installed/configured")
                memory = read_text(memory_path)
            continue

        # Validate forbidden git ops
        head_after = git_head(repo)
        branch_after = git_current_branch(repo)
        if forbid_agent_git_commits and head_after != head_before:
            write_text(iter_dir / "agent_error.txt", "Agent created a commit (HEAD changed). Controller forbids agent commits.")
            git_revert_workspace(repo)
            append_memory(memory_path, f"- Agent committed during {m.id} → forbid commits in tool policy → controller reverted")
            memory = read_text(memory_path)
            continue
        if forbid_agent_branch_change and branch_after != branch_before:
            write_text(iter_dir / "agent_error.txt", "Agent switched branches. Controller forbids branch changes.")
            try:
                run_cmd(["git", "checkout", branch_before], cwd=repo, capture=True)
            except Exception:
                pass
            git_revert_workspace(repo)
            append_memory(memory_path, f"- Agent switched branches during {m.id} → forbid branch changes → controller reverted")
            memory = read_text(memory_path)
            continue

        # Risk gate on diff size (staged+unstaged)
        changed_files = git_changed_files(repo)
        changed_files_n, changed_lines = git_diff_stats(repo)
        if (changed_lines > max_diff_lines) or (changed_files_n > max_changed_files):
            agent_obj["status"] = "needs_review"
            agent_obj["risk_notes"] = list(agent_obj.get("risk_notes", [])) + [
                f"Risk gate: diff too large (files≈{changed_files_n}, lines≈{changed_lines}). Review required."
            ]
            write_json(iter_dir / "agent_output.json", agent_obj)

        # Determine whether tests were changed
        tests_were_changed = tests_changed(changed_files, tests_markers)

        # ---- Phase 2: run oracle, and if failing, ask agent to fix up to max_agent_fix_retries
        soft_tests_policy = allow_failing_tests_if_tests_changed
        oracle_res = run_oracle(
            repo=repo,
            oracle_path=oracle_path,
            allow_soft_tests=soft_tests_policy,
            tests_were_changed=tests_were_changed,
        )
        last_oracle = oracle_res

        # If oracle fails in a way that is NOT soft-allowed, do fix retries
        fix_attempts = 0
        while not oracle_res.ok_required_with_soft_tests and fix_attempts < max_agent_fix_retries:
            fix_attempts += 1
            diff_text = git_unified_diff(repo)

            fix_dir = iter_dir / f"fix-{fix_attempts:02d}"
            fix_dir.mkdir(parents=True, exist_ok=True)

            extra = build_fix_prompt(oracle=oracle_res, diff_text=diff_text, attempt=fix_attempts, max_attempts=max_agent_fix_retries)
            fix_prompt = build_prompt(prd, m, memory, last_oracle, schema_hint, extra_feedback=extra)
            write_text(fix_dir / "prompt.txt", fix_prompt)

            # Guard before retry
            head_before_fix = git_head(repo)
            branch_before_fix = git_current_branch(repo)

            try:
                agent_obj = run_agent_with_validation(
                    adapter=adapter,
                    prompt=fix_prompt,
                    repo=repo,
                    out_dir=fix_dir,
                    schema_path=schema_path,
                    output_path=fix_dir / "agent_output.json",
                )
            except Exception as e:
                write_text(fix_dir / "agent_error.txt", str(e))
                print(f"Agent fix attempt failed: {e}", file=sys.stderr)
                break

            # Validate forbidden git ops
            head_after_fix = git_head(repo)
            branch_after_fix = git_current_branch(repo)
            if forbid_agent_git_commits and head_after_fix != head_before_fix:
                write_text(fix_dir / "agent_error.txt", "Agent created a commit during fix. Forbidden.")
                git_revert_workspace(repo)
                append_memory(memory_path, f"- Agent committed during fix for {m.id} → forbid commits → controller reverted")
                memory = read_text(memory_path)
                oracle_res = run_oracle(repo, oracle_path, soft_tests_policy, tests_were_changed)
                break
            if forbid_agent_branch_change and branch_after_fix != branch_before_fix:
                write_text(fix_dir / "agent_error.txt", "Agent switched branches during fix. Forbidden.")
                try:
                    run_cmd(["git", "checkout", branch_before_fix], cwd=repo, capture=True)
                except Exception:
                    pass
                git_revert_workspace(repo)
                append_memory(memory_path, f"- Agent switched branches during fix for {m.id} → forbid branch changes → reverted")
                memory = read_text(memory_path)
                oracle_res = run_oracle(repo, oracle_path, soft_tests_policy, tests_were_changed)
                break

            # Recompute change context & rerun oracle after fix attempt
            changed_files = git_changed_files(repo)
            tests_were_changed = tests_changed(changed_files, tests_markers)
            oracle_res = run_oracle(repo, oracle_path, soft_tests_policy, tests_were_changed)
            last_oracle = oracle_res

        # ---- Hard failure after retries (unless only soft-test failures remain)
        hard_fail = not oracle_res.ok_required_with_soft_tests
        if hard_fail:
            fail_counts[m.id] = fail_counts.get(m.id, 0) + 1
            print(f"Hard oracle failure after retries: {oracle_res.summary}")
            # Hard failure policy: revert to last clean state
            git_revert_workspace(repo)
            append_memory(memory_path, f"- Oracle failure on {m.id} after {max_agent_fix_retries} retries → investigate environment or reduce scope")
            memory = read_text(memory_path)

            if fail_counts[m.id] >= max_failed_per_ms:
                mm = milestone_map(milestones)
                mm[m.id].status = "blocked"
                save_milestones(plan_path, milestones)
                print(f"Milestone {m.id} marked blocked after repeated failures.")
            continue

        # From here: oracle is acceptable, possibly with soft test failures
        if oracle_res.soft_failed_required_tests:
            # partial state allowed: do not revert merely due to tests failing
            print("Oracle: required steps acceptable, but tests failed and are soft-allowed (tests changed).")

        # Optional fuzzy review as a risk gate (advisory or blocking)
        fuzzy_obj: Optional[Dict[str, Any]] = None
        fuzzy_blocks_commit = False
        if enable_fuzzy_review:
            try:
                diff_text = git_unified_diff(repo)
                fuzzy_dir = iter_dir / "fuzzy-review"
                fuzzy_dir.mkdir(parents=True, exist_ok=True)
                # Use reviewer adapter; for Codex, use read-only sandbox.
                # NOTE: reviewer.run enforces agent schema, so we bypass by using a direct prompt + parsing.
                # Simplest: call reviewer, then re-parse from its output file is tool-specific; instead:
                # We call reviewer with the fuzzy schema path and require it to output the fuzzy JSON.
                fuzzy_raw = reviewer.run(prompt=build_fuzzy_prompt(diff_text), repo=repo, out_dir=fuzzy_dir, schema_path=fuzzy_schema_path)
                validate_fuzzy_shape(fuzzy_raw)
                fuzzy_obj = fuzzy_raw
                write_json(fuzzy_dir / "fuzzy.json", fuzzy_obj)

                if fuzzy_blocks_high_risk and fuzzy_obj.get("regression_risk") == "high":
                    fuzzy_blocks_commit = True
            except Exception as e:
                # Fuzzy review failures should not automatically block; log and continue.
                write_text(iter_dir / "fuzzy_error.txt", str(e))
                print(f"Fuzzy review failed (non-blocking): {e}", file=sys.stderr)

        # Report for iteration
        report = {
            "iteration": it,
            "timestamp": now_iso(),
            "milestone": dataclasses.asdict(m),
            "agent_output": agent_obj,
            "git": {
                "branch": git_current_branch(repo),
                "head": git_head(repo),
                "changed_files": changed_files,
                "changed_files_approx": changed_files_n,
                "changed_lines_estimate": changed_lines,
                "tests_were_changed": tests_were_changed,
            },
            "oracle": {
                "ok_required_strict": oracle_res.ok_required_strict,
                "ok_required_with_soft_tests": oracle_res.ok_required_with_soft_tests,
                "soft_failed_required_tests": oracle_res.soft_failed_required_tests,
                "score": list(oracle_res.score),
                "summary": oracle_res.summary,
                "steps": [dataclasses.asdict(s) for s in oracle_res.step_results],
                "fix_attempts_used": fix_attempts,
            },
            "fuzzy_review": fuzzy_obj,
            "fuzzy_blocks_commit": fuzzy_blocks_commit,
        }
        write_json(iter_dir / "report.json", report)

        # Decide improvement:
        # Even if tests are soft-failing, we still allow "progress" and (optionally) commit.
        improved = (best_score is None) or score_better(oracle_res.score, best_score)

        if improved:
            no_improve_streak = 0
            best_score = oracle_res.score

            # Apply plan updates
            apply_plan_updates(milestones, agent_obj)

            # If tests are soft-failing, do NOT auto-mark done (usually)
            if agent_obj.get("status") == "done" and not oracle_res.soft_failed_required_tests:
                mm = milestone_map(milestones)
                mm[m.id].status = "done"
            elif mm[m.id].status == "todo":
                mm[m.id].status = "doing"

            save_milestones(plan_path, milestones)

            # Commit policy:
            # - If fuzzy review blocks, do not commit automatically.
            # - If tests soft-failed, committing is allowed (WIP), but message notes it.
            if args.dry_run:
                print(f"[dry-run] Would commit: {commit_message(m, oracle_res)}")
            else:
                if fuzzy_blocks_commit:
                    print("Fuzzy review indicates high regression risk. Not auto-committing; leaving changes in worktree.")
                    append_memory(memory_path, f"- Fuzzy review high risk on {m.id} → manual review required before commit")
                    memory = read_text(memory_path)
                else:
                    note = "soft-tests-failing" if oracle_res.soft_failed_required_tests else ""
                    git_commit(repo, commit_message(m, oracle_res, note=note))
                    print(f"Committed: {commit_message(m, oracle_res, note=note)}")

            # Persist state
            state = read_json(state_path)
            state["best_score"] = list(best_score)
            state.setdefault("history", [])
            state["history"].append(
                {
                    "iteration": it,
                    "timestamp": now_iso(),
                    "milestone_id": m.id,
                    "oracle_score": list(best_score),
                    "oracle_summary": oracle_res.summary,
                    "agent_status": agent_obj.get("status"),
                    "report_path": str(iter_dir.relative_to(repo)),
                    "soft_failed_required_tests": oracle_res.soft_failed_required_tests,
                    "fix_attempts_used": fix_attempts,
                    "fuzzy_blocks_commit": fuzzy_blocks_commit,
                }
            )
            write_json(state_path, state)

            milestones = load_milestones(plan_path)
            if all(ms.status == "done" for ms in milestones):
                print("All milestones done. Stopping.")
                break
        else:
            # No improvement: if only soft-test failures exist, do not revert automatically;
            # but if there is no improvement and no soft-test situation, revert to avoid drift.
            no_improve_streak += 1
            print(f"No oracle improvement (streak {no_improve_streak}/{no_improve_limit}).")

            if oracle_res.soft_failed_required_tests:
                print("Keeping partial state (soft-failing tests). Not reverting.")
            else:
                print("Reverting (no improvement and not in soft-test mode).")
                git_revert_workspace(repo)

            append_memory(memory_path, f"- No improvement on {m.id} → simplify change and target weakest oracle signal")
            memory = read_text(memory_path)

            if no_improve_streak >= no_improve_limit:
                print("Stop: local optimum / no improvement limit reached.")
                break

    print("\nController finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
