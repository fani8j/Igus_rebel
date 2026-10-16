#!/usr/bin/env python3
"""Build and query a local project/session RAG index.

The local backend is dependency-free SQLite FTS5. It indexes the repository
source/docs and the agent sessions (omp and Claude Code) that worked in this
repository or one of its worktrees. `build` rebuilds the index atomically;
`refresh` re-reads only new/changed/deleted files. Optional LightRAG commands
use its REST API; credentials are read only from the environment.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = 2
DEFAULT_INDEX = Path("workspace_scan_artifacts/project_rag.sqlite3")
DEFAULT_MAP = Path("workspace_scan_artifacts/project_rag_map.json")
DEFAULT_KNOWLEDGE = Path(__file__).with_name("project_knowledge.json")
DEFAULT_OMP_ROOT = Path.home() / ".omp" / "agent" / "sessions"
DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"
MAX_SOURCE_BYTES = 4_000_000
MAX_JSON_BYTES = 1_000_000
MAX_EVIDENCE_JSON_BYTES = 64_000
MAX_EVIDENCE_BYTES = 256_000
MAX_CHUNK_CHARS = 4_000
OVERLAP_LINES = 4
MAX_TITLE_CHARS = 100

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    ".claude",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "backups",
    "build",
    "cad",
    "install",
    "isaac_exports",
    "log",
    "Log",
    "node_modules",
    "workspace_scan_artifacts",
}
EXCLUDED_DIR_PATTERNS = ("build_*", "install_*", "log_*", "ros-humble-*", "*.egg-info")
SOURCE_SUFFIXES = {
    ".action",
    ".bash",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".launch",
    ".md",
    ".msg",
    ".py",
    ".rviz",
    ".sh",
    ".srv",
    ".txt",
    ".urdf",
    ".xacro",
    ".xml",
    ".yaml",
    ".yml",
}
SOURCE_NAMES = {"CMakeLists.txt", "Dockerfile", "Makefile", "package.xml"}
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
CODE_COPY_SUFFIXES = {
    ".c", ".cc", ".cmake", ".cpp", ".h", ".hpp", ".launch", ".py", ".sh", ".urdf", ".xacro", ".xml", ".yaml", ".yml",
}
EVIDENCE_DIR = "scan_results"
SNAPSHOT_COPY_RE = re.compile(
    r"\.(?:c|cc|cmake|cpp|h|hpp|launch|msg|py|sh|srv|urdf|xacro|xml|yaml|yml)\.txt$", re.IGNORECASE
)
SESSION_KINDS = {"session_turn"}
CURATED_KINDS = ("verified_fact", "unresolved_item", "improvement")
KIND_ALIASES = {
    "source": ("source_code",),
    "code": ("source_code",),
    "doc": ("project_document",),
    "docs": ("project_document",),
    "document": ("project_document",),
    "documents": ("project_document",),
    "session": ("session_turn",),
    "sessions": ("session_turn",),
    "turn": ("session_turn",),
    "curated": CURATED_KINDS,
    "knowledge": CURATED_KINDS,
}
STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "been",
    "before",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "not",
    "our",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]{2,}")
REFERENCE_PATTERNS = (
    re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_.]+)", re.MULTILINE),
    re.compile(r"\bfind_package\s*\(\s*([A-Za-z0-9_+-]+)", re.IGNORECASE),
    re.compile(r"<(?:build_|build_export_|exec_|test_)?depend>\s*([^<\s]+)\s*</(?:build_|build_export_|exec_|test_)?depend>"),
)
# Claude Code wraps non-prompt material in tags; those spans are removed from user text.
CLAUDE_WRAPPER_RE = re.compile(
    r"<(system-reminder|command-name|command-message|command-args|local-command-stdout|local-command-stderr"
    r"|local-command-caveat|bash-input|bash-stdout|bash-stderr|task-notification|user-prompt-submit-hook"
    r"|ide_opened_file|ide_selection)>.*?</\1>",
    re.DOTALL,
)
CLAUDE_SKIP_PREFIXES = ("[Request interrupted", "Caveat: The messages below were generated")
CWD_RE = re.compile(r'"cwd"\s*:\s*"((?:[^"\\]|\\.)*)"')
CLAUDE_MESSAGE_RE = re.compile(r'"type"\s*:\s*"(?:user|assistant)"')
CLAUDE_TITLE_RE = re.compile(r'"type"\s*:\s*"(?:custom-title|ai-title|summary)"')
CLAUDE_TOOL_RESULT_RE = re.compile(r'"type"\s*:\s*"tool_result"')
CLAUDE_TEXT_BLOCK_RE = re.compile(r'"type"\s*:\s*"text"')


@dataclass(frozen=True)
class Document:
    id: str
    kind: str
    source: str
    locator: str
    session_id: str | None
    timestamp: str | None
    role: str | None
    topics: tuple[str, ...]
    content: str
    content_sha256: str
    symbol: str | None = None


@dataclass
class SessionInfo:
    id: str
    source: str
    cwd: str
    title: str
    created: str
    last_activity: str
    kind: str = "top_level"
    parent_session_id: str | None = None
    user_messages: int = 0
    assistant_messages: int = 0
    document_count: int = 0
    worktree: str | None = None
    format: str = "omp"

    @property
    def knowledge_events(self) -> int:
        return self.user_messages + self.assistant_messages


@dataclass
class InputFile:
    """One file the index depends on, identified by its documents.source value."""

    key: str
    path: Path
    group: str  # source | omp | claude
    size: int
    mtime_ns: int
    matched: bool = True


@dataclass
class IndexConfig:
    repo_root: Path
    omp_roots: tuple[Path, ...] = ()
    claude_roots: tuple[Path, ...] = ()
    worktree_roots: tuple[Path, ...] = ()
    knowledge_path: Path = DEFAULT_KNOWLEDGE
    index_path: Path = DEFAULT_INDEX
    map_path: Path = DEFAULT_MAP

    def fingerprint(self) -> str:
        return sha256_text(
            json.dumps(
                {
                    "repo": str(self.repo_root),
                    "omp": sorted(str(item) for item in self.omp_roots),
                    "claude": sorted(str(item) for item in self.claude_roots),
                    "worktrees": sorted(str(item) for item in self.worktree_roots),
                    "knowledge": str(self.knowledge_path),
                }
            )
        )


class LightRagHttpError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"LightRAG HTTP {status}: {detail}")
        self.status = status


class IndexLocked(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def load_knowledge(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported knowledge schema in {path}")
    return value


def compile_topics(knowledge: dict[str, Any]) -> list[tuple[str, re.Pattern[str]]]:
    topics: list[tuple[str, re.Pattern[str]]] = []
    for item in knowledge.get("topics", []):
        patterns = item.get("patterns") or []
        if not patterns:
            continue
        topics.append((item["id"], re.compile("|".join(patterns), re.IGNORECASE)))
    return topics


def classify_topics(text: str, topics: Sequence[tuple[str, re.Pattern[str]]]) -> tuple[str, ...]:
    return tuple(topic_id for topic_id, pattern in topics if pattern.search(text))


def text_parts(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> Iterator[tuple[int, int, str]]:
    lines = text.splitlines() or [text]
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines):
            addition = len(lines[end]) + 1
            if end > start and size + addition > max_chars:
                break
            if addition > max_chars and end == start:
                line = lines[end]
                offset = 0
                while offset < len(line):
                    piece = line[offset : offset + max_chars]
                    yield start + 1, start + 1, piece
                    offset += max_chars
                end += 1
                size = 0
                break
            size += addition
            end += 1
        if end > start and size:
            yield start + 1, end, "\n".join(lines[start:end]).strip()
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)


# --------------------------------------------------------------------------- symbol-aware chunking


@dataclass
class Span:
    start: int  # 1-based, inclusive
    end: int
    name: str


def camel_split(symbol: str) -> str:
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", symbol)
    return " ".join(part.lower() for part in parts)


def symbol_search_text(symbol: str | None) -> str:
    if not symbol:
        return ""
    return f"{symbol} {camel_split(symbol)}"


def extend_over_comments(lines: Sequence[str], start: int, prefixes: tuple[str, ...]) -> int:
    """Move a span start up over comment-only lines directly above it (1-based)."""
    while start > 1 and lines[start - 2].strip().startswith(prefixes):
        start -= 1
    return start


def python_spans(text: str) -> list[Span] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    lines = text.splitlines()
    spans: list[Span] = []

    def visit(body: Sequence[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                first = min([node.lineno] + [item.lineno for item in node.decorator_list])
                end = getattr(node, "end_lineno", None) or node.lineno
                name = prefix + node.name
                spans.append(Span(extend_over_comments(lines, first, ("#",)), end, name))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + ".")

    visit(tree.body, "")
    return spans


_CPP_NOISE = (
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r'|R"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|^[ \t]*\#(?:[^\n]|\\\n)*"
)
CPP_NOISE_RE = re.compile(_CPP_NOISE, re.DOTALL | re.MULTILINE)
CPP_TOKEN_RE = re.compile(_CPP_NOISE + r"|[{};]", re.DOTALL | re.MULTILINE)
CPP_NEXT_CODE_RE = re.compile(r"(?:\s|//[^\n]*|/\*.*?\*/)*(\S)", re.DOTALL)
CPP_ACCESS_RE = re.compile(r"^(?:(?:public|private|protected)\s*:(?!:)|(?:Q_SLOTS|Q_SIGNALS|signals|slots)\s*:)\s*")
CPP_CONTROL_RE = re.compile(r"^(?:if|for|while|switch|catch|else|do|try)\b")
CPP_OPERATOR_RE = re.compile(r"((?:[A-Za-z_]\w*\s*::\s*)*)operator\s*(\(\s*\)|[^\s\w(]+|\w+(?:\s*\[\s*\])?)\s*\(")
CPP_FUNC_NAME_RE = re.compile(r"((?:~?[A-Za-z_]\w*(?:\s*<[^<>()]*>)?\s*::\s*)*~?[A-Za-z_]\w*)\s*\(")
CPP_NOT_FUNCTIONS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "decltype", "alignas", "alignof", "noexcept",
    "static_assert", "requires", "throw", "new", "delete", "typeof", "defined", "__attribute__", "__declspec",
}


def blank_cpp_noise(text: str) -> str:
    return CPP_NOISE_RE.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), text)


def strip_template_prefix(header: str) -> str:
    while header.startswith("template"):
        rest = header[len("template") :].lstrip()
        if not rest.startswith("<"):
            break
        depth = 0
        end = None
        for index, char in enumerate(rest):
            if char == "<":
                depth += 1
            elif char == ">" and rest[index - 1 : index] != "-":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        header = rest[end:].lstrip()
    return header


def has_toplevel_assign(header: str) -> bool:
    depth = 0
    for index, char in enumerate(header):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "=" and depth == 0:
            before = header[index - 1] if index else ""
            after = header[index + 1] if index + 1 < len(header) else ""
            if after == "=" or before in "=!<>":
                continue
            return True
    return False


def cpp_classify(header: str, scope: str | None) -> tuple[str, str | None]:
    """Classify the text before an opening brace at namespace/class scope."""
    h = " ".join(header.split())
    while True:
        match = CPP_ACCESS_RE.match(h)
        if not match:
            break
        h = h[match.end() :]
    h = strip_template_prefix(h)
    if not h:
        return "other", None
    if h == "extern" or (re.match(r"^(?:inline\s+)?namespace\b", h) and "(" not in h):
        return "namespace", None
    match = re.match(r"^(?:typedef\s+)?(?:class|struct|union)\b(.*)$", h)
    if match:
        head = re.split(r"(?<!:):(?!:)", match.group(1))[0]
        if "(" not in head:
            for _ in range(3):
                head = re.sub(r"<[^<>]*>", "", head)
            head = re.sub(r"\[\[.*?\]\]", "", head)
            words = [word for word in re.findall(r"[A-Za-z_]\w*(?:::\w+)*", head) if word != "final"]
            return ("class", words[-1]) if words else ("other", None)
    if CPP_CONTROL_RE.match(h) or "(" not in h:
        return "other", None
    name: str | None = None
    operator = CPP_OPERATOR_RE.search(h)
    probe = h
    if operator:
        name = re.sub(r"\s+", "", operator.group(1)) + "operator" + re.sub(r"\s+", "", operator.group(2))
        probe = h[: operator.start()] + "operator_ (" + h[operator.end() :]
    if has_toplevel_assign(probe):
        return "other", None
    if name is None:
        for match in CPP_FUNC_NAME_RE.finditer(h):
            raw = match.group(1)
            last = re.split(r"\s*::\s*", raw)[-1]
            if last in CPP_NOT_FUNCTIONS or last.startswith("__attribute"):
                continue
            name = re.sub(r"\s+", "", raw)
            if match.start() == 0 and re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
                call = re.match(r"[A-Z][A-Z0-9_]+\s*\(([^()]*)\)", h)
                if call:
                    name = f"{name}({' '.join(call.group(1).split())})"
            break
    if not name:
        return "other", None
    if "::" not in name and scope:
        name = f"{scope}::{name}"
    return "function", name


def cpp_spans(text: str) -> list[Span] | None:
    """Brace-depth heuristic: top-level function and class spans, or None if the scan does not balance."""
    if not text.strip():
        return None
    lines = text.splitlines()
    line_starts = [0]
    for match in re.finditer(r"\n", text):
        line_starts.append(match.end())

    def line_of(position: int) -> int:
        return bisect.bisect_right(line_starts, position)

    containers: list[tuple[str, str | None, int]] = []  # (kind, name, start_line)
    spans: list[Span] = []
    header_start = 0
    opaque = 0
    current: tuple[str, int] | None = None  # (function name, start line)
    for match in CPP_TOKEN_RE.finditer(text):
        token = match.group()
        if len(token) != 1 or token not in "{};":
            continue
        position = match.start()
        if token == "{":
            if opaque:
                opaque += 1
                continue
            header = blank_cpp_noise(text[header_start:position])
            scope = next((name for kind, name, _ in reversed(containers) if kind == "class" and name), None)
            kind, name = cpp_classify(header, scope)
            if kind == "namespace":
                containers.append(("namespace", None, 0))
                header_start = match.end()
            elif kind == "class":
                start = header_start + len(header) - len(header.lstrip())
                start_line = extend_over_comments(lines, line_of(start), ("//", "/*", "*"))
                qualified = f"{scope}::{name}" if scope else name
                containers.append(("class", qualified, start_line))
                header_start = match.end()
            else:
                opaque = 1
                current = None
                if kind == "function" and name:
                    start = header_start + len(header) - len(header.lstrip())
                    current = (name, extend_over_comments(lines, line_of(start), ("//", "/*", "*")))
        elif token == "}":
            if opaque:
                opaque -= 1
                if opaque == 0:
                    follow = CPP_NEXT_CODE_RE.match(text, match.end())
                    if current is not None and follow and follow.group(1) in ",{":
                        current = None  # brace initializer inside a constructor initializer list
                        continue
                    if current is not None:
                        spans.append(Span(current[1], line_of(position), current[0]))
                    current = None
                    header_start = match.end()
            else:
                if not containers:
                    return None
                kind, name, start_line = containers.pop()
                if kind == "class" and name:
                    spans.append(Span(start_line, line_of(position), name))
                header_start = match.end()
        elif not opaque:
            header_start = match.end()
    if opaque or containers:
        return None
    return spans


def symbol_chunks(text: str, spans: Sequence[Span]) -> list[tuple[str | None, int, int, str]]:
    lines = text.splitlines()
    owner = [-1] * (len(lines) + 2)
    for index in sorted(range(len(spans)), key=lambda item: -(spans[item].end - spans[item].start)):
        span = spans[index]
        for number in range(max(span.start, 1), min(span.end, len(lines)) + 1):
            owner[number] = index
    result: list[tuple[str | None, int, int, str]] = []
    start = 1
    while start <= len(lines):
        end = start
        while end + 1 <= len(lines) and owner[end + 1] == owner[start]:
            end += 1
        segment = "\n".join(lines[start - 1 : end])
        if re.search(r"\w", segment):
            symbol = spans[owner[start]].name if owner[start] >= 0 else None
            for first, last, chunk in chunk_text(segment):
                if chunk:
                    result.append((symbol, start - 1 + first, start - 1 + last, chunk))
        start = end + 1
    return result


def source_chunks(relative: str, text: str) -> list[tuple[str | None, int, int, str]]:
    suffix = Path(relative).suffix.lower()
    spans: list[Span] | None = None
    if suffix == ".py":
        spans = python_spans(text)
    elif suffix in CPP_SUFFIXES:
        spans = cpp_spans(text)
    if spans:
        return symbol_chunks(text, spans)
    return [(None, start, end, chunk) for start, end, chunk in chunk_text(text) if chunk]


def make_document(
    *,
    kind: str,
    source: str,
    locator: str,
    content: str,
    topics: Sequence[tuple[str, re.Pattern[str]]],
    session_id: str | None = None,
    timestamp: str | None = None,
    role: str | None = None,
    symbol: str | None = None,
) -> Document:
    content = content.strip()
    digest = sha256_text(content)
    return Document(
        id=stable_id(kind, source, locator, digest),
        kind=kind,
        source=source,
        locator=locator,
        session_id=session_id,
        timestamp=timestamp,
        role=role,
        topics=classify_topics(content, topics),
        content=content,
        content_sha256=digest,
        symbol=symbol,
    )


# --------------------------------------------------------------------------- sessions


def worktree_name(cwd: str, worktree_roots: Sequence[Path]) -> str | None:
    path = os.path.normpath(cwd)
    for root in worktree_roots:
        prefix = os.path.normpath(str(root))
        if path.startswith(prefix + os.sep):
            return path[len(prefix) + 1 :].split(os.sep)[0] or None
    return None


def cwd_matches(cwd: str, repo_root: Path, worktree_roots: Sequence[Path]) -> bool:
    """True for the repo root, any path below it, and any path below a worktree root."""
    if not cwd:
        return False
    path = os.path.normpath(cwd)
    roots = {os.path.normpath(str(repo_root))}
    try:
        roots.add(os.path.normpath(str(repo_root.resolve())))
    except OSError:
        pass
    roots.update(os.path.normpath(str(root)) for root in worktree_roots)
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def default_worktree_roots(repo_root: Path) -> list[Path]:
    return [repo_root / ".claude" / "worktrees", Path.home() / "orca" / "workspaces" / repo_root.name]


def session_header(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if line_number > 30:
                    return None
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "session":
                    return event
    except OSError:
        return None
    return None


def claude_cwds(path: Path, max_lines: int = 300) -> list[str]:
    """Distinct cwd values from the first lines of a Claude Code session, without parsing whole events."""
    found: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if line_number > max_lines:
                    break
                match = CWD_RE.search(line)
                if not match:
                    continue
                try:
                    value = json.loads('"' + match.group(1) + '"')
                except json.JSONDecodeError:
                    continue
                if value not in found:
                    found.append(value)
    except OSError:
        return found
    return found


def stat_key(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_size, info.st_mtime_ns


def scan_omp_files(
    roots: Sequence[Path],
    repo_root: Path,
    worktree_roots: Sequence[Path],
    known: dict[str, tuple[int, int, bool]] | None = None,
) -> list[InputFile]:
    found: list[InputFile] = []
    for root in roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            stat = stat_key(path)
            if stat is None:
                continue
            key = str(path)
            previous = (known or {}).get(key)
            if previous and previous[:2] == stat:
                matched = previous[2]
            else:
                header = session_header(path)
                matched = bool(header and header.get("cwd") and cwd_matches(str(header["cwd"]), repo_root, worktree_roots))
            found.append(InputFile(key, path, "omp", stat[0], stat[1], matched))
    return found


def scan_claude_files(
    roots: Sequence[Path],
    repo_root: Path,
    worktree_roots: Sequence[Path],
    known: dict[str, tuple[int, int, bool]] | None = None,
) -> list[InputFile]:
    found: list[InputFile] = []
    for root in roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        top_matched: dict[str, bool] = {}
        candidates = sorted(root.glob("*/*.jsonl")) + sorted(root.glob("*/*/subagents/*.jsonl"))
        for path in candidates:
            stat = stat_key(path)
            if stat is None:
                continue
            key = str(path)
            is_subagent = path.parent.name == "subagents"
            previous = (known or {}).get(key)
            if previous and previous[:2] == stat:
                matched = previous[2]
            else:
                cwds = claude_cwds(path)
                if cwds:
                    matched = any(cwd_matches(cwd, repo_root, worktree_roots) for cwd in cwds)
                elif is_subagent:
                    matched = top_matched.get(str(path.parent.parent.parent / f"{path.parent.parent.name}.jsonl"), False)
                else:
                    matched = False
            if not is_subagent:
                top_matched[key] = matched
            found.append(InputFile(key, path, "claude", stat[0], stat[1], matched))
    return found


def discover_session_files(
    session_root: Path, repo_root: Path, worktree_roots: Sequence[Path] = ()
) -> list[Path]:
    """omp-format session files whose recorded cwd is the repo, below it, or in a worktree root."""
    return [
        item.path
        for item in scan_omp_files([session_root], repo_root.resolve(), worktree_roots)
        if item.matched
    ]


def parent_session_id(path: Path) -> str | None:
    candidate = path.parent.parent / f"{path.parent.name}.jsonl"
    if not candidate.is_file():
        return None
    header = session_header(candidate)
    return str(header["id"]) if header and header.get("id") else None


def one_line(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def read_session(
    path: Path,
    topics: Sequence[tuple[str, re.Pattern[str]]],
    worktree_roots: Sequence[Path] = (),
) -> tuple[SessionInfo, list[Document]]:
    header = session_header(path)
    if header is None:
        raise ValueError(f"No session header in {path}")
    session_id = str(header["id"])
    title = str(header.get("title") or "")
    created = str(header.get("timestamp") or "")
    last_activity = created
    parent_id = parent_session_id(path)
    cwd = str(header.get("cwd") or "")
    info = SessionInfo(
        id=session_id,
        source=str(path),
        cwd=cwd,
        title=title,
        created=created,
        last_activity=last_activity,
        kind="subagent" if parent_id else "top_level",
        parent_session_id=parent_id,
        worktree=worktree_name(cwd, worktree_roots),
        format="omp",
    )
    documents: list[Document] = []
    message_index = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = str(event.get("timestamp") or "")
            if timestamp and timestamp > info.last_activity:
                info.last_activity = timestamp
            if event.get("type") in {"title", "title_change"} and event.get("title"):
                info.title = str(event["title"])
            if event.get("type") != "message" or not isinstance(event.get("message"), dict):
                continue
            message = event["message"]
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = text_parts(message)
            if not text:
                continue
            message_index += 1
            if role == "user":
                info.user_messages += 1
            else:
                info.assistant_messages += 1
            for chunk_index, (start, end, chunk) in enumerate(chunk_text(text), 1):
                locator = f"message:{message_index}:chunk:{chunk_index}:lines:{start}-{end}"
                documents.append(
                    make_document(
                        kind="session_turn",
                        source=str(path),
                        locator=locator,
                        content=chunk,
                        topics=topics,
                        session_id=session_id,
                        timestamp=timestamp or None,
                        role=str(role),
                    )
                )
    if not info.title:
        first_user = next((doc.content for doc in documents if doc.role == "user"), "")
        info.title = one_line(first_user, MAX_TITLE_CHARS)
    info.document_count = len(documents)
    return info, documents


def claude_event_text(event: dict[str, Any], role: str) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    if role == "assistant" and message.get("model") == "<synthetic>":
        return ""
    content = message.get("content")
    pieces: list[str] = []
    if isinstance(content, str):
        pieces.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
    text = "\n".join(pieces)
    if role == "user":
        text = CLAUDE_WRAPPER_RE.sub("", text)
    text = text.strip()
    if text.startswith(CLAUDE_SKIP_PREFIXES):
        return ""
    return text


def read_claude_session(
    path: Path,
    topics: Sequence[tuple[str, re.Pattern[str]]],
    worktree_roots: Sequence[Path] = (),
) -> tuple[SessionInfo, list[Document]]:
    is_subagent = path.parent.name == "subagents"
    parent_id = path.parent.parent.name if is_subagent else None
    session_id = path.stem
    info = SessionInfo(
        id=session_id,
        source=str(path),
        cwd="",
        title="",
        created="",
        last_activity="",
        kind="subagent" if is_subagent else "top_level",
        parent_session_id=parent_id,
        format="claude",
    )
    custom_title = ai_title = summary_title = ""
    documents: list[Document] = []
    message_index = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not CLAUDE_MESSAGE_RE.search(line) and not CLAUDE_TITLE_RE.search(line[:200]):
                continue  # cheap prefilter: most lines are attachments, API blobs, hooks
            if CLAUDE_TOOL_RESULT_RE.search(line) and not CLAUDE_TEXT_BLOCK_RE.search(line):
                continue  # tool results carry no prompt/answer text
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "custom-title" and event.get("customTitle"):
                custom_title = str(event["customTitle"])
                continue
            if event_type == "ai-title" and event.get("aiTitle"):
                ai_title = str(event["aiTitle"])
                continue
            if event_type == "summary" and event.get("summary"):
                summary_title = str(event["summary"])
                continue
            if event_type not in {"user", "assistant"} or event.get("isMeta"):
                continue
            if not info.cwd and event.get("cwd"):
                info.cwd = str(event["cwd"])
            timestamp = str(event.get("timestamp") or "")
            text = claude_event_text(event, str(event_type))
            if not text:
                continue
            if timestamp:
                if not info.created or timestamp < info.created:
                    info.created = timestamp
                if timestamp > info.last_activity:
                    info.last_activity = timestamp
            message_index += 1
            if event_type == "user":
                info.user_messages += 1
                if not info.title:
                    info.title = one_line(text, MAX_TITLE_CHARS)
            else:
                info.assistant_messages += 1
            for chunk_index, (start, end, chunk) in enumerate(chunk_text(text), 1):
                locator = f"message:{message_index}:chunk:{chunk_index}:lines:{start}-{end}"
                documents.append(
                    make_document(
                        kind="session_turn",
                        source=str(path),
                        locator=locator,
                        content=chunk,
                        topics=topics,
                        session_id=session_id,
                        timestamp=timestamp or None,
                        role=str(event_type),
                    )
                )
    first_prompt = info.title
    info.title = custom_title or ai_title or summary_title or first_prompt
    if is_subagent:
        meta_path = path.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            label = ": ".join(str(meta[key]) for key in ("agentType", "description") if meta.get(key))
            if label:
                info.title = one_line(label, MAX_TITLE_CHARS)
        except (OSError, ValueError, TypeError):
            pass
    info.title = one_line(info.title, MAX_TITLE_CHARS)
    if not info.created:
        stat = path.stat()
        info.created = info.last_activity = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    info.worktree = worktree_name(info.cwd, worktree_roots)
    info.document_count = len(documents)
    return info, documents


# --------------------------------------------------------------------------- source files


def is_excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIRS or any(fnmatch.fnmatchcase(name, pattern) for pattern in EXCLUDED_DIR_PATTERNS)


def source_skip_reason(relative: str, filename: str, size: int) -> str | None:
    """Why a candidate source file is not indexed, or None. 'too_large' reasons are logged by callers."""
    parts = relative.split("/")
    suffix = Path(filename).suffix.lower()
    if relative.startswith("tools/session_rag/") and (
        filename.startswith("test_") or filename == "project_knowledge.json"
    ):
        return "excluded"
    if filename not in SOURCE_NAMES and suffix not in SOURCE_SUFFIXES:
        return "suffix"
    evidence = parts[0] == EVIDENCE_DIR
    if evidence:
        if SNAPSHOT_COPY_RE.search(filename):
            return "source_copy"
        if suffix in CODE_COPY_SUFFIXES and any(
            part in {"baseline", "after", "before"} or part.startswith("baseline_") for part in parts[1:-1]
        ):
            return "source_copy"
        limit = MAX_EVIDENCE_JSON_BYTES if suffix == ".json" else MAX_EVIDENCE_BYTES
    else:
        limit = MAX_JSON_BYTES if suffix == ".json" else MAX_SOURCE_BYTES
    if size > limit:
        return "too_large"
    return None


def scan_source_files(
    repo_root: Path, exclude: Sequence[Path] = ()
) -> tuple[list[InputFile], list[tuple[str, int]]]:
    """Walk the repo; `exclude` lists generated files (index, map) that must never be indexed."""
    skipped_paths = {os.path.normpath(str(path)) for path in exclude}
    found: list[InputFile] = []
    too_large: list[tuple[str, int]] = []
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        dirs[:] = sorted(directory for directory in dirs if not is_excluded_dir(directory))
        for filename in sorted(files):
            path = root_path / filename
            if os.path.normpath(str(path)) in skipped_paths:
                continue
            relative = path.relative_to(repo_root).as_posix()
            try:
                info = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            reason = source_skip_reason(relative, filename, info.st_size)
            if reason == "too_large":
                too_large.append((relative, info.st_size))
            if reason:
                continue
            found.append(InputFile(relative, path, "source", info.st_size, info.st_mtime_ns))
    return drop_evidence_source_copies(found), too_large


def drop_evidence_source_copies(items: list[InputFile]) -> list[InputFile]:
    """Remove code/config files under scan_results/ that duplicate a same-named current repo file."""
    current = {
        Path(item.key).name
        for item in items
        if not item.key.startswith(EVIDENCE_DIR + "/") and Path(item.key).suffix.lower() in CODE_COPY_SUFFIXES
    }
    kept = []
    for item in items:
        if item.key.startswith(EVIDENCE_DIR + "/"):
            path = Path(item.key)
            if path.suffix.lower() in CPP_SUFFIXES or (
                path.suffix.lower() in CODE_COPY_SUFFIXES and path.name in current
            ):
                continue
        kept.append(item)
    return kept


def iter_source_files(repo_root: Path) -> Iterator[Path]:
    for item in scan_source_files(repo_root)[0]:
        yield item.path


def read_source_file(
    item: InputFile,
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[list[Document], set[str], str] | None:
    try:
        data = item.path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:4096]:
        return None
    digest = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8", errors="replace")
    references: set[str] = set()
    for pattern in REFERENCE_PATTERNS:
        references.update(pattern.findall(text))
    kind = "project_document" if item.path.suffix.lower() in {".md", ".txt"} else "source_code"
    documents: list[Document] = []
    for symbol, start, end, chunk in source_chunks(item.key, text):
        locator = f"lines:{start}-{end}"
        if symbol:
            locator = f"symbol:{symbol} {locator}"
        documents.append(
            make_document(kind=kind, source=item.key, locator=locator, content=chunk, topics=topics, symbol=symbol)
        )
    return documents, references, digest


def read_source_documents(
    repo_root: Path,
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[list[Document], dict[str, set[str]]]:
    documents: list[Document] = []
    references: dict[str, set[str]] = defaultdict(set)
    for item in scan_source_files(repo_root)[0]:
        result = read_source_file(item, topics)
        if result is None:
            continue
        documents.extend(result[0])
        references[item.key].update(result[1])
    return documents, references


def curated_documents(
    knowledge_path: Path,
    knowledge: dict[str, Any],
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> list[Document]:
    source = str(knowledge_path)
    documents: list[Document] = []
    for index, fact in enumerate(knowledge.get("verified_facts", []), 1):
        documents.append(
            make_document(
                kind="verified_fact",
                source=source,
                locator=f"verified_fact:{index}",
                content=str(fact),
                topics=topics,
            )
        )
    for index, issue in enumerate(knowledge.get("unresolved_items", []), 1):
        documents.append(
            make_document(
                kind="unresolved_item",
                source=source,
                locator=f"unresolved_item:{index}",
                content=str(issue),
                topics=topics,
            )
        )
    for index, improvement in enumerate(knowledge.get("improvements", []), 1):
        content = (
            f"Improvement: {improvement['id']}\n"
            f"Problem: {improvement['problem']}\n"
            f"Action: {improvement['action']}\n"
            f"Benefit: {improvement['benefit']}"
        )
        documents.append(
            make_document(
                kind="improvement",
                source=source,
                locator=f"improvement:{index}",
                content=content,
                topics=topics,
            )
        )
    return documents


# --------------------------------------------------------------------------- database


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE documents(
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            locator TEXT NOT NULL,
            session_id TEXT,
            timestamp TEXT,
            role TEXT,
            topics TEXT NOT NULL,
            content TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            symbol TEXT
        );
        -- rowid of documents_fts equals rowid of documents
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            content,
            source,
            topics,
            symbol,
            tokenize='porter unicode61'
        );
        CREATE TABLE sessions(
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            parent_session_id TEXT,
            cwd TEXT NOT NULL,
            worktree TEXT,
            title TEXT NOT NULL,
            created TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            user_messages INTEGER NOT NULL,
            assistant_messages INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            format TEXT NOT NULL
        );
        CREATE TABLE file_state(
            key TEXT PRIMARY KEY,
            grp TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT,
            matched INTEGER NOT NULL
        );
        CREATE TABLE refs(source TEXT NOT NULL, reference TEXT NOT NULL, PRIMARY KEY(source, reference));
        CREATE TABLE nodes(
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            label TEXT NOT NULL,
            metadata TEXT NOT NULL
        );
        CREATE TABLE edges(
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL NOT NULL,
            metadata TEXT NOT NULL,
            PRIMARY KEY(source_id, target_id, type)
        );
        CREATE INDEX documents_source_idx ON documents(source);
        CREATE INDEX documents_session_idx ON documents(session_id);
        CREATE INDEX sessions_source_idx ON sessions(source);
        CREATE INDEX edges_source_idx ON edges(source_id);
        CREATE INDEX edges_target_idx ON edges(target_id);
        """
    )


def insert_document(connection: sqlite3.Connection, document: Document, fts_source: str | None = None) -> bool:
    topics_json = json.dumps(document.topics, separators=(",", ":"))
    cursor = connection.execute(
        "INSERT OR IGNORE INTO documents"
        "(id,kind,source,locator,session_id,timestamp,role,topics,content,content_sha256,symbol)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            document.id,
            document.kind,
            document.source,
            document.locator,
            document.session_id,
            document.timestamp,
            document.role,
            topics_json,
            document.content,
            document.content_sha256,
            document.symbol,
        ),
    )
    if cursor.rowcount != 1:
        return False
    connection.execute(
        "INSERT INTO documents_fts(rowid,content,source,topics,symbol) VALUES(?,?,?,?,?)",
        (
            cursor.lastrowid,
            document.content,
            document.source if fts_source is None else fts_source,
            " ".join(document.topics),
            symbol_search_text(document.symbol),
        ),
    )
    return True


def delete_source_rows(connection: sqlite3.Connection, key: str) -> int:
    """Delete every row derived from one file (FTS rows included); return the document count removed."""
    count = connection.execute("SELECT COUNT(*) FROM documents WHERE source=?", (key,)).fetchone()[0]
    connection.execute(
        "DELETE FROM documents_fts WHERE rowid IN (SELECT rowid FROM documents WHERE source=?)", (key,)
    )
    connection.execute("DELETE FROM documents WHERE source=?", (key,))
    connection.execute("DELETE FROM refs WHERE source=?", (key,))
    connection.execute("DELETE FROM sessions WHERE source=?", (key,))
    connection.execute("DELETE FROM file_state WHERE key=?", (key,))
    return count


def insert_session(connection: sqlite3.Connection, info: SessionInfo) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            info.id,
            info.source,
            info.kind,
            info.parent_session_id,
            info.cwd,
            info.worktree,
            info.title,
            info.created,
            info.last_activity,
            info.user_messages,
            info.assistant_messages,
            info.document_count,
            info.format,
        ),
    )


def session_fts_source(info: SessionInfo) -> str:
    return " ".join(part for part in (info.title, info.worktree or "") if part)


def index_file(
    connection: sqlite3.Connection,
    item: InputFile,
    config: IndexConfig,
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[int, str | None]:
    """Read one input file and insert its rows; return (documents inserted, content sha256 for sources)."""
    if item.group == "source":
        result = read_source_file(item, topics)
        if result is None:
            return 0, None
        documents, references, digest = result
        count = sum(insert_document(connection, document) for document in documents)
        connection.executemany(
            "INSERT OR IGNORE INTO refs VALUES(?,?)", [(item.key, reference) for reference in sorted(references)]
        )
        return count, digest
    if item.group == "omp":
        info, documents = read_session(item.path, topics, config.worktree_roots)
    else:
        info, documents = read_claude_session(item.path, topics, config.worktree_roots)
    fts_source = session_fts_source(info)
    count = sum(insert_document(connection, document, fts_source) for document in documents)
    info.document_count = count
    insert_session(connection, info)
    return count, None


def add_node(connection: sqlite3.Connection, node_id: str, node_type: str, label: str, metadata: dict[str, Any]) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO nodes VALUES(?,?,?,?)",
        (node_id, node_type, label, json.dumps(metadata, sort_keys=True)),
    )


def add_edge(
    connection: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO edges VALUES(?,?,?,?,?)",
        (source_id, target_id, edge_type, weight, json.dumps(metadata or {}, sort_keys=True)),
    )


def load_sessions(connection: sqlite3.Connection) -> list[SessionInfo]:
    rows = connection.execute(
        "SELECT id,source,cwd,title,created,last_activity,kind,parent_session_id,user_messages,"
        "assistant_messages,document_count,worktree,format FROM sessions"
    ).fetchall()
    return [SessionInfo(*row) for row in rows]


def rebuild_graph(
    connection: sqlite3.Connection,
    repo_root: Path,
    knowledge: dict[str, Any],
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> str | None:
    """Recompute nodes/edges from the documents, sessions and refs tables; return the newest session id."""
    connection.execute("DELETE FROM nodes")
    connection.execute("DELETE FROM edges")
    topic_counts: Counter[str] = Counter()
    for topic, count in connection.execute(
        "SELECT j.value, COUNT(*) FROM documents d, json_each(d.topics) j GROUP BY j.value"
    ):
        topic_counts[topic] = count
    for (source,) in connection.execute("SELECT DISTINCT source FROM documents UNION SELECT DISTINCT source FROM refs").fetchall():
        add_node(connection, f"source:{source}", "source", source, {})
    for topic_id, _ in topics:
        add_node(connection, f"topic:{topic_id}", "topic", topic_id, {"document_count": topic_counts[topic_id]})
    for source, reference in connection.execute("SELECT source, reference FROM refs ORDER BY source, reference").fetchall():
        reference_node = f"reference:{reference}"
        add_node(connection, reference_node, "declared_reference", reference, {})
        add_edge(connection, f"source:{source}", reference_node, "DECLARES_REFERENCE")
    for source, topic, count in connection.execute(
        "SELECT d.source, j.value, COUNT(*) FROM documents d, json_each(d.topics) j GROUP BY d.source, j.value"
    ).fetchall():
        add_edge(connection, f"source:{source}", f"topic:{topic}", "MENTIONS", float(count))
    session_topics: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for session_id, topic, count in connection.execute(
        "SELECT d.session_id, j.value, COUNT(*) FROM documents d, json_each(d.topics) j "
        "WHERE d.session_id IS NOT NULL GROUP BY d.session_id, j.value"
    ).fetchall():
        session_topics[session_id].append((topic, count))

    sessions = load_sessions(connection)
    top_level = [session for session in sessions if session.kind == "top_level"]
    newest = max(top_level, key=lambda item: item.created, default=None)
    for session in sessions:
        recommendation = (
            "KEEP_WITH_PARENT"
            if session.kind == "subagent"
            else "KEEP_NEWEST"
            if newest and session.id == newest.id
            else "DELETE_SAFE_CANDIDATE"
            if session.knowledge_events <= 1 and session.assistant_messages == 0
            else "KEEP_ARCHIVE_AND_INDEX"
        )
        session_metadata = asdict(session) | {
            "knowledge_events": session.knowledge_events,
            "recommendation": recommendation,
        }
        add_node(connection, f"session:{session.id}", "session", session.title or "Untitled", session_metadata)
        add_edge(connection, f"session:{session.id}", f"source:{session.source}", "STORED_IN")
        for topic, count in session_topics.get(session.id, ()):
            add_edge(connection, f"session:{session.id}", f"topic:{topic}", "DISCUSSES", float(count))
        if session.parent_session_id:
            add_edge(connection, f"session:{session.id}", f"session:{session.parent_session_id}", "SUBAGENT_OF")
        elif newest and session.id != newest.id:
            relation = "NO_MEANINGFUL_RELATION" if recommendation == "DELETE_SAFE_CANDIDATE" else "HISTORICAL_PROJECT_KNOWLEDGE"
            weight = 0.0 if recommendation == "DELETE_SAFE_CANDIDATE" else 1.0
            add_edge(connection, f"session:{newest.id}", f"session:{session.id}", relation, weight)

    for improvement in knowledge.get("improvements", []):
        node_id = f"improvement:{improvement['id']}"
        add_node(connection, node_id, "improvement", improvement["id"], improvement)
        add_edge(connection, "project:current", node_id, "SHOULD_APPLY", 1.0)
    add_node(connection, "project:current", "project", repo_root.name, {"root": str(repo_root)})
    return newest.id if newest else None


def inventory(connection: sqlite3.Connection) -> dict[str, Any]:
    kind_counts = dict(connection.execute("SELECT kind, COUNT(*) FROM documents GROUP BY kind ORDER BY kind").fetchall())
    topic_counts = dict(
        connection.execute(
            "SELECT j.value, COUNT(*) FROM documents d, json_each(d.topics) j GROUP BY j.value ORDER BY j.value"
        ).fetchall()
    )
    session_kinds = dict(connection.execute("SELECT kind, COUNT(*) FROM sessions GROUP BY kind").fetchall())
    session_formats = Counter()
    for fmt, kind, count in connection.execute("SELECT format, kind, COUNT(*) FROM sessions GROUP BY format, kind"):
        session_formats[f"{fmt}_{kind}"] = count
    source_files = connection.execute(
        "SELECT COUNT(DISTINCT source) FROM documents WHERE kind IN ('source_code','project_document')"
    ).fetchone()[0]
    return {
        "session_count": sum(session_kinds.values()),
        "top_level_session_count": session_kinds.get("top_level", 0),
        "subagent_session_count": session_kinds.get("subagent", 0),
        "session_counts_by_format": dict(sorted(session_formats.items())),
        "source_file_count": source_files,
        "document_count": sum(kind_counts.values()),
        "kind_counts": kind_counts,
        "topic_counts": topic_counts,
    }


def write_map(
    connection: sqlite3.Connection,
    config: IndexConfig,
    knowledge: dict[str, Any],
    newest_session_id: str | None,
    generated_at: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    recommendations = [
        json.loads(metadata)
        for (metadata,) in connection.execute(
            "SELECT metadata FROM nodes WHERE type='session' ORDER BY json_extract(metadata,'$.created') DESC"
        )
    ]
    map_value = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "repo_root": str(config.repo_root),
        "index": str(config.index_path),
        "newest_session_id": newest_session_id,
        "inventory": inventory(connection),
        "last_run": summary,
        "sessions": recommendations,
        "improvements": knowledge.get("improvements", []),
        "light_rag": {
            "repository": "https://github.com/HKUDS/LightRAG",
            "api": {
                "insert": "POST /documents/texts",
                "query": "POST /query",
                "authentication": "Optional X-API-Key from LIGHTRAG_API_KEY",
                "default_url": "http://127.0.0.1:9621",
            },
            "local_only_default": True,
        },
    }
    config.map_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_map = config.map_path.with_suffix(config.map_path.suffix + ".tmp")
    temporary_map.write_text(json.dumps(map_value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_map, config.map_path)
    return map_value


@contextmanager
def index_lock(index_path: Path, wait: float = 0.0) -> Iterator[None]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(index_path.with_name(index_path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + wait
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise IndexLocked(f"another index build/refresh holds {index_path}.lock") from None
                time.sleep(0.2)
        yield
    finally:
        os.close(descriptor)  # closing releases the flock


def resolve_config(
    repo_root: Path,
    session_root: Path | Sequence[Path] | None,
    index_path: Path,
    map_path: Path,
    knowledge_path: Path,
    claude_roots: Sequence[Path] = (),
    worktree_roots: Sequence[Path] | None = None,
) -> IndexConfig:
    repo_root = repo_root.resolve()
    if session_root is None:
        omp_roots: tuple[Path, ...] = ()
    elif isinstance(session_root, (str, Path)):
        omp_roots = (Path(session_root).expanduser(),)
    else:
        omp_roots = tuple(Path(item).expanduser() for item in session_root)
    return IndexConfig(
        repo_root=repo_root,
        omp_roots=omp_roots,
        claude_roots=tuple(Path(item).expanduser() for item in claude_roots),
        worktree_roots=tuple(
            Path(item).expanduser() for item in (default_worktree_roots(repo_root) if worktree_roots is None else worktree_roots)
        ),
        knowledge_path=knowledge_path.resolve(),
        index_path=index_path if index_path.is_absolute() else repo_root / index_path,
        map_path=map_path if map_path.is_absolute() else repo_root / map_path,
    )


def collect_inputs(
    config: IndexConfig, known: dict[str, tuple[int, int, bool]] | None = None
) -> tuple[list[InputFile], list[tuple[str, int]]]:
    generated = [config.index_path, config.map_path, config.map_path.with_suffix(config.map_path.suffix + ".tmp")]
    sources, too_large = scan_source_files(config.repo_root, generated)
    sessions = scan_omp_files(config.omp_roots, config.repo_root, config.worktree_roots, known)
    sessions += scan_claude_files(config.claude_roots, config.repo_root, config.worktree_roots, known)
    return sources + sessions, too_large


def report_too_large(too_large: Sequence[tuple[str, int]], quiet: bool) -> None:
    outside = [item for item in too_large if not item[0].startswith(EVIDENCE_DIR + "/")]
    for relative, size in outside[:20]:
        log(f"skipped (too large, {size} bytes): {relative}", quiet)
    if len(outside) > 20:
        log(f"... and {len(outside) - 20} more large files skipped", quiet)
    evidence = len(too_large) - len(outside)
    if evidence:
        log(f"skipped {evidence} oversize files under {EVIDENCE_DIR}/ (json > {MAX_EVIDENCE_JSON_BYTES} bytes, others > {MAX_EVIDENCE_BYTES})", quiet)


def store_file_state(connection: sqlite3.Connection, item: InputFile, digest: str | None) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO file_state VALUES(?,?,?,?,?,?)",
        (item.key, item.group, item.size, item.mtime_ns, digest, 1 if item.matched else 0),
    )


def finish_index(
    connection: sqlite3.Connection,
    config: IndexConfig,
    knowledge: dict[str, Any],
    topics: Sequence[tuple[str, re.Pattern[str]]],
    summary: dict[str, Any],
    full: bool,
) -> dict[str, Any]:
    generated_at = utc_now()
    newest = rebuild_graph(connection, config.repo_root, knowledge, topics)
    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "generated_at": generated_at,
        "repo_root": str(config.repo_root),
        "session_root": ",".join(str(root) for root in config.omp_roots),
        "claude_root": ",".join(str(root) for root in config.claude_roots),
        "knowledge_path": str(config.knowledge_path),
        "knowledge_sha256": sha256_text(config.knowledge_path.read_text(encoding="utf-8")),
        "config_fingerprint": config.fingerprint(),
    }
    if full:
        metadata["full_build_at"] = generated_at
    connection.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)", metadata.items())
    connection.commit()
    return write_map(connection, config, knowledge, newest, generated_at, summary)


def populate(
    connection: sqlite3.Connection,
    config: IndexConfig,
    topics: Sequence[tuple[str, re.Pattern[str]]],
    pending: Sequence[InputFile],
    quiet: bool,
) -> int:
    documents = 0
    started = time.monotonic()
    for index, item in enumerate(pending, 1):
        if item.matched:
            count, digest = index_file(connection, item, config, topics)
            documents += count
        else:
            digest = None
        store_file_state(connection, item, digest)
        if not quiet and index % 500 == 0:
            log(f"  indexed {index}/{len(pending)} files, {documents} documents, {time.monotonic() - started:.0f}s")
    return documents


def build_index(
    repo_root: Path,
    session_root: Path | Sequence[Path] | None,
    index_path: Path,
    map_path: Path,
    knowledge_path: Path,
    *,
    claude_roots: Sequence[Path] = (),
    worktree_roots: Sequence[Path] | None = None,
    quiet: bool = True,
    wait: float = 0.0,
    locked: bool = False,
) -> dict[str, Any]:
    """Full rebuild into a temporary database that atomically replaces the index."""
    config = resolve_config(repo_root, session_root, index_path, map_path, knowledge_path, claude_roots, worktree_roots)
    if locked:
        return _build_full(config, quiet)
    with index_lock(config.index_path, wait):
        return _build_full(config, quiet)


def _build_full(config: IndexConfig, quiet: bool) -> dict[str, Any]:
    started = time.monotonic()
    knowledge = load_knowledge(config.knowledge_path)
    topics = compile_topics(knowledge)
    inputs, too_large = collect_inputs(config)
    report_too_large(too_large, quiet)
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=config.index_path.name + ".", suffix=".tmp", dir=config.index_path.parent
    )
    os.close(descriptor)
    temporary_index = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary_index)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            create_schema(connection)
            documents = populate(connection, config, topics, [item for item in inputs], quiet)
            for document in curated_documents(config.knowledge_path, knowledge, topics):
                documents += insert_document(connection, document)
            matched = [item for item in inputs if item.matched]
            summary = {
                "mode": "full",
                "files_added": len(matched),
                "files_changed": 0,
                "files_removed": 0,
                "files_unmatched": len(inputs) - len(matched),
                "documents_added": documents,
                "documents_removed": 0,
                "skipped_too_large": len(too_large),
                "seconds": 0.0,
            }
            summary["seconds"] = round(time.monotonic() - started, 2)
            map_value = finish_index(connection, config, knowledge, topics, summary, full=True)
            summary["seconds"] = round(time.monotonic() - started, 2)
            map_value["last_run"] = summary
        finally:
            connection.close()
        os.replace(temporary_index, config.index_path)
    except BaseException:
        temporary_index.unlink(missing_ok=True)
        raise
    return map_value


def read_state(connection: sqlite3.Connection) -> dict[str, tuple[int, int, bool, str | None, str]]:
    return {
        key: (size, mtime_ns, bool(matched), sha, grp)
        for key, grp, size, mtime_ns, sha, matched in connection.execute(
            "SELECT key,grp,size,mtime_ns,sha256,matched FROM file_state"
        )
    }


def index_metadata(index_path: Path) -> dict[str, str] | None:
    if not index_path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
        try:
            return dict(connection.execute("SELECT key,value FROM metadata").fetchall())
        finally:
            connection.close()
    except sqlite3.Error:
        return None


def needs_full_build(config: IndexConfig) -> str | None:
    metadata = index_metadata(config.index_path)
    if metadata is None:
        return "no usable index"
    if metadata.get("schema_version") != str(SCHEMA_VERSION):
        return f"schema {metadata.get('schema_version')} != {SCHEMA_VERSION}"
    if metadata.get("config_fingerprint") != config.fingerprint():
        return "roots or repository changed"
    try:
        knowledge_sha = sha256_text(config.knowledge_path.read_text(encoding="utf-8"))
    except OSError:
        return "knowledge file unreadable"
    if metadata.get("knowledge_sha256") != knowledge_sha:
        return "knowledge file changed"
    return None


def diff_inputs(
    inputs: Sequence[InputFile],
    state: dict[str, tuple[int, int, bool, str | None, str]],
) -> tuple[list[InputFile], list[InputFile], list[str]]:
    """Return (added, changed, removed keys) by size/mtime; source hash filtering happens on apply."""
    added: list[InputFile] = []
    changed: list[InputFile] = []
    seen: set[str] = set()
    for item in inputs:
        seen.add(item.key)
        previous = state.get(item.key)
        if previous is None:
            added.append(item)
        elif previous[0] != item.size or previous[1] != item.mtime_ns:
            changed.append(item)
        elif previous[2] != item.matched:
            changed.append(item)
    removed = [key for key in state if key not in seen]
    return added, changed, removed


def pending_changes(config: IndexConfig) -> dict[str, Any] | None:
    """Counts of files that a refresh would touch, or None when there is no usable index."""
    if not config.index_path.is_file() or needs_full_build(config):
        return None
    connection = sqlite3.connect(f"file:{config.index_path}?mode=ro", uri=True)
    try:
        state = read_state(connection)
    finally:
        connection.close()
    known = {key: (value[0], value[1], value[2]) for key, value in state.items()}
    inputs, _ = collect_inputs(config, known)
    added, changed, removed = diff_inputs(inputs, state)
    return {
        "added": sum(1 for item in added if item.matched),
        "changed": len(changed),
        "removed": len(removed),
    }


def refresh_index(
    repo_root: Path,
    session_root: Path | Sequence[Path] | None,
    index_path: Path,
    map_path: Path,
    knowledge_path: Path,
    *,
    claude_roots: Sequence[Path] = (),
    worktree_roots: Sequence[Path] | None = None,
    full: bool = False,
    quiet: bool = True,
    wait: float = 0.0,
) -> dict[str, Any]:
    """Incrementally re-read new/changed/deleted files; falls back to a full build when needed."""
    config = resolve_config(repo_root, session_root, index_path, map_path, knowledge_path, claude_roots, worktree_roots)
    with index_lock(config.index_path, wait):
        reason = "forced" if full else needs_full_build(config)
        if reason:
            log(f"full rebuild ({reason})", quiet)
            return _build_full(config, quiet)
        started = time.monotonic()
        knowledge = load_knowledge(config.knowledge_path)
        topics = compile_topics(knowledge)
        connection = sqlite3.connect(config.index_path, timeout=60)
        try:
            state = read_state(connection)
            known = {key: (value[0], value[1], value[2]) for key, value in state.items()}
            inputs, too_large = collect_inputs(config, known)
            if too_large:
                log(f"{len(too_large)} files skipped as too large (`build` lists them)", quiet)
            added, changed, removed = diff_inputs(inputs, state)
            summary = {
                "mode": "refresh",
                "files_added": 0,
                "files_changed": 0,
                "files_removed": 0,
                "files_unchanged_by_hash": 0,
                "documents_added": 0,
                "documents_removed": 0,
                "skipped_too_large": len(too_large),
                "seconds": 0.0,
            }
            connection.execute("BEGIN")
            try:
                for key in removed:
                    if state[key][2]:
                        summary["files_removed"] += 1
                    summary["documents_removed"] += delete_source_rows(connection, key)
                for item in added + changed:
                    previous = state.get(item.key)
                    was_matched = bool(previous and previous[2])
                    if (
                        previous is not None
                        and item.group == "source"
                        and was_matched
                        and item.matched
                        and previous[3]
                        and hashlib.sha256(item.path.read_bytes()).hexdigest() == previous[3]
                    ):
                        store_file_state(connection, item, previous[3])
                        summary["files_unchanged_by_hash"] += 1
                        continue
                    if previous is not None:
                        summary["documents_removed"] += delete_source_rows(connection, item.key)
                    if item.matched:
                        summary["files_changed" if was_matched else "files_added"] += 1
                    elif was_matched:
                        summary["files_removed"] += 1
                    summary["documents_added"] += populate(connection, config, topics, [item], True)
                touched = bool(removed or summary["files_added"] or summary["files_changed"])
                summary["seconds"] = round(time.monotonic() - started, 2)
                connection.execute("INSERT OR REPLACE INTO metadata VALUES('last_refresh_at',?)", (utc_now(),))
                if touched:
                    map_value = finish_index(connection, config, knowledge, topics, summary, full=False)
                else:
                    connection.commit()
                    map_value = {"inventory": inventory(connection)}
            except BaseException:
                connection.rollback()
                raise
            summary["seconds"] = round(time.monotonic() - started, 2)
            map_value["last_run"] = summary
            return map_value
        finally:
            connection.close()


# --------------------------------------------------------------------------- query


def query_tokens(query: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(query.lower()):
        token = token.strip("./:+-")
        if len(token) < 2 or token in STOP_WORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def fts_expression(query: str) -> str:
    tokens = query_tokens(query)
    if not tokens:
        raise ValueError("Query has no searchable terms")
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def expand_kinds(values: Sequence[str]) -> list[str]:
    """Accept raw kinds plus aliases (source, docs, session, curated) and comma-separated lists."""
    result: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            for kind in KIND_ALIASES.get(item.lower(), (item,)):
                if kind not in result:
                    result.append(kind)
    return result


def parse_since(value: str | None) -> str | None:
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"--since must be YYYY-MM-DD, got {value!r}") from error
    return value


def relevance_weight(item: dict[str, Any]) -> float:
    """Multiplier applied to -bm25: current source/docs outrank session turns and old evidence at equal relevance."""
    if item["kind"] in SESSION_KINDS:
        return 0.78 if item.get("session_kind") == "subagent" else 0.85
    if item["source"].startswith(EVIDENCE_DIR + "/"):
        return 0.8
    return 1.0


def name_bonus(item: dict[str, Any], terms: set[str]) -> float:
    """Up to +35% when query terms appear in the file path, symbol name or session title."""
    if not terms:
        return 1.0
    if item["kind"] in SESSION_KINDS:
        names = re.findall(r"[a-z0-9]+", (item.get("session_title") or "").lower())
    else:
        names = re.findall(r"[a-z0-9]+", item["source"].lower())
        symbol = item.get("symbol") or ""
        names += re.findall(r"[a-z0-9]+", symbol.lower()) + camel_split(symbol).split()
    return 1.0 + 0.35 * len(terms.intersection(names)) / len(terms)


def query_database(
    index_path: Path,
    query: str,
    top_k: int = 8,
    kinds: Sequence[str] = (),
    since: str | None = None,
) -> list[dict[str, Any]]:
    kinds = expand_kinds(kinds)
    since = parse_since(since)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        where = "documents_fts MATCH ?"
        parameters: list[Any] = [fts_expression(query)]
        if kinds:
            where += " AND d.kind IN (" + ",".join("?" for _ in kinds) + ")"
            parameters.extend(kinds)
        if since:
            where += " AND (d.kind <> 'session_turn' OR substr(COALESCE(d.timestamp, s.created), 1, 10) >= ?)"
            parameters.append(since)
        parameters.append(max(top_k * 40, 300))
        rows = connection.execute(
            f"""
            SELECT d.id, d.kind, d.source, d.locator, d.session_id, d.timestamp, d.role, d.topics,
                   d.content, d.content_sha256, d.symbol,
                   s.title AS session_title, s.worktree AS session_worktree, s.kind AS session_kind,
                   s.created AS session_created,
                   bm25(documents_fts, 10.0, 4.0, 3.0, 12.0) AS rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            LEFT JOIN sessions s ON s.id = d.session_id
            WHERE {where}
            ORDER BY rank ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    finally:
        connection.close()

    priority = {"verified_fact": 0.35, "improvement": 0.25, "unresolved_item": 0.20}
    terms = {part for token in query_tokens(query) for part in re.findall(r"[a-z0-9]+", token)}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["topics"] = json.loads(item["topics"])
        relevance = -float(item.pop("rank")) * relevance_weight(item) * name_bonus(item, terms)
        item["score"] = round(relevance + priority.get(item["kind"], 0.0), 6)
        candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_session_text: set[str] = set()
    per_source: Counter[str] = Counter()
    for item in candidates:
        if item["kind"] in SESSION_KINDS:
            if item["content_sha256"] in seen_session_text:
                continue  # forked/resumed sessions copy earlier turns
        if per_source[item["source"]] >= 3:
            continue
        selected.append(item)
        selected_ids.add(item["id"])
        per_source[item["source"]] += 1
        if item["kind"] in SESSION_KINDS:
            seen_session_text.add(item["content_sha256"])
        if len(selected) == top_k:
            break
    for item in candidates:
        if len(selected) == top_k:
            break
        if item["id"] not in selected_ids and not (
            item["kind"] in SESSION_KINDS and item["content_sha256"] in seen_session_text
        ):
            selected.append(item)
            selected_ids.add(item["id"])
            if item["kind"] in SESSION_KINDS:
                seen_session_text.add(item["content_sha256"])
    return selected


def render_results(query: str, results: Sequence[dict[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps({"query": query, "results": list(results)}, indent=2)
    heading = "Retrieved project evidence" if output_format == "text" else "RAG context for the requested task"
    lines = [heading, f"Query: {query}", ""]
    if output_format == "prompt":
        lines.extend(
            [
                "Use only relevant evidence below. Verify current source before editing; historical session claims can be stale.",
                "Distinguish observed, verified, inferred, unresolved, simulation-only, and physically executed evidence.",
                "",
            ]
        )
    for index, item in enumerate(results, 1):
        metadata = [item["kind"], item["source"], item["locator"]]
        if item["kind"] in SESSION_KINDS:
            stamp = item.get("timestamp") or item.get("session_created") or ""
            if stamp:
                metadata.append(stamp[:10])
            if item.get("session_title"):
                metadata.append('session="' + one_line(item["session_title"], 60) + '"')
            if item.get("session_worktree"):
                metadata.append("worktree=" + item["session_worktree"])
            if item.get("session_kind") == "subagent":
                metadata.append("subagent")
            if item.get("role"):
                metadata.append("role=" + item["role"])
        elif item.get("timestamp"):
            metadata.append(item["timestamp"])
        if item.get("topics"):
            metadata.append("topics=" + ",".join(item["topics"]))
        lines.append(f"[{index}] " + " | ".join(metadata))
        lines.append(item["content"])
        lines.append("")
    return "\n".join(lines).rstrip()


def graph_neighbors(index_path: Path, node_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        node = connection.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if node is None:
            raise KeyError(node_id)
        edges = connection.execute(
            """
            SELECT e.*, n.type AS neighbor_type, n.label AS neighbor_label, n.metadata AS neighbor_metadata
            FROM edges e
            JOIN nodes n ON n.id = CASE WHEN e.source_id=? THEN e.target_id ELSE e.source_id END
            WHERE e.source_id=? OR e.target_id=?
            ORDER BY e.weight DESC, e.type, neighbor_label
            """,
            (node_id, node_id, node_id),
        ).fetchall()
    finally:
        connection.close()
    return {
        "node": dict(node) | {"metadata": json.loads(node["metadata"])},
        "edges": [dict(row) | {"metadata": json.loads(row["metadata"]), "neighbor_metadata": json.loads(row["neighbor_metadata"])} for row in edges],
    }


def session_recommendations(index_path: Path, since: str | None = None, include_subagents: bool = True) -> list[dict[str, Any]]:
    since = parse_since(since)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT id,label,metadata FROM nodes WHERE type='session' ORDER BY json_extract(metadata,'$.created') DESC").fetchall()
    finally:
        connection.close()
    result = [dict(row) | {"metadata": json.loads(row["metadata"])} for row in rows]
    if since:
        result = [item for item in result if (item["metadata"].get("last_activity") or item["metadata"].get("created") or "")[:10] >= since]
    if not include_subagents:
        result = [item for item in result if item["metadata"].get("kind") != "subagent"]
    return result


def require_safe_url(base_url: str, allow_remote: bool) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid LightRAG URL: {base_url}")
    if not allow_remote and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Remote LightRAG endpoints require --allow-remote")
    return base_url.rstrip("/")


def api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    api_key = os.environ.get("LIGHTRAG_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def request_json(base_url: str, route: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        base_url + route,
        data=json.dumps(payload).encode("utf-8"),
        headers=api_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise LightRagHttpError(error.code, detail) from error
    return json.loads(body) if body else {}


def lightrag_documents(index_path: Path, kinds: Sequence[str]) -> list[tuple[str, str]]:
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        where = ""
        parameters: list[Any] = []
        if kinds:
            where = "WHERE kind IN (" + ",".join("?" for _ in kinds) + ")"
            parameters.extend(kinds)
        rows = connection.execute(
            f"SELECT id,kind,source,locator,topics,content FROM documents {where} ORDER BY kind,source,locator",
            parameters,
        ).fetchall()
    finally:
        connection.close()
    documents: list[tuple[str, str]] = []
    for row in rows:
        source_name = f"omp-rag-{row['id'][:24]}.txt"
        text = (
            f"Kind: {row['kind']}\nSource: {row['source']}\nLocator: {row['locator']}\n"
            f"Topics: {', '.join(json.loads(row['topics']))}\n\n{row['content']}"
        )
        documents.append((source_name, text))
    return documents


def ingest_lightrag(
    index_path: Path,
    base_url: str,
    allow_remote: bool,
    kinds: Sequence[str],
    batch_size: int,
    timeout: float,
    dry_run: bool,
) -> dict[str, Any]:
    base_url = require_safe_url(base_url, allow_remote)
    documents = lightrag_documents(index_path, kinds)
    summary = {"endpoint": base_url, "documents": len(documents), "batches": (len(documents) + batch_size - 1) // batch_size, "dry_run": dry_run}
    if dry_run:
        summary["sample_sources"] = [source for source, _ in documents[:3]]
        return summary
    accepted = 0
    skipped_existing = 0
    responses = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        try:
            responses.append(
                request_json(
                    base_url,
                    "/documents/texts",
                    {"texts": [text for _, text in batch], "file_sources": [source for source, _ in batch]},
                    timeout,
                )
            )
            accepted += len(batch)
        except LightRagHttpError as error:
            if error.status != 409:
                raise
            for source, text in batch:
                try:
                    responses.append(
                        request_json(base_url, "/documents/text", {"text": text, "file_source": source}, timeout)
                    )
                    accepted += 1
                except LightRagHttpError as single_error:
                    if single_error.status != 409:
                        raise
                    skipped_existing += 1
    summary["accepted"] = accepted
    summary["skipped_existing"] = skipped_existing
    summary["responses"] = responses
    return summary


def doctor(
    repo_root: Path,
    session_root: Path | Sequence[Path] | None,
    index_path: Path,
    claude_roots: Sequence[Path] = (),
    worktree_roots: Sequence[Path] | None = None,
    knowledge_path: Path = DEFAULT_KNOWLEDGE,
    map_path: Path = DEFAULT_MAP,
) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(content)")
        fts5 = True
    except sqlite3.OperationalError:
        fts5 = False
    finally:
        connection.close()
    config = resolve_config(repo_root, session_root, index_path, map_path, knowledge_path, claude_roots, worktree_roots)

    omp_report = []
    for root in config.omp_roots:
        files = scan_omp_files([root], config.repo_root, config.worktree_roots)
        omp_report.append(
            {
                "path": str(root),
                "exists": root.is_dir(),
                "files": len(files),
                "matching_sessions": sum(1 for item in files if item.matched),
            }
        )
    claude_report = []
    for root in config.claude_roots:
        files = scan_claude_files([root], config.repo_root, config.worktree_roots)
        matched = [item for item in files if item.matched]
        claude_report.append(
            {
                "path": str(root),
                "exists": root.is_dir(),
                "files": len(files),
                "matching_top_level": sum(1 for item in matched if item.path.parent.name != "subagents"),
                "matching_subagent": sum(1 for item in matched if item.path.parent.name == "subagents"),
            }
        )

    index_report: dict[str, Any] = {"exists": config.index_path.is_file()}
    metadata = index_metadata(config.index_path)
    if metadata is not None:
        reference = metadata.get("last_refresh_at") or metadata.get("generated_at")
        age = None
        if reference:
            age = round((datetime.now(timezone.utc) - datetime.fromisoformat(reference)).total_seconds())
        reader = sqlite3.connect(f"file:{config.index_path}?mode=ro", uri=True)
        try:
            summary = inventory(reader)
        except sqlite3.Error:
            summary = {}
        finally:
            reader.close()
        rebuild_reason = needs_full_build(config)
        index_report |= {
            "schema_version": metadata.get("schema_version"),
            "generated_at": metadata.get("generated_at"),
            "last_refresh_at": metadata.get("last_refresh_at"),
            "full_build_at": metadata.get("full_build_at"),
            "age_seconds": age,
            "size_bytes": config.index_path.stat().st_size,
            "documents": summary.get("document_count"),
            "kind_counts": summary.get("kind_counts"),
            "sessions_by_format": summary.get("session_counts_by_format"),
            "full_rebuild_needed": rebuild_reason,
            "files_changed_since_last_refresh": None if rebuild_reason else pending_changes(config),
        }
    return {
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "fts5": fts5,
        "repo_root": str(config.repo_root),
        "session_root": str(config.omp_roots[0]) if config.omp_roots else None,
        "omp_roots": omp_report,
        "claude_roots": claude_report,
        "worktree_roots": [str(root) for root in config.worktree_roots],
        "matching_session_files": sum(item["matching_sessions"] for item in omp_report),
        "matching_claude_session_files": sum(
            item["matching_top_level"] + item["matching_subagent"] for item in claude_report
        ),
        "index_exists": config.index_path.exists(),
        "index": index_report,
        "lightrag_url": os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621"),
        "lightrag_api_key_configured": bool(os.environ.get("LIGHTRAG_API_KEY")),
    }


def path_for_repo(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root.resolve() / value


def require_index(index_path: Path) -> None:
    if not index_path.is_file():
        raise FileNotFoundError(f"No index at {index_path}; run `session_rag.py build` first")


def summary_line(result: dict[str, Any]) -> str:
    run = result["last_run"]
    inv = result["inventory"]
    sessions = inv["session_count"]
    if run["mode"] == "full":
        head = f"full build: {run['files_added']} files"
    else:
        head = (
            f"refresh: {run['files_added']} added, {run['files_changed']} changed, "
            f"{run['files_removed']} removed"
        )
    return (
        f"{head}; documents +{run['documents_added']} -{run['documents_removed']} "
        f"(total {inv['document_count']}, {sessions} sessions); {run['seconds']}s"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path.cwd())
    result.add_argument(
        "--session-root",
        dest="session_roots",
        type=Path,
        action="append",
        default=[],
        help=f"omp session root (repeatable; default {DEFAULT_OMP_ROOT})",
    )
    result.add_argument(
        "--claude-root",
        dest="claude_roots",
        type=Path,
        action="append",
        default=[],
        help=f"Claude Code projects root (repeatable; default {DEFAULT_CLAUDE_ROOT})",
    )
    result.add_argument(
        "--worktree-root",
        dest="worktree_roots",
        type=Path,
        action="append",
        default=[],
        help="extra directory whose subdirectories are worktrees of this repo "
        "(defaults: <repo>/.claude/worktrees and ~/orca/workspaces/<repo name>)",
    )
    result.add_argument("--no-omp", action="store_true", help="do not index omp sessions")
    result.add_argument("--no-claude", action="store_true", help="do not index Claude Code sessions")
    result.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    result.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    result.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Rebuild the SQLite index and relationship map atomically (full)")
    build.add_argument("--quiet", "-q", action="store_true")
    build.add_argument("--json", action="store_true", help="print the full run summary as JSON")
    build.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the index lock")

    refresh = subparsers.add_parser(
        "refresh", help="Incrementally re-read new/changed/deleted files (full build if the index is missing or stale)"
    )
    refresh.add_argument("--full", action="store_true", help="force a full rebuild")
    refresh.add_argument("--quiet", "-q", action="store_true")
    refresh.add_argument("--json", action="store_true", help="print the full run summary as JSON")
    refresh.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the index lock")

    query = subparsers.add_parser("query", help="Retrieve local project/session evidence")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=8)
    query.add_argument(
        "--kind",
        action="append",
        default=[],
        help="filter by kind: source, docs, session, curated, or a raw kind such as source_code (repeatable, comma ok)",
    )
    query.add_argument("--since", help="only session turns on/after YYYY-MM-DD (source and docs are unaffected)")
    query.add_argument("--format", choices=("text", "json", "prompt"), default="text")

    graph = subparsers.add_parser("graph", help="Show a node and its graph neighbors")
    graph.add_argument("node_id")

    sessions = subparsers.add_parser("sessions", help="List indexed sessions and retention recommendations")
    sessions.add_argument("--since", help="only sessions with activity on/after YYYY-MM-DD")
    sessions.add_argument("--limit", type=int, default=0, help="show at most N sessions (newest first)")
    sessions.add_argument("--no-subagents", action="store_true")
    subparsers.add_parser("doctor", help="Check prerequisites, session roots and index freshness without network access")

    ingest = subparsers.add_parser("lightrag-ingest", help="Send indexed documents to a LightRAG server")
    ingest.add_argument("--url", default=os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621"))
    ingest.add_argument("--allow-remote", action="store_true")
    ingest.add_argument("--kind", action="append", default=[])
    ingest.add_argument("--batch-size", type=int, default=16)
    ingest.add_argument("--timeout", type=float, default=60.0)
    ingest.add_argument("--dry-run", action="store_true")

    remote_query = subparsers.add_parser("lightrag-query", help="Query a configured LightRAG server")
    remote_query.add_argument("query")
    remote_query.add_argument("--url", default=os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621"))
    remote_query.add_argument("--allow-remote", action="store_true")
    remote_query.add_argument("--mode", choices=("local", "global", "hybrid", "naive", "mix", "bypass"), default="mix")
    remote_query.add_argument("--context-only", action="store_true")
    remote_query.add_argument("--timeout", type=float, default=120.0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    index_path = path_for_repo(repo_root, args.index)
    map_path = path_for_repo(repo_root, args.map_path)
    knowledge_path = args.knowledge if args.knowledge.is_absolute() else path_for_repo(repo_root, args.knowledge)
    omp_roots = [] if args.no_omp else (args.session_roots or [DEFAULT_OMP_ROOT])
    claude_roots = [] if args.no_claude else (args.claude_roots or [DEFAULT_CLAUDE_ROOT])
    worktree_roots = args.worktree_roots or None

    try:
        if args.command in {"build", "refresh"}:
            try:
                if args.command == "build":
                    result = build_index(
                        repo_root, omp_roots, index_path, map_path, knowledge_path,
                        claude_roots=claude_roots, worktree_roots=worktree_roots, quiet=args.quiet, wait=args.wait,
                    )
                else:
                    result = refresh_index(
                        repo_root, omp_roots, index_path, map_path, knowledge_path,
                        claude_roots=claude_roots, worktree_roots=worktree_roots, full=args.full,
                        quiet=args.quiet, wait=args.wait,
                    )
            except IndexLocked as error:
                if args.command == "build":
                    raise
                print(f"refresh skipped: {error}")
                return 0
            if args.json:
                print(json.dumps({"summary": result["last_run"], "inventory": result["inventory"]}, indent=2))
            elif not args.quiet:
                print(summary_line(result))
        elif args.command == "query":
            if args.top_k < 1:
                raise ValueError("--top-k must be positive")
            require_index(index_path)
            print(render_results(args.query, query_database(index_path, args.query, args.top_k, args.kind, args.since), args.format))
        elif args.command == "graph":
            require_index(index_path)
            print(json.dumps(graph_neighbors(index_path, args.node_id), indent=2))
        elif args.command == "sessions":
            require_index(index_path)
            listing = session_recommendations(index_path, args.since, not args.no_subagents)
            if args.limit > 0:
                listing = listing[: args.limit]
            print(json.dumps(listing, indent=2))
        elif args.command == "doctor":
            print(
                json.dumps(
                    doctor(repo_root, omp_roots, index_path, claude_roots, worktree_roots, knowledge_path, map_path),
                    indent=2,
                )
            )
        elif args.command == "lightrag-ingest":
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            require_index(index_path)
            print(json.dumps(ingest_lightrag(index_path, args.url, args.allow_remote, expand_kinds(args.kind), args.batch_size, args.timeout, args.dry_run), indent=2))
        elif args.command == "lightrag-query":
            base_url = require_safe_url(args.url, args.allow_remote)
            payload = {"query": args.query, "mode": args.mode}
            if args.context_only:
                payload["only_need_context"] = True
            print(json.dumps(request_json(base_url, "/query", payload, args.timeout), indent=2))
        else:
            raise AssertionError(args.command)
    except (FileNotFoundError, KeyError, RuntimeError, sqlite3.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
