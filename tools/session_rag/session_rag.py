#!/usr/bin/env python3
"""Build and query a local project/session RAG index.

The local backend is dependency-free SQLite FTS5. Optional LightRAG commands use
its REST API; credentials are read only from the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = 1
DEFAULT_INDEX = Path("workspace_scan_artifacts/project_rag.sqlite3")
DEFAULT_MAP = Path("workspace_scan_artifacts/project_rag_map.json")
DEFAULT_KNOWLEDGE = Path(__file__).with_name("project_knowledge.json")
MAX_SOURCE_BYTES = 512_000
MAX_CHUNK_CHARS = 4_000
OVERLAP_LINES = 4

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "__pycache__",
    "backups",
    "build",
    "install",
    "isaac_exports",
    "log",
    "node_modules",
    "workspace_scan_artifacts",
}
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

    @property
    def knowledge_events(self) -> int:
        return self.user_messages + self.assistant_messages


class LightRagHttpError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"LightRAG HTTP {status}: {detail}")
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


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
    )


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


def discover_session_files(session_root: Path, repo_root: Path) -> list[Path]:
    expected = repo_root.resolve()
    found: list[Path] = []
    if not session_root.is_dir():
        return found
    for path in session_root.rglob("*.jsonl"):
        header = session_header(path)
        if not header or not header.get("cwd"):
            continue
        try:
            cwd = Path(header["cwd"]).expanduser().resolve()
        except OSError:
            continue
        if cwd == expected:
            found.append(path)
    return sorted(found)


def parent_session_id(path: Path) -> str | None:
    candidate = path.parent.parent / f"{path.parent.name}.jsonl"
    if not candidate.is_file():
        return None
    header = session_header(candidate)
    return str(header["id"]) if header and header.get("id") else None


def read_session(
    path: Path,
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[SessionInfo, list[Document]]:
    header = session_header(path)
    if header is None:
        raise ValueError(f"No session header in {path}")
    session_id = str(header["id"])
    title = str(header.get("title") or "")
    created = str(header.get("timestamp") or "")
    last_activity = created
    parent_id = parent_session_id(path)
    info = SessionInfo(
        id=session_id,
        source=str(path),
        cwd=str(header.get("cwd") or ""),
        title=title,
        created=created,
        last_activity=last_activity,
        kind="subagent" if parent_id else "top_level",
        parent_session_id=parent_id,
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
    info.document_count = len(documents)
    return info, documents


def iter_source_files(repo_root: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        dirs[:] = sorted(directory for directory in dirs if directory not in EXCLUDED_DIRS)
        for filename in sorted(files):
            path = root_path / filename
            relative = path.relative_to(repo_root).as_posix()
            if relative.startswith("tools/session_rag/") and (
                filename.startswith("test_") or filename == "project_knowledge.json"
            ):
                continue
            if filename not in SOURCE_NAMES and path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_SOURCE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def read_source_documents(
    repo_root: Path,
    topics: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[list[Document], dict[str, set[str]]]:
    documents: list[Document] = []
    references: dict[str, set[str]] = defaultdict(set)
    for path in iter_source_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in REFERENCE_PATTERNS:
            references[relative].update(pattern.findall(text))
        kind = "project_document" if path.suffix.lower() in {".md", ".txt"} else "source_code"
        for start, end, chunk in chunk_text(text):
            if not chunk:
                continue
            documents.append(
                make_document(
                    kind=kind,
                    source=relative,
                    locator=f"lines:{start}-{end}",
                    content=chunk,
                    topics=topics,
                )
            )
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
            content_sha256 TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            id UNINDEXED,
            content,
            source,
            topics,
            tokenize='porter unicode61'
        );
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
        CREATE INDEX edges_source_idx ON edges(source_id);
        CREATE INDEX edges_target_idx ON edges(target_id);
        """
    )


def insert_document(connection: sqlite3.Connection, document: Document) -> None:
    topics_json = json.dumps(document.topics, separators=(",", ":"))
    connection.execute(
        "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
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
        ),
    )
    connection.execute(
        "INSERT INTO documents_fts(id,content,source,topics) VALUES(?,?,?,?)",
        (document.id, document.content, document.source, " ".join(document.topics)),
    )


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


def build_index(
    repo_root: Path,
    session_root: Path,
    index_path: Path,
    map_path: Path,
    knowledge_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    knowledge_path = knowledge_path.resolve()
    knowledge = load_knowledge(knowledge_path)
    topics = compile_topics(knowledge)

    session_files = discover_session_files(session_root.expanduser(), repo_root)
    sessions: list[SessionInfo] = []
    documents: list[Document] = []
    for path in session_files:
        info, session_documents = read_session(path, topics)
        sessions.append(info)
        documents.extend(session_documents)

    source_documents, references = read_source_documents(repo_root, topics)
    documents.extend(source_documents)
    documents.extend(curated_documents(knowledge_path, knowledge, topics))

    index_path = index_path if index_path.is_absolute() else repo_root / index_path
    map_path = map_path if map_path.is_absolute() else repo_root / map_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(prefix=index_path.name + ".", suffix=".tmp", dir=index_path.parent)
    os.close(descriptor)
    temporary_index = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary_index)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            create_schema(connection)
            generated_at = utc_now()
            metadata = {
                "schema_version": str(SCHEMA_VERSION),
                "generated_at": generated_at,
                "repo_root": str(repo_root),
                "session_root": str(session_root.expanduser()),
                "knowledge_path": str(knowledge_path),
            }
            connection.executemany("INSERT INTO metadata VALUES(?,?)", metadata.items())
            for document in documents:
                insert_document(connection, document)

            topic_counts: Counter[str] = Counter()
            source_topic_counts: Counter[tuple[str, str]] = Counter()
            session_topic_counts: Counter[tuple[str, str]] = Counter()
            for document in documents:
                source_node = f"source:{document.source}"
                add_node(connection, source_node, "source", document.source, {})
                for topic in document.topics:
                    topic_counts[topic] += 1
                    source_topic_counts[(source_node, topic)] += 1
                    if document.session_id:
                        session_topic_counts[(document.session_id, topic)] += 1

            for topic_id, _ in topics:
                add_node(connection, f"topic:{topic_id}", "topic", topic_id, {"document_count": topic_counts[topic_id]})
            for source in references:
                source_node = f"source:{source}"
                add_node(connection, source_node, "source", source, {})
                for reference in sorted(references[source]):
                    reference_node = f"reference:{reference}"
                    add_node(connection, reference_node, "declared_reference", reference, {})
                    add_edge(connection, source_node, reference_node, "DECLARES_REFERENCE")
            for (source_node, topic), count in source_topic_counts.items():
                add_edge(connection, source_node, f"topic:{topic}", "MENTIONS", float(count))

            top_level_sessions = [session for session in sessions if session.kind == "top_level"]
            newest = max(top_level_sessions, key=lambda item: item.created, default=None)
            recommendations: list[dict[str, Any]] = []
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
                recommendations.append(session_metadata)
                add_edge(connection, f"session:{session.id}", f"source:{session.source}", "STORED_IN")
                for (session_id, topic), count in session_topic_counts.items():
                    if session_id == session.id:
                        add_edge(connection, f"session:{session.id}", f"topic:{topic}", "DISCUSSES", float(count))
                if session.parent_session_id:
                    add_edge(connection, f"session:{session.id}", f"session:{session.parent_session_id}", "SUBAGENT_OF")
                elif newest and session.id != newest.id:
                    relation = "NO_MEANINGFUL_RELATION" if recommendation == "DELETE_SAFE_CANDIDATE" else "HISTORICAL_PROJECT_KNOWLEDGE"
                    weight = 0.0 if recommendation == "DELETE_SAFE_CANDIDATE" else 1.0
                    add_edge(connection, f"session:{newest.id}", f"session:{session.id}", relation, weight)

            for index, improvement in enumerate(knowledge.get("improvements", []), 1):
                node_id = f"improvement:{improvement['id']}"
                add_node(connection, node_id, "improvement", improvement["id"], improvement)
                add_edge(connection, "project:current", node_id, "SHOULD_APPLY", 1.0)
            add_node(connection, "project:current", "project", repo_root.name, {"root": str(repo_root)})
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary_index, index_path)
    except BaseException:
        temporary_index.unlink(missing_ok=True)
        raise

    kind_counts = Counter(document.kind for document in documents)
    map_value = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "repo_root": str(repo_root),
        "index": str(index_path),
        "newest_session_id": newest.id if newest else None,
        "inventory": {
            "session_count": len(sessions),
            "top_level_session_count": len(top_level_sessions),
            "subagent_session_count": len(sessions) - len(top_level_sessions),
            "source_file_count": len({document.source for document in source_documents}),
            "document_count": len(documents),
            "kind_counts": dict(sorted(kind_counts.items())),
            "topic_counts": dict(sorted(topic_counts.items())),
        },
        "sessions": sorted(recommendations, key=lambda item: item["created"], reverse=True),
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
    temporary_map = map_path.with_suffix(map_path.suffix + ".tmp")
    temporary_map.write_text(json.dumps(map_value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_map, map_path)
    return map_value


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


def query_database(index_path: Path, query: str, top_k: int = 8, kinds: Sequence[str] = ()) -> list[dict[str, Any]]:
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        where = "documents_fts MATCH ?"
        parameters: list[Any] = [fts_expression(query)]
        if kinds:
            where += " AND d.kind IN (" + ",".join("?" for _ in kinds) + ")"
            parameters.extend(kinds)
        parameters.append(max(top_k * 6, top_k))
        rows = connection.execute(
            f"""
            SELECT d.*, bm25(documents_fts, 0.0, 8.0, 1.0, 3.0) AS rank
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.id
            WHERE {where}
            ORDER BY rank ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    finally:
        connection.close()

    priority = {"verified_fact": 0.35, "improvement": 0.25, "unresolved_item": 0.20}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["topics"] = json.loads(item["topics"])
        item["score"] = round(-float(item.pop("rank")) + priority.get(item["kind"], 0.0), 6)
        candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_source: Counter[str] = Counter()
    for item in candidates:
        if per_source[item["source"]] >= 3:
            continue
        selected.append(item)
        selected_ids.add(item["id"])
        per_source[item["source"]] += 1
        if len(selected) == top_k:
            break
    for item in candidates:
        if len(selected) == top_k:
            break
        if item["id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["id"])
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
        if item.get("timestamp"):
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


def session_recommendations(index_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT id,label,metadata FROM nodes WHERE type='session' ORDER BY json_extract(metadata,'$.created') DESC").fetchall()
    finally:
        connection.close()
    return [dict(row) | {"metadata": json.loads(row["metadata"])} for row in rows]


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


def doctor(repo_root: Path, session_root: Path, index_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(content)")
        fts5 = True
    except sqlite3.OperationalError:
        fts5 = False
    finally:
        connection.close()
    return {
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "fts5": fts5,
        "repo_root": str(repo_root.resolve()),
        "session_root": str(session_root.expanduser()),
        "matching_session_files": len(discover_session_files(session_root.expanduser(), repo_root.resolve())),
        "index_exists": index_path.exists(),
        "lightrag_url": os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621"),
        "lightrag_api_key_configured": bool(os.environ.get("LIGHTRAG_API_KEY")),
    }


def path_for_repo(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root.resolve() / value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path.cwd())
    result.add_argument("--session-root", type=Path, default=Path.home() / ".omp" / "agent" / "sessions")
    result.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    result.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    result.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    subparsers = result.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Rebuild the SQLite index and relationship map atomically")

    query = subparsers.add_parser("query", help="Retrieve local project/session evidence")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=8)
    query.add_argument("--kind", action="append", default=[])
    query.add_argument("--format", choices=("text", "json", "prompt"), default="text")

    graph = subparsers.add_parser("graph", help="Show a node and its graph neighbors")
    graph.add_argument("node_id")

    subparsers.add_parser("sessions", help="List indexed sessions and retention recommendations")
    subparsers.add_parser("doctor", help="Check local prerequisites without network access")

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

    try:
        if args.command == "build":
            print(json.dumps(build_index(repo_root, args.session_root, index_path, map_path, knowledge_path), indent=2))
        elif args.command == "query":
            if args.top_k < 1:
                raise ValueError("--top-k must be positive")
            print(render_results(args.query, query_database(index_path, args.query, args.top_k, args.kind), args.format))
        elif args.command == "graph":
            print(json.dumps(graph_neighbors(index_path, args.node_id), indent=2))
        elif args.command == "sessions":
            print(json.dumps(session_recommendations(index_path), indent=2))
        elif args.command == "doctor":
            print(json.dumps(doctor(repo_root, args.session_root, index_path), indent=2))
        elif args.command == "lightrag-ingest":
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            print(json.dumps(ingest_lightrag(index_path, args.url, args.allow_remote, args.kind, args.batch_size, args.timeout, args.dry_run), indent=2))
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
