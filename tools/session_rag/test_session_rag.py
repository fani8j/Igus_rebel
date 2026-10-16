import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import session_rag
from session_rag import (
    IndexLocked,
    build_index,
    cpp_spans,
    doctor,
    graph_neighbors,
    index_lock,
    main,
    pending_changes,
    python_spans,
    query_database,
    refresh_index,
    session_recommendations,
    source_chunks,
    source_skip_reason,
)


class SessionRagTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        self.sessions = Path(self.temporary.name) / "sessions"
        (self.root / "src").mkdir(parents=True)
        self.sessions.mkdir()
        (self.root / "src" / "parser.cpp").write_text(
            "uint64_t parse_dout(const std::string &token) { return std::stoull(token, nullptr, 16); }\n",
            encoding="utf-8",
        )
        tool_dir = self.root / "tools" / "session_rag"
        tool_dir.mkdir(parents=True)
        (tool_dir / "test_noise.py").write_text(
            "CRI DOUT uint64 hexadecimal parser crash noise\n", encoding="utf-8"
        )
        self.knowledge = self.root / "knowledge.json"
        self.knowledge.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "topics": [{"id": "cri_network", "patterns": ["dout", "stoull", "cri"]}],
                    "verified_facts": ["The CRI DOUT mask is parsed as uint64 hexadecimal."],
                    "unresolved_items": [],
                    "improvements": [
                        {
                            "id": "reproduce_then_fix",
                            "problem": "The failure was not isolated.",
                            "action": "Reproduce the exact DOUT parser failure before editing.",
                            "benefit": "The same reproduction proves the fix.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self._write_session(
            "current.jsonl",
            "current-session",
            "2026-09-13T15:00:00Z",
            [("user", "How should we index debugging history?")],
        )
        self._write_session(
            "history.jsonl",
            "history-session",
            "2026-09-12T15:00:00Z",
            [
                ("user", "The DOUT parser crashes on 8000000000000000."),
                ("assistant", "Use full-width uint64 hexadecimal parsing and validate the token."),
            ],
        )
        self._write_session(
            "empty.jsonl",
            "empty-session",
            "2026-09-11T15:00:00Z",
            [("user", "updaste")],
        )
        subagent_dir = self.sessions / "history"
        subagent_dir.mkdir()
        (subagent_dir / "parser-scout.jsonl").write_text(
            json.dumps(
                {
                    "type": "session",
                    "id": "parser-scout",
                    "timestamp": "2026-09-12T16:00:00Z",
                    "cwd": str(self.root),
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "message",
                    "timestamp": "2026-09-12T16:01:00Z",
                    "message": {"role": "user", "content": [{"type": "text", "text": "Inspect the CRI parser."}]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        other = self.sessions / "other.jsonl"
        other.write_text(
            json.dumps(
                {
                    "type": "session",
                    "id": "other-repo",
                    "timestamp": "2026-09-10T15:00:00Z",
                    "cwd": str(self.root.parent / "other"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.index = self.root / "rag.sqlite3"
        self.map = self.root / "rag-map.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_session(self, name, session_id, timestamp, messages):
        path = self.sessions / name
        events = [
            {
                "type": "session",
                "id": session_id,
                "timestamp": timestamp,
                "cwd": str(self.root),
                "title": session_id,
            }
        ]
        for index, (role, text) in enumerate(messages, 1):
            events.append(
                {
                    "type": "message",
                    "timestamp": f"{timestamp[:-1]}.{index:03d}Z",
                    "message": {"role": role, "content": [{"type": "text", "text": text}]},
                }
            )
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    def _build(self):
        return build_index(self.root, self.sessions, self.index, self.map, self.knowledge)

    def test_build_filters_other_repositories_and_records_retention(self):
        result = self._build()
        self.assertEqual(4, result["inventory"]["session_count"])
        self.assertEqual(3, result["inventory"]["top_level_session_count"])
        self.assertEqual(1, result["inventory"]["subagent_session_count"])
        recommendations = {
            item["metadata"]["id"]: item["metadata"]["recommendation"]
            for item in session_recommendations(self.index)
        }
        self.assertEqual("KEEP_NEWEST", recommendations["current-session"])
        self.assertEqual("KEEP_ARCHIVE_AND_INDEX", recommendations["history-session"])
        self.assertEqual("KEEP_WITH_PARENT", recommendations["parser-scout"])
        self.assertEqual("DELETE_SAFE_CANDIDATE", recommendations["empty-session"])
        self.assertNotIn("other-repo", recommendations)

    def test_query_returns_verified_fact_and_current_source(self):
        self._build()
        results = query_database(self.index, "CRI DOUT uint64 hexadecimal parser", top_k=6)
        contents = "\n".join(item["content"] for item in results)
        sources = {item["source"] for item in results}
        self.assertIn("parsed as uint64 hexadecimal", contents)
        self.assertIn("src/parser.cpp", sources)
        self.assertNotIn("tools/session_rag/test_noise.py", sources)
        self.assertTrue(all("toolResult" not in item["content"] for item in results))

    def test_graph_links_historical_session_to_topic(self):
        self._build()
        graph = graph_neighbors(self.index, "session:history-session")
        edge_types = {edge["type"] for edge in graph["edges"]}
        neighbors = {edge["neighbor_label"] for edge in graph["edges"]}
        self.assertIn("DISCUSSES", edge_types)
        self.assertIn("cri_network", neighbors)


CPP_SAMPLE = """#include <string>
namespace demo
{
// Parses a mask.
static uint64_t parseMask(const std::string & token)
{
  if (token.empty()) {
    return 0;
  }
  return std::stoull(token, nullptr, 16);
}

class Planner
{
public:
  Planner(int a) : x_{a}, y_(a)
  {
    init();
  }
  int inlineGetter() const {return x_;}
  void plan();

private:
  int x_;
  int y_ = 0;
};

void Planner::plan()
{
  const char * s = "brace } in string { ";  // comment with {
  auto f = [&](int v) {
    return v + 1;
  };
  (void)f;
}

}  // namespace demo

TEST(Suite, Name)
{
  EXPECT_TRUE(true);
}
"""

PY_SAMPLE = """import os


# helper comment
def helper(a):
    return a + 1


class Thing:
    \"\"\"Doc.\"\"\"

    @staticmethod
    def make():
        return Thing()

    def method(self):
        return 2
"""


def claude_event(event_type, session_id, cwd, content, timestamp, **extra):
    event = {
        "type": event_type,
        "sessionId": session_id,
        "cwd": cwd,
        "timestamp": timestamp,
        "message": {"role": event_type, "content": content},
    }
    event.update(extra)
    return event


class ClaudeSessionRagTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.root = base / "repo"
        self.orca = base / "orca" / "repo"
        self.claude = base / "claude_projects"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "main.py").write_text("def unique_alpha_function():\n    return 1\n", encoding="utf-8")
        self.knowledge = self.root.parent / "knowledge.json"
        self.knowledge.write_text(
            json.dumps({"schema_version": 1, "topics": [], "verified_facts": [], "unresolved_items": [], "improvements": []}),
            encoding="utf-8",
        )
        self.index = self.root / "rag.sqlite3"
        self.map = self.root / "rag-map.json"
        project = self.claude / "-tmp-repo"
        project.mkdir(parents=True)
        self.main_session = project / "aaaa1111.jsonl"
        self._write(
            self.main_session,
            [
                {"type": "queue-operation", "operation": "enqueue", "sessionId": "aaaa1111"},
                {"type": "attachment", "sessionId": "aaaa1111", "cwd": str(self.root), "timestamp": "2026-10-01T10:00:00Z"},
                claude_event("user", "aaaa1111", str(self.root), "How does zebrafish calibration work?", "2026-10-01T10:00:01Z"),
                claude_event(
                    "user", "aaaa1111", str(self.root),
                    [{"type": "text", "text": "<system-reminder>reminder wrapper plutonium</system-reminder>"}],
                    "2026-10-01T10:00:02Z",
                ),
                claude_event(
                    "user", "aaaa1111", str(self.root),
                    "<command-name>/clear</command-name><command-message>clear</command-message>", "2026-10-01T10:00:03Z",
                ),
                claude_event("user", "aaaa1111", str(self.root), "meta caveat uranium", "2026-10-01T10:00:04Z", isMeta=True),
                claude_event(
                    "assistant", "aaaa1111", str(self.root),
                    [{"type": "thinking", "thinking": "secret neptunium reasoning", "signature": "x"}],
                    "2026-10-01T10:00:05Z",
                ),
                claude_event(
                    "assistant", "aaaa1111", str(self.root),
                    [{"type": "text", "text": "The zebrafish calibration uses a pivot method."}],
                    "2026-10-01T10:00:06Z",
                ),
                claude_event(
                    "assistant", "aaaa1111", str(self.root),
                    [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "echo osmium"}}],
                    "2026-10-01T10:00:07Z",
                ),
                claude_event(
                    "user", "aaaa1111", str(self.root),
                    [{"type": "tool_result", "tool_use_id": "t1", "content": "tool output iridium"}],
                    "2026-10-01T10:00:08Z",
                ),
                {"type": "ai-title", "aiTitle": "Zebrafish calibration chat", "sessionId": "aaaa1111"},
            ],
        )
        subagents = project / "aaaa1111" / "subagents"
        subagents.mkdir(parents=True)
        self.subagent = subagents / "agent-s1.jsonl"
        self._write(
            self.subagent,
            [
                claude_event(
                    "user", "aaaa1111", str(self.root), "Inspect the platypus module.", "2026-10-01T11:00:00Z", isSidechain=True
                ),
                claude_event(
                    "assistant", "aaaa1111", str(self.root),
                    [{"type": "text", "text": "The platypus module is fine."}], "2026-10-01T11:00:01Z", isSidechain=True,
                ),
            ],
        )
        (subagents / "agent-s1.meta.json").write_text(
            json.dumps({"agentType": "Explore", "description": "Map the platypus"}), encoding="utf-8"
        )
        orca_cwd = self.orca / "box-3d" / "src"
        self._write(
            self.claude / "-orca-box" / "bbbb2222.jsonl",
            [claude_event("user", "bbbb2222", str(orca_cwd), "Orca worktree question about narwhal.", "2026-10-02T10:00:00Z")],
            compact=False,
        )
        wt_cwd = self.root / ".claude" / "worktrees" / "feature-x"
        self._write(
            self.claude / "-repo-wt" / "cccc3333.jsonl",
            [claude_event("user", "cccc3333", str(wt_cwd), "Claude worktree question about axolotl.", "2026-10-03T10:00:00Z")],
        )
        self._write(
            self.claude / "-other" / "dddd4444.jsonl",
            [claude_event("user", "dddd4444", str(base / "other_repo"), "Unrelated question about capybara.", "2026-10-04T10:00:00Z")],
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write(path, events, compact=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        separators = (",", ":") if compact else None  # real Claude Code files are compact; stay tolerant of spaces
        path.write_text("".join(json.dumps(event, separators=separators) + "\n" for event in events), encoding="utf-8")

    def _build(self):
        return build_index(
            self.root, None, self.index, self.map, self.knowledge,
            claude_roots=[self.claude], worktree_roots=[self.orca, self.root / ".claude" / "worktrees"],
        )

    def _refresh(self, **extra):
        return refresh_index(
            self.root, None, self.index, self.map, self.knowledge,
            claude_roots=[self.claude], worktree_roots=[self.orca, self.root / ".claude" / "worktrees"], **extra,
        )

    def _sessions(self):
        connection = sqlite3.connect(self.index)
        connection.row_factory = sqlite3.Row
        try:
            return {row["id"]: dict(row) for row in connection.execute("SELECT * FROM sessions")}
        finally:
            connection.close()

    def _fts_consistent(self):
        connection = sqlite3.connect(self.index)
        try:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            fts = connection.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
            orphans = connection.execute(
                "SELECT COUNT(*) FROM documents_fts WHERE rowid NOT IN (SELECT rowid FROM documents)"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(documents, fts)
        self.assertEqual(0, orphans)

    def test_claude_sessions_match_by_cwd_and_worktree(self):
        result = self._build()
        sessions = self._sessions()
        self.assertEqual({"aaaa1111", "agent-s1", "bbbb2222", "cccc3333"}, set(sessions))
        self.assertEqual(3, result["inventory"]["top_level_session_count"])
        self.assertEqual(1, result["inventory"]["subagent_session_count"])
        self.assertEqual("box-3d", sessions["bbbb2222"]["worktree"])
        self.assertEqual("feature-x", sessions["cccc3333"]["worktree"])
        self.assertIsNone(sessions["aaaa1111"]["worktree"])
        self.assertEqual("claude", sessions["aaaa1111"]["format"])

    def test_claude_text_only_and_wrappers_skipped(self):
        self._build()
        text = "\n".join(item["content"] for item in query_database(self.index, "zebrafish calibration pivot", top_k=10))
        self.assertIn("pivot method", text)
        for forbidden in ("plutonium", "uranium", "neptunium", "osmium", "iridium", "clear"):
            hits = query_database(self.index, forbidden, top_k=5, kinds=["session"])
            self.assertEqual([], hits, forbidden)
        sessions = self._sessions()
        self.assertEqual(1, sessions["aaaa1111"]["user_messages"])
        self.assertEqual(1, sessions["aaaa1111"]["assistant_messages"])
        self.assertEqual("Zebrafish calibration chat", sessions["aaaa1111"]["title"])

    def test_subagent_links_to_parent_and_title_from_meta(self):
        self._build()
        sessions = self._sessions()
        self.assertEqual("subagent", sessions["agent-s1"]["kind"])
        self.assertEqual("aaaa1111", sessions["agent-s1"]["parent_session_id"])
        self.assertEqual("Explore: Map the platypus", sessions["agent-s1"]["title"])
        graph = graph_neighbors(self.index, "session:agent-s1")
        self.assertIn("SUBAGENT_OF", {edge["type"] for edge in graph["edges"]})
        recommendations = {item["metadata"]["id"]: item["metadata"]["recommendation"] for item in session_recommendations(self.index)}
        self.assertEqual("KEEP_WITH_PARENT", recommendations["agent-s1"])

    def test_first_prompt_title_when_no_title_event(self):
        self._build()
        self.assertTrue(self._sessions()["bbbb2222"]["title"].startswith("Orca worktree question"))

    def test_other_repo_session_not_indexed(self):
        self._build()
        self.assertEqual([], query_database(self.index, "capybara", top_k=5))

    def test_refresh_is_incremental(self):
        self._build()
        noop = self._refresh()["last_run"]
        self.assertEqual("refresh", noop["mode"])
        self.assertEqual((0, 0, 0, 0, 0), (
            noop["files_added"], noop["files_changed"], noop["files_removed"], noop["documents_added"], noop["documents_removed"]
        ))
        # add a source file, modify a source file, append to a session, delete a session
        (self.root / "src" / "extra.py").write_text("def brand_new_beta_function():\n    return 2\n", encoding="utf-8")
        (self.root / "src" / "main.py").write_text("def unique_gamma_function():\n    return 3\n", encoding="utf-8")
        with self.main_session.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(claude_event("assistant", "aaaa1111", str(self.root), [{"type": "text", "text": "Added a wombat note."}], "2026-10-05T10:00:00Z"))
                + "\n"
            )
        self.subagent.unlink()
        (self.subagent.parent / "agent-s1.meta.json").unlink()
        self._write(
            self.claude / "-repo-wt" / "eeee5555.jsonl",
            [claude_event("user", "eeee5555", str(self.root), "A brand new session about quokka.", "2026-10-06T10:00:00Z")],
        )
        run = self._refresh()["last_run"]
        self.assertEqual("refresh", run["mode"])
        self.assertEqual(2, run["files_added"])
        self.assertEqual(2, run["files_changed"])
        self.assertEqual(1, run["files_removed"])
        self.assertEqual([], query_database(self.index, "unique_alpha_function", top_k=5))
        self.assertEqual("src/main.py", query_database(self.index, "unique_gamma_function", top_k=1)[0]["source"])
        self.assertEqual("src/extra.py", query_database(self.index, "brand_new_beta_function", top_k=1)[0]["source"])
        self.assertTrue(query_database(self.index, "wombat", top_k=3))
        self.assertTrue(query_database(self.index, "quokka", top_k=3))
        self.assertEqual([], query_database(self.index, "platypus", top_k=3))
        self.assertNotIn("agent-s1", self._sessions())
        self.assertIn("eeee5555", self._sessions())
        self._fts_consistent()
        # session graph reflects the refresh
        with self.assertRaises(KeyError):
            graph_neighbors(self.index, "session:agent-s1")
        again = self._refresh()["last_run"]
        self.assertEqual((0, 0, 0), (again["files_added"], again["files_changed"], again["files_removed"]))
        self.assertEqual({"added": 0, "changed": 0, "removed": 0}, pending_changes(
            session_rag.resolve_config(
                self.root, None, self.index, self.map, self.knowledge, [self.claude],
                [self.orca, self.root / ".claude" / "worktrees"],
            )
        ))

    def test_refresh_hash_unchanged_source_is_not_reindexed(self):
        self._build()
        path = self.root / "src" / "main.py"
        content = path.read_bytes()
        later = time.time() + 5
        os.utime(path, (later, later))
        path.write_bytes(content)
        os.utime(path, (later, later))
        run = self._refresh()["last_run"]
        self.assertEqual(0, run["files_changed"])
        self.assertEqual(1, run["files_unchanged_by_hash"])

    def test_schema_mismatch_forces_full_rebuild(self):
        self._build()
        connection = sqlite3.connect(self.index)
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        connection.commit()
        connection.close()
        self.assertEqual("full", self._refresh()["last_run"]["mode"])
        self.assertEqual("refresh", self._refresh()["last_run"]["mode"])
        self.assertEqual("full", self._refresh(full=True)["last_run"]["mode"])

    def test_missing_index_refresh_builds_fully(self):
        self.assertEqual("full", self._refresh()["last_run"]["mode"])
        self._fts_consistent()

    def test_lock_prevents_concurrent_runs(self):
        with index_lock(self.index):
            with self.assertRaises(IndexLocked):
                with index_lock(self.index):
                    pass
            with self.assertRaises(IndexLocked):
                self._refresh()
        with index_lock(self.index):
            pass

    def test_since_and_kind_filters(self):
        self._build()
        self.assertTrue(query_database(self.index, "narwhal", top_k=3, kinds=["session"], since="2026-10-02"))
        self.assertEqual([], query_database(self.index, "narwhal", top_k=3, kinds=["session"], since="2026-10-03"))
        self.assertEqual([], query_database(self.index, "narwhal", top_k=3, kinds=["source"]))
        self.assertTrue(query_database(self.index, "unique_alpha_function", top_k=3, kinds=["source,docs"]))
        with self.assertRaises(ValueError):
            query_database(self.index, "narwhal", since="yesterday")

    def test_query_results_show_session_metadata(self):
        self._build()
        results = query_database(self.index, "narwhal", top_k=3)
        text = session_rag.render_results("narwhal", results, "prompt")
        self.assertIn("2026-10-02", text)
        self.assertIn("worktree=box-3d", text)
        self.assertIn('session="Orca worktree question', text)

    def test_doctor_reports_roots_and_changes(self):
        self._build()
        report = doctor(
            self.root, None, self.index, [self.claude], [self.orca, self.root / ".claude" / "worktrees"],
            self.knowledge, self.map,
        )
        self.assertEqual(3, report["claude_roots"][0]["matching_top_level"])
        self.assertEqual(1, report["claude_roots"][0]["matching_subagent"])
        self.assertEqual({"added": 0, "changed": 0, "removed": 0}, report["index"]["files_changed_since_last_refresh"])
        (self.root / "src" / "later.py").write_text("x = 1\n", encoding="utf-8")
        report = doctor(
            self.root, None, self.index, [self.claude], [self.orca, self.root / ".claude" / "worktrees"],
            self.knowledge, self.map,
        )
        self.assertEqual(1, report["index"]["files_changed_since_last_refresh"]["added"])
        self.assertIsNotNone(report["index"]["age_seconds"])

    def test_cli_refresh_and_query(self):
        arguments = [
            "--repo-root", str(self.root), "--no-omp", "--claude-root", str(self.claude),
            "--worktree-root", str(self.orca), "--index", str(self.index), "--map", str(self.map),
            "--knowledge", str(self.knowledge),
        ]
        self.assertEqual(0, main(arguments + ["refresh", "-q"]))
        self.assertEqual(0, main(arguments + ["query", "narwhal", "--kind", "session", "--since", "2026-10-01"]))
        self.assertEqual(0, main(arguments + ["refresh", "-q"]))


class SourceIndexingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.knowledge = self.root.parent / "knowledge.json"
        self.knowledge.write_text(
            json.dumps({"schema_version": 1, "topics": [], "verified_facts": [], "unresolved_items": [], "improvements": []}),
            encoding="utf-8",
        )
        self.index = self.root / "rag.sqlite3"
        self.map = self.root / "rag-map.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, relative, text="content findme\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _build(self):
        return build_index(self.root, None, self.index, self.map, self.knowledge)

    def _sources(self):
        connection = sqlite3.connect(self.index)
        try:
            return {row[0] for row in connection.execute("SELECT DISTINCT source FROM documents")}
        finally:
            connection.close()

    def test_directory_and_evidence_exclusions(self):
        self._write("src/keep.py", "def keep():\n    pass\n")
        for directory in ("build", "build_yaw", "install_p3", "log", "Log", "log_old", "ros-humble-moveit-servo-2.5.10", ".claude", "cad", ".pytest_cache"):
            self._write(f"{directory}/skip.md")
        self._write("src/keep.cpp", "int main() {return 0;}\n")
        self._write("scan_results/run1/summary.md", "# summary\n")
        self._write("scan_results/run1/result.json", "{\"a\": 1}\n")
        self._write("scan_results/run1/big.json", "x" * (session_rag.MAX_EVIDENCE_JSON_BYTES + 1))
        self._write("scan_results/run1/executor.cpp.txt", "int snapshot_copy;\n")
        self._write("scan_results/run1/baseline/old.hpp", "int baseline_copy;\n")
        self._write("scan_results/run1/baseline/result.json", "{\"b\": 2}\n")
        self._write("scan_results/run1/after/script.py", "x = 1\n")
        self._write("scan_results/run1/logs/keep.py", "def keep():\n    pass\n")
        self._write("scan_results/run1/own_script.py", "def own():\n    pass\n")
        self.assertEqual(
            {
                "src/keep.py",
                "src/keep.cpp",
                "scan_results/run1/summary.md",
                "scan_results/run1/result.json",
                "scan_results/run1/baseline/result.json",
                "scan_results/run1/own_script.py",
            },
            self._build_and_sources(),
        )

    def _build_and_sources(self):
        self._build()
        return self._sources()

    def test_large_source_cap_and_logging(self):
        self.assertGreaterEqual(session_rag.MAX_SOURCE_BYTES, 4_000_000)
        self.assertIsNone(source_skip_reason("src/big.cpp", "big.cpp", 598_000))
        self.assertEqual("too_large", source_skip_reason("src/big.cpp", "big.cpp", session_rag.MAX_SOURCE_BYTES + 1))
        self.assertEqual("too_large", source_skip_reason("scan_results/a/x.json", "x.json", 70_000))
        original = session_rag.MAX_SOURCE_BYTES
        session_rag.MAX_SOURCE_BYTES = 100
        try:
            self._write("src/huge.cpp", "int x;\n" * 100)
            self._write("src/small.cpp", "int y;\n")
            result = self._build()
        finally:
            session_rag.MAX_SOURCE_BYTES = original
        self.assertEqual({"src/small.cpp"}, self._sources())
        self.assertEqual(1, result["last_run"]["skipped_too_large"])

    def test_cpp_symbol_chunking(self):
        spans = {span.name: span for span in cpp_spans(CPP_SAMPLE)}
        self.assertEqual(
            {
                "parseMask",
                "Planner",
                "Planner::Planner",
                "Planner::inlineGetter",
                "Planner::plan",
                "TEST(Suite, Name)",
            },
            set(spans),
        )
        self.assertEqual((4, 11), (spans["parseMask"].start, spans["parseMask"].end))  # leading comment included
        chunks = source_chunks("a.cpp", CPP_SAMPLE)
        symbols = {symbol for symbol, _, _, _ in chunks}
        self.assertIn("parseMask", symbols)
        self.assertIn("Planner::plan", symbols)
        plan = [chunk for symbol, _, _, chunk in chunks if symbol == "Planner::plan"][0]
        self.assertIn("brace } in string", plan)
        self.assertNotIn("inlineGetter", plan)

    def test_cpp_unbalanced_falls_back_to_line_chunks(self):
        self.assertIsNone(cpp_spans("void f() {\n  int x;\n"))
        chunks = source_chunks("broken.cpp", "void f() {\n  int x;\n")
        self.assertEqual([None], [symbol for symbol, _, _, _ in chunks])

    def test_python_symbol_chunking(self):
        spans = {span.name: span for span in python_spans(PY_SAMPLE)}
        self.assertEqual({"helper", "Thing", "Thing.make", "Thing.method"}, set(spans))
        self.assertEqual(4, spans["helper"].start)  # leading comment line included
        self.assertEqual(12, spans["Thing.make"].start)  # decorator included
        self.assertIsNone(python_spans("def broken(:\n"))
        self.assertEqual([None], [symbol for symbol, _, _, _ in source_chunks("broken.py", "def broken(:\n    pass\n")])

    def test_long_function_splits_and_keeps_symbol(self):
        body = "".join(f"    value_{index} = {index}\n" for index in range(600))
        text = "def huge_function():\n" + body + "\n\ndef small_one():\n    return 1\n"
        chunks = [item for item in source_chunks("big.py", text) if item[0] == "huge_function"]
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= session_rag.MAX_CHUNK_CHARS for _, _, _, chunk in chunks))

    def test_symbol_query_ranks_definition_first(self):
        self._write(
            "src/exec.cpp",
            "// filler about the planner\nvoid helperOne() { callThing(); }\n\n"
            "int Executor::validateSpecialWrist(int a)\n{\n  return a;\n}\n\n"
            "void other()\n{\n  // calls validateSpecialWrist indirectly in a comment\n  helperOne();\n}\n",
        )
        self._write("notes.md", "We discuss validateSpecialWrist here once.\n")
        self._build()
        top = query_database(self.index, "validateSpecialWrist", top_k=3)[0]
        self.assertEqual("src/exec.cpp", top["source"])
        self.assertEqual("Executor::validateSpecialWrist", top["symbol"])
        self.assertIn("symbol:Executor::validateSpecialWrist lines:", top["locator"])
        split = query_database(self.index, "validate special wrist", top_k=1)[0]
        self.assertEqual("Executor::validateSpecialWrist", split["symbol"])

    def test_source_ranks_above_session_turns_at_equal_relevance(self):
        self._write("docs/note.md", "The quasar calibration sequence is documented here.\n")
        sessions = self.root / "sessions"
        sessions.mkdir()
        events = [
            {"type": "session", "id": "s1", "timestamp": "2026-09-01T10:00:00Z", "cwd": str(self.root), "title": "old chat"},
            {
                "type": "message",
                "timestamp": "2026-09-01T10:00:01Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "The quasar calibration sequence is documented here."}]},
            },
        ]
        (sessions / "s1.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        build_index(self.root, sessions, self.index, self.map, self.knowledge)
        results = query_database(self.index, "quasar calibration sequence", top_k=2)
        self.assertEqual(["project_document", "session_turn"], [item["kind"] for item in results])
        self.assertEqual("old chat", results[1]["session_title"])


if __name__ == "__main__":
    unittest.main()
