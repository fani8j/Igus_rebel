import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from session_rag import build_index, graph_neighbors, query_database, session_recommendations


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


if __name__ == "__main__":
    unittest.main()
