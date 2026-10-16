# session_rag

Local, dependency-free (stdlib + SQLite FTS5, Python 3.10) search over this repository and the agent
sessions that worked on it. One index answers "where is X defined", "what did we decide about Y" and
"which session touched Z".

```bash
python3 tools/session_rag/session_rag.py build                       # full rebuild, atomic
python3 tools/session_rag/session_rag.py refresh                     # incremental (fast no-op when nothing changed)
python3 tools/session_rag/session_rag.py query "<symbol or words>" --format prompt --kind source
python3 tools/session_rag/session_rag.py sessions --since 2026-10-01 --limit 20
python3 tools/session_rag/session_rag.py doctor
python3 -m pytest -p no:anyio tools/session_rag -q                   # tests (the -p flag is mandatory here)
```

Run the heavy parts at low priority (`nice -n 19`); cameras and perception run on the same machine.
The index lives in `workspace_scan_artifacts/project_rag.sqlite3` (git-ignored) next to a
`project_rag_map.json` summary.

## Commands

Global options go **before** the command.

| Option | Meaning |
|---|---|
| `--repo-root` | repository to index (default: cwd) |
| `--session-root` | omp session root, repeatable (default `~/.omp/agent/sessions`) |
| `--claude-root` | Claude Code projects root, repeatable (default `~/.claude/projects`) |
| `--worktree-root` | extra directory whose children are worktrees of this repo, repeatable. Defaults: `<repo>/.claude/worktrees` and `~/orca/workspaces/<repo name>` |
| `--no-omp`, `--no-claude` | skip one session family |
| `--index`, `--map`, `--knowledge` | output/input paths |

- `build` rebuilds everything into a temporary database and atomically replaces the index.
- `refresh` re-reads only new, changed and deleted files and deletes their old rows (FTS rows included). It falls
  back to a full build when the index is missing, the schema version differs, the roots changed or the knowledge
  file changed. `--full` forces a rebuild. It prints one summary line (files added/changed/removed, documents,
  seconds); `--json` prints the structured summary, `-q` prints nothing.
- Both take a lock file (`<index>.lock`). `refresh` exits 0 with "refresh skipped" if another run holds it;
  `build` fails. `--wait SECONDS` waits for the lock instead.
- `query "text"` options: `--top-k N`, `--format text|json|prompt`,
  `--kind source|docs|session|curated` (repeatable or comma separated; raw kinds such as `source_code`,
  `project_document`, `session_turn`, `verified_fact` also work), `--since YYYY-MM-DD` (limits **session turns**
  only; source and docs are always current).
- `sessions` lists indexed sessions with retention recommendations (`--since`, `--limit`, `--no-subagents`).
- `graph NODE_ID` shows neighbours of a graph node (`session:<id>`, `source:<path>`, `topic:<id>`).
- `doctor` reports the python/sqlite/FTS5 versions, both session roots with matching-session counts, the index age,
  size, document counts, and how many files were added/changed/removed since the last refresh.
- `lightrag-ingest` / `lightrag-query` talk to an optional LightRAG server (see `lightrag.env.example`).

## What is indexed

- **Source and docs**: the files under the repo with a source suffix (`.cpp .hpp .h .cc .py .md .yaml .xml .urdf
  .xacro .launch .json .txt ...`, plus `CMakeLists.txt`, `package.xml`, `Dockerfile`, `Makefile`).
  - Python (`ast`) and C/C++ (a brace-depth scanner) are split at function, method and class boundaries. The symbol
    is stored in the locator (`symbol:BoxHCutExecutor::planHCut lines:3274-3359`) and in a weighted FTS column that
    also holds the CamelCase words, so `validateFreeSpaceTransitWrist` and `validate free space transit wrist`
    both rank the definition first. Long functions are split into parts that keep the symbol; if the scanner does
    not balance (or Python does not parse) the file falls back to plain line chunks.
  - Size cap 4 MB (`MAX_SOURCE_BYTES`), 1 MB for `.json`. Files over the cap are logged, never skipped silently.
- **Claude Code sessions** (`~/.claude/projects/<dir>/<uuid>.jsonl` and `<uuid>/subagents/*.jsonl`): user and
  assistant **text** only. Skipped: `isMeta` events, `<system-reminder>`/`<command-*>`/`<local-command-*>`
  wrappers, tool results, thinking and tool calls (kept out on purpose to stay small). A session belongs to this repo
  when a `cwd` in its first 300 lines is the repo root, a path below it (including `.claude/worktrees/*`), or a
  path below an Orca worktree root. The worktree name is stored with the session. Title: `custom-title`, else
  `ai-title`, else `summary`, else the first real prompt; subagents use their `.meta.json` description. Subagent
  files are linked to their parent session (`kind=subagent`, graph edge `SUBAGENT_OF`).
- **omp sessions**: as before (`type: "session"` header with `cwd`, `message` events), now also matched below the
  repo root and in worktree roots.
- **Curated facts** from `project_knowledge.json`.

## Exclusions

- Directories (any depth): `.git .idea .vscode .venv .claude .pytest_cache .mypy_cache .ruff_cache __pycache__
  backups build install isaac_exports log Log node_modules workspace_scan_artifacts cad` and the patterns
  `build_* install_* log_* ros-humble-* *.egg-info`. Worktrees are separate checkouts: their source is not indexed
  (their sessions are).
- `scan_results/` keeps summaries and evidence (`.md`, `.json`, logs, scripts) but drops copies of source: `*.cpp.txt`
  style snapshots, code files under `baseline/`, `after/`, `before/`, any `.cpp/.hpp/.h/.cc`, and any code/config
  file whose name matches a current repo file. Evidence `.json` is capped at 64 KB, other evidence at 256 KB.
- The index, its map and the lock are never indexed. `tools/session_rag/test_*` and `project_knowledge.json` are skipped.
- `.gitignore` is not consulted on purpose: `scan_results/` is git-ignored but is wanted evidence.

## Ranking

BM25 over `content` (weight 10), `symbol` (12), `source` (4) and `topics` (3), then re-ranked: session turns count
85 % (subagent turns 78 %), `scan_results/` evidence 80 %, and up to +35 % when query words appear in the file path,
symbol name or session title. Identical session text (forked/resumed sessions) is shown once. Results show the
session date, title and worktree so stale claims are visible; always verify a session claim against current source.

## Auto-refresh (examples only, nothing is installed)

- `systemd/session-rag-refresh.service` and `.timer`: user units, `nice -n 19`, every 2 h plus nightly.
- `claude_session_start_hook.md`: a `SessionStart` hook that runs `refresh` in the background with a timeout.

## Files and schema

`SCHEMA_VERSION` (currently 2) is stored in the index; a mismatch forces a full rebuild, so bump it whenever
the schema, exclusions or chunking change. `file_state` records path, size, mtime and (for source) SHA-256 of every
input; session files are re-read whole when their size or mtime changes (they are append-only).
