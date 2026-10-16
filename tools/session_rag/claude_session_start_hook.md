# Claude Code SessionStart hook for the RAG index (example)

Nothing installs this. To use it, merge the block below into the `hooks` section of
`~/.claude/settings.json` (user-wide) or `.claude/settings.local.json` (this checkout only),
adjusting the repository path.

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/igus_rebel_ros2\" && (nohup timeout 900 nice -n 19 python3 tools/session_rag/session_rag.py refresh --quiet >/dev/null 2>&1 &) ; exit 0",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

Why it is shaped this way:

- The refresh runs in a detached subshell (`( ... &)` with `nohup`), so the hook returns immediately and
  session start is never delayed. The hook `timeout` of 5 s only covers launching it.
- `timeout 900` kills a stuck refresh after 15 minutes; `nice -n 19` keeps it out of the way of the cameras.
- `refresh` takes a lock file next to the index, so overlapping sessions (and the systemd timer) never run two
  refreshes at once; the second one prints "refresh skipped" and exits 0.
- It is incremental: unchanged files are only `stat`ed, so a typical run is a few seconds.
