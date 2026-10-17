"""box_seam_detection.live.capture_staging -- RAM-first capture staging.

A capture (rgb .png, pointcloud .pcd ~24 MB, captures manifest .json) is first written to a RAM staging dir
(default ``/dev/shm/box_seam_detection/staging/<session>/{rgb,pointcloud,captures}``), the detection runs from
there, and only afterwards the files are archived to the slow archive drive:

* the capture that produced the final result of a SUCCESSFUL run (``status accepted``) -> the normal layout
  ``<output_root>/{rgb,pointcloud,captures}/<capture_id>.*`` (same names as before, so ``find_matching_pcd`` /
  ``--from-image`` keep working);
* every other capture (rejected attempt, retry, error, cancelled, blocked, run that raised, crash leftover)
  -> the bounded ring ``<output_root>/failed_recent/{rgb,pointcloud,captures}`` (3 most recent by archive time;
  the ring only ever deletes files its own journal ``.ring.json`` lists).

Archived manifests get their image / pointcloud paths rewritten to the archive location and are re-validated with
``orchestrator.load_capture_manifest`` there. The pipeline output dir (``<output_root>/pipeline/<id>/<run>/``) stays
on the archive drive; its JSON records get the exact staged paths text-replaced by the archive paths. The planner
snapshot (``planner_snapshot.json`` of a pipeline dir, the published snapshot and its quarantine file) is NEVER
touched. Snapshot publication always happens before any archiving (the callers archive after ``run_detection``).

This module is pure Python (no ROS). Safety: a session dir is only ever removed/archived by the process that holds
its flock or by a sweeper that obtained that flock (= the owner is dead).
"""

import fcntl
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .snapshot import SIDE_SEED_REJECTED_SUFFIX, write_json_atomically

DEFAULT_STAGING_ROOT = "/dev/shm/box_seam_detection/staging"
STAGING_ENV = "BOX_SEAM_CAPTURE_STAGING"
STAGING_OFF = "off"
MIN_FREE_BYTES = 512 * 1024 * 1024
FAILED_RING_DIR = "failed_recent"
FAILED_RING_KEEP = 3
RING_JOURNAL = ".ring.json"
RING_LOCK = ".ring.lock"
CAPTURE_SUBDIRS = ("rgb", "pointcloud", "captures")
_SUFFIX = {"rgb": ".png", "pointcloud": ".pcd", "captures": ".json"}
_SESSION_RE = re.compile(r"^s\d+_[0-9a-f]{8}$")
_SESSION_LOCK = ".lock"
_NO_LOCK_GRACE_S = 120.0         # a session dir without .lock is only swept when older than this
_PROTECTED_JSON_NAMES = ("planner_snapshot.json",)


def _say(message):
    print(message, flush=True)


def _warn(message):
    print(message, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- shared file helpers
def link_or_copy(source, destination):
    """Hard-link ``source`` at ``destination`` (same bytes, no extra 27 MB); copy if linking fails
    (e.g. across file systems). Atomic: written under a temp name, then os.replace()d."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary_path.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_manifest_atomically(path, manifest):
    """tmp + fsync + os.replace (like the recorder), mode 0o644 (temp files are 0600)."""
    write_json_atomically(path, manifest)
    os.chmod(path, 0o644)


# ----------------------------------------------------------------------------- settings
def resolve_staging_setting(explicit=None, environ=None):
    """The staging root to use, or None for ``off`` (write straight into the archive).

    Precedence: an explicit value (CLI option / node parameter) that is not the built-in default, then the
    ``BOX_SEAM_CAPTURE_STAGING`` environment variable, then ``DEFAULT_STAGING_ROOT``.
    """
    environ = os.environ if environ is None else environ
    value = None if explicit is None else str(explicit).strip()
    if not value or value == DEFAULT_STAGING_ROOT:
        value = str(environ.get(STAGING_ENV, "")).strip() or DEFAULT_STAGING_ROOT
    if value.lower() == STAGING_OFF:
        return None
    return Path(value).expanduser()


# ----------------------------------------------------------------------------- run classification
def classify_run(result):
    """(winner_capture_id | None, discard_ids) of a finished run.

    The winner is the capture of a SUCCESSFUL run (``result.success`` and status ``accepted``: the snapshot was
    published, or the dry-run / no-snapshot-path run accepted it). ``discard_ids`` are burst source frames of an
    averaged attempt (their average replaces them; they are not archived). ``result`` may be None (the run raised).
    """
    if result is None:
        return None, set()
    winner = None
    if getattr(result, "success", False) and getattr(result, "status", "") == "accepted" \
            and getattr(result, "capture_id", None):
        winner = str(result.capture_id)
    attempts = list(getattr(result, "attempts", None) or [])
    attempt_ids = {e.get("capture_id") for e in attempts if isinstance(e, dict)}
    discard = set()
    for entry in attempts:
        averaging = entry.get("averaging") if isinstance(entry, dict) else None
        if isinstance(averaging, dict):
            discard.update(str(s) for s in averaging.get("source_capture_ids") or []
                           if s not in attempt_ids)
    return winner, discard


def run_archive_kwargs(cfg, result):
    """archive_staged() keyword arguments for a finished run: where its pipeline records live and which files must
    never be rewritten (the published planner snapshot and its side-seed quarantine file)."""
    attempts = list(getattr(result, "attempts", None) or [])
    pipeline_dirs = [e.get("pipeline_dir") for e in attempts if isinstance(e, dict)]
    pipeline_dirs.append(getattr(result, "pipeline_dir", None))
    snapshot = getattr(cfg, "snapshot_path", None)
    protected = [snapshot, Path(snapshot).with_name(Path(snapshot).name + SIDE_SEED_REJECTED_SUFFIX)] \
        if snapshot is not None and str(snapshot).strip() not in ("", ".") else []
    protected.append(getattr(result, "snapshot_path", None))
    protected.append(getattr(result, "staged_snapshot_path", None))
    return {"pipeline_roots": [getattr(cfg, "pipeline_root", None)],
            "pipeline_dirs": [d for d in pipeline_dirs if d],
            "extra_files": [getattr(cfg, "camera_geometry_path", None), getattr(cfg, "base_points_path", None)],
            "protected": [p for p in protected if p]}


# ----------------------------------------------------------------------------- report
@dataclass
class ArchiveReport:
    winner: Optional[str] = None
    success_ids: list = field(default_factory=list)
    failed_ids: list = field(default_factory=list)
    discarded_ids: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    elapsed_s: float = 0.0
    archive_root: Optional[Path] = None

    def summary(self):
        parts = []
        if self.success_ids:
            parts.append(f"成功 {', '.join(self.success_ids)} → {self.archive_root}/{{rgb,pointcloud,captures}}")
        if self.failed_ids:
            parts.append(f"失敗 {len(self.failed_ids)} 組 → {self.archive_root}/{FAILED_RING_DIR}"
                         f"（保留最近 {FAILED_RING_KEEP} 組）")
        if self.discarded_ids:
            parts.append(f"丟棄 burst 來源幀 {len(self.discarded_ids)} 張")
        if not parts:
            parts.append("沒有 staging 檔案需要歸檔")
        if self.errors:
            parts.append(f"{len(self.errors)} 個錯誤: {'; '.join(self.errors)}")
        return f"→ capture 歸檔（RAM staging → 磁碟）: {'；'.join(parts)}；用時 {self.elapsed_s:.3f} s"


class ArchiveError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- failed ring
class _FailedRing:
    """``<archive_root>/failed_recent``: the 3 most recent failed captures; only journal-listed files are deleted."""

    def __init__(self, archive_root, keep=FAILED_RING_KEEP):
        self.root = Path(archive_root) / FAILED_RING_DIR
        self.keep = int(keep)
        self._journal = self.root / RING_JOURNAL
        self._lock_fd = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(self.root / RING_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _load(self):
        try:
            with open(self._journal, encoding="utf-8") as handle:
                data = json.load(handle)
            entries = data.get("entries") if isinstance(data, dict) else None
            return [e for e in entries if isinstance(e, dict) and isinstance(e.get("capture_id"), str)
                    and isinstance(e.get("archived_at_ns"), int)]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError):
            # corrupt journal: keep it for diagnosis and start fresh; nothing is deleted on its account
            try:
                os.replace(self._journal, self._journal.with_name(RING_JOURNAL + f".corrupt-{time.time_ns()}"))
            except OSError:
                pass
            return []

    def _save(self, entries):
        write_json_atomically(self._journal, {"schema": "failed_recent_ring/v1", "keep": self.keep,
                                              "entries": entries})

    def register(self, capture_id):
        """Journal the id BEFORE its files are copied (so a crash cannot leave unlisted, never-pruned files)."""
        entries = [e for e in self._load() if e["capture_id"] != capture_id]
        entries.append({"capture_id": capture_id, "archived_at_ns": time.time_ns()})
        self._save(entries)

    def prune(self):
        """Delete the files of the oldest entries beyond ``keep``. Returns the removed ids."""
        entries = sorted(self._load(), key=lambda e: (e["archived_at_ns"], e["capture_id"]))
        stale = entries[:-self.keep] if len(entries) > self.keep else []
        for entry in stale:
            for sub in CAPTURE_SUBDIRS:
                (self.root / sub / f"{entry['capture_id']}{_SUFFIX[sub]}").unlink(missing_ok=True)
        if stale:
            self._save(entries[len(stale):])
        return [e["capture_id"] for e in stale]


# ----------------------------------------------------------------------------- staged capture access
def _staged_paths(session_dir, capture_id):
    return {sub: Path(session_dir) / sub / f"{capture_id}{_SUFFIX[sub]}" for sub in CAPTURE_SUBDIRS}


def staged_capture_ids(session_dir):
    """Capture ids present in a staging session dir (stems of rgb/pointcloud/captures files, no temp files),
    oldest first."""
    found = {}
    for sub in CAPTURE_SUBDIRS:
        directory = Path(session_dir) / sub
        try:
            names = list(directory.iterdir())
        except OSError:
            continue
        for path in names:
            if path.name.startswith(".") or ".tmp-" in path.name or path.suffix != _SUFFIX[sub]:
                continue
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            found[path.stem] = min(found.get(path.stem, mtime), mtime)
    return [cid for cid, _ in sorted(found.items(), key=lambda item: (item[1], item[0]))]


def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ArchiveError(f"cannot read the staged manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArchiveError(f"staged manifest {path} is not a JSON object")
    return value


def _replace_strings(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_strings(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _replace_strings(v, mapping) for k, v in value.items()}
    return value


def _path_forms(path, recorded=None):
    forms = {str(path)}
    try:
        forms.add(str(Path(path).resolve()))
    except OSError:
        pass
    if isinstance(recorded, str) and recorded:
        forms.add(recorded)
    return forms


def _text_variants(text):
    return {text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1]}


def rewrite_json_paths(files, text_map, protected=()):
    """Text-replace the exact staged absolute paths with the archive paths in the given JSON files.

    ``text_map`` maps old text -> new text. ``protected`` paths (the planner snapshot and its quarantine file)
    and any ``planner_snapshot.json`` are never read or written. A file is only rewritten when it contains an old
    text and still parses as JSON afterwards. Returns the number of files rewritten.
    """
    protected_set = set()
    for item in protected:
        if item:
            try:
                protected_set.add(Path(item).resolve())
            except OSError:
                protected_set.add(Path(item))
    ordered = sorted(text_map.items(), key=lambda kv: -len(kv[0]))
    changed = 0
    for path in files:
        path = Path(path)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if (path.name in _PROTECTED_JSON_NAMES or path.name.endswith(SIDE_SEED_REJECTED_SUFFIX)
                or resolved in protected_set or not path.is_file()):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text = text
        for old, new in ordered:
            new_text = new_text.replace(old, new)
        if new_text == text:
            continue
        try:
            json.loads(new_text)
        except ValueError:
            continue
        temporary_path = path.with_name(f".{path.name}.rewrite-{os.getpid()}-{uuid.uuid4().hex[:6]}")
        try:
            mode = path.stat().st_mode & 0o7777
            with open(temporary_path, "w", encoding="utf-8") as handle:
                handle.write(new_text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, path)
            changed += 1
        except OSError:
            temporary_path.unlink(missing_ok=True)
    return changed


def _pipeline_json_files(capture_id, pipeline_roots, pipeline_dirs, extra_files):
    files, seen = [], set()

    def add(path):
        key = str(path)
        if key not in seen:
            seen.add(key)
            files.append(path)

    dirs = [Path(d) for d in pipeline_dirs if d]
    for root in pipeline_roots:
        if root:
            capture_dir = Path(root) / "pipeline" / capture_id
            if capture_dir.is_dir():
                dirs.extend(sorted(p for p in capture_dir.iterdir() if p.is_dir()))
    for directory in dirs:
        if directory.is_dir():
            for path in sorted(directory.rglob("*.json")):
                add(path)
    for extra in extra_files:
        if extra:
            add(Path(extra))
    return files


# ----------------------------------------------------------------------------- archiving
def _archive_one(session_dir, capture_id, dest_root, *, allow_overwrite, pipeline_roots, pipeline_dirs,
                 extra_files, protected):
    """Copy one staged capture to ``dest_root/{rgb,pointcloud,captures}``, rewrite its manifest + pipeline
    records, validate, then remove the staged files. Raises ArchiveError (staged files are kept)."""
    staged = _staged_paths(session_dir, capture_id)
    if not staged["captures"].is_file() or not staged["rgb"].is_file() or not staged["pointcloud"].is_file():
        raise ArchiveError(f"staged capture {capture_id} is incomplete")
    dest = {sub: Path(dest_root) / sub / f"{capture_id}{_SUFFIX[sub]}" for sub in CAPTURE_SUBDIRS}
    if not allow_overwrite:
        clash = [str(p) for p in dest.values() if os.path.lexists(p)]
        if clash:
            raise ArchiveError(f"{capture_id}: archive target already exists ({clash[0]}); not overwriting")
    manifest = _read_manifest(staged["captures"])
    text_map = {}
    for sub in CAPTURE_SUBDIRS:
        new = str(dest[sub].resolve())
        recorded = None
        if sub == "rgb":
            recorded = (manifest.get("image") or {}).get("path")
        elif sub == "pointcloud":
            recorded = (manifest.get("pointcloud") or {}).get("path")
        for old in _path_forms(staged[sub], recorded):
            text_map[old] = new
    created = []
    try:
        for sub in ("rgb", "pointcloud"):
            dest[sub].parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(staged[sub], dest[sub])
            created.append(dest[sub])
        dest["captures"].parent.mkdir(parents=True, exist_ok=True)
        image_new = str(dest["rgb"].resolve())
        pcd_new = str(dest["pointcloud"].resolve())
        manifest = _replace_strings(manifest, text_map)
        if isinstance(manifest.get("image"), dict):
            manifest["image"]["path"] = image_new
        if isinstance(manifest.get("pointcloud"), dict):
            manifest["pointcloud"]["path"] = pcd_new
        write_manifest_atomically(dest["captures"], manifest)
        created.append(dest["captures"])
        from . import orchestrator  # noqa: PLC0415 — lazy: orchestrator imports this module
        orchestrator.load_capture_manifest(dest["captures"], image_new, pcd_new)
    except (OSError, ValueError, ArchiveError) as error:
        for path in created:               # only files this call created
            path.unlink(missing_ok=True)
        raise ArchiveError(f"{capture_id}: {error}") from error
    # pipeline records mention the staged paths -> archive paths (never the planner snapshot)
    try:
        rewrite_json_paths(_pipeline_json_files(capture_id, pipeline_roots, pipeline_dirs, extra_files),
                           _build_text_map(text_map), protected)
    except Exception as error:  # noqa: BLE001 — the records are diagnostics; the capture itself is archived
        _warn(f"pipeline 記錄路徑改寫失敗（{capture_id}）: {error}")
    for path in staged.values():
        path.unlink(missing_ok=True)


def _build_text_map(path_map):
    """old/new path text -> the same pair in every JSON escaping variant (plain, ascii-escaped, unicode)."""
    result = {}
    for old, new in path_map.items():
        for old_variant, new_variant in zip(
                (old, json.dumps(old)[1:-1], json.dumps(old, ensure_ascii=False)[1:-1]),
                (new, json.dumps(new)[1:-1], json.dumps(new, ensure_ascii=False)[1:-1])):
            result[old_variant] = new_variant
    return result


def _remove_staged(session_dir, capture_id):
    for path in _staged_paths(session_dir, capture_id).values():
        path.unlink(missing_ok=True)


def _derived_discards(session_dir, ids):
    """Burst source frames of an averaged capture that is itself staged."""
    discards = set()
    for capture_id in ids:
        path = _staged_paths(session_dir, capture_id)["captures"]
        try:
            averaging = _read_manifest(path).get("averaging")
        except ArchiveError:
            continue
        if isinstance(averaging, dict):
            discards.update(str(s) for s in averaging.get("source_capture_ids") or []
                            if s != capture_id)
    return discards


def archive_staged(session_dir, archive_root, winner=None, ids=None, discard_ids=(), *, pipeline_roots=(),
                   pipeline_dirs=(), extra_files=(), protected=(), log=_say, warn=_warn):
    """Archive the staged captures of ``session_dir`` (``ids``: those; default all).

    ``winner`` -> the normal archive layout; every other id -> the failed_recent ring; ``discard_ids`` (and burst
    sources of a staged averaged capture) are just removed from RAM. Never raises for a single capture's failure
    (collected in ``report.errors``); a winner that cannot be archived keeps its staged files.
    """
    t0 = time.monotonic()
    archive_root = Path(archive_root)
    report = ArchiveReport(winner=winner, archive_root=archive_root)
    ids = list(staged_capture_ids(session_dir) if ids is None else dict.fromkeys(ids))
    discards = set(discard_ids) | _derived_discards(session_dir, ids)
    pipeline_roots = list(dict.fromkeys(str(r) for r in [archive_root, *pipeline_roots] if r))
    kwargs = dict(pipeline_roots=pipeline_roots, pipeline_dirs=pipeline_dirs, extra_files=extra_files,
                  protected=protected)
    for capture_id in [i for i in ids if i in discards and i != winner]:
        _remove_staged(session_dir, capture_id)
        report.discarded_ids.append(capture_id)
    todo = [i for i in ids if i not in discards or i == winner]
    if winner is not None and winner in todo:
        try:
            _archive_one(session_dir, winner, archive_root, allow_overwrite=False, **kwargs)
            report.success_ids.append(winner)
        except ArchiveError as error:
            report.errors.append(str(error))
            warn(f"capture 歸檔失敗（成功的 capture {winner}，staged 檔案保留）: {error}")
        todo.remove(winner)
    ring_ids = list(todo)
    if ring_ids:
        try:
            with _FailedRing(archive_root) as ring:
                for capture_id in ring_ids:
                    try:
                        ring.register(capture_id)
                        _archive_one(session_dir, capture_id, ring.root, allow_overwrite=True, **kwargs)
                        report.failed_ids.append(capture_id)
                    except (ArchiveError, OSError) as error:
                        report.errors.append(f"{capture_id}: {error}")
                        warn(f"失敗 capture {capture_id} 歸檔失敗（從 RAM 丟棄）: {error}")
                        _remove_staged(session_dir, capture_id)
                ring.prune()
        except OSError as error:
            report.errors.append(f"failed_recent: {error}")
            warn(f"failed_recent 無法使用（{error}）；失敗的 staged capture 從 RAM 丟棄")
            for capture_id in ring_ids:
                _remove_staged(session_dir, capture_id)
    report.elapsed_s = time.monotonic() - t0
    return report


# ----------------------------------------------------------------------------- session
class CaptureStaging:
    """One process' staging session: ``root`` is the dir to hand to the recorder / SceneCapture as output root."""

    def __init__(self, session_dir, lock_fd, archive_root, log=_say, warn=_warn):
        self.root = Path(session_dir)
        self.dirs = {sub: self.root / sub for sub in CAPTURE_SUBDIRS}
        self.archive_root = Path(archive_root)
        self._lock_fd = lock_fd
        self._log, self._warn = log, warn
        self._queue = queue.Queue()
        self._thread = None
        self._cond = threading.Condition()
        self._pending = 0
        self._closed = False

    # ---- archiving
    def archive_run(self, result, ids=None, **kwargs):
        """Synchronously archive after a run (``result`` None = the run raised: everything is a failure)."""
        winner, discard = classify_run(result)
        kwargs.setdefault("log", self._log)
        kwargs.setdefault("warn", self._warn)
        return archive_staged(self.root, self.archive_root, winner, ids, discard, **kwargs)

    def submit_archive(self, result, ids, **kwargs):
        """Queue an archive job for the single background archiver thread (jobs run serially, in order)."""
        with self._cond:
            if self._closed:
                raise RuntimeError("capture staging is closed")
            self._pending += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name="capture-archiver", daemon=True)
                self._thread.start()
        self._queue.put((result, list(ids), kwargs))

    def _worker(self):
        while True:
            job = self._queue.get()
            if job is None:
                return
            result, ids, kwargs = job
            try:
                report = self.archive_run(result, ids, **kwargs)
                self._log(report.summary())
            except Exception as error:  # noqa: BLE001 — never kill the archiver thread
                self._warn(f"capture 歸檔執行緒錯誤: {type(error).__name__}: {error}")
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()

    def wait_idle(self, timeout_s):
        """Wait until all queued archive jobs are done; False on timeout."""
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        with self._cond:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def close(self, timeout_s=30.0):
        """Wait (bounded) for pending archiving, then drop the (empty) session dir and release the lock.
        A session dir that still holds files stays for the next start-up sweep."""
        if self._closed:
            return True
        idle = self.wait_idle(timeout_s)
        with self._cond:
            self._closed = True
        if idle and self._thread is not None:
            self._queue.put(None)
        if not idle:
            self._warn(f"capture 歸檔在 {timeout_s:g} s 內未完成；staging 檔案留給下次啟動的清掃")
            return False
        leftovers = staged_capture_ids(self.root)
        if not leftovers:
            shutil.rmtree(self.root, ignore_errors=True)
        else:
            self._warn(f"staging 尚有未歸檔 capture {leftovers}，保留 {self.root}（下次啟動歸檔到 {FAILED_RING_DIR}）")
        self._release()
        return True

    def _release(self):
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)      # drops the flock
            finally:
                self._lock_fd = None


def _is_session_alive(session_dir):
    """True when another live process holds the session's flock (or the dir is brand new / unreadable)."""
    lock_path = Path(session_dir) / _SESSION_LOCK
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        try:
            return time.time() - Path(session_dir).stat().st_mtime < _NO_LOCK_GRACE_S
        except OSError:
            return True
    except OSError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def sweep_stale(staging_root, archive_root, *, exclude=None, log=_say, warn=_warn):
    """Archive the captures left behind by earlier, no longer alive processes into failed_recent (outcome
    unknown), then remove their session dirs. Returns the list of ArchiveReports."""
    reports = []
    staging_root = Path(staging_root)
    try:
        candidates = sorted(p for p in staging_root.iterdir()
                            if p.is_dir() and _SESSION_RE.match(p.name))
    except OSError:
        return reports
    for session in candidates:
        if exclude is not None and Path(session) == Path(exclude):
            continue
        if _is_session_alive(session):
            continue
        ids = staged_capture_ids(session)
        if ids:
            report = archive_staged(session, archive_root, None, ids, log=log, warn=warn)
            reports.append(report)
            log(f"→ 前次執行遺留的 staging capture（結果未知）{report.summary()[2:]}")
        if not staged_capture_ids(session):
            shutil.rmtree(session, ignore_errors=True)
    return reports


def open_staging(setting, archive_root, *, sweep=True, log=_say, warn=_warn):
    """Create this process' staging session under ``setting`` (a path, ``off``/None, see
    resolve_staging_setting). Returns None (direct-to-archive) when staging is off, or - with a warning - when
    the root cannot be created / is not writable / has < 512 MB free."""
    root = setting if isinstance(setting, Path) else resolve_staging_setting(setting)
    if root is None:
        return None
    archive_root = Path(archive_root)
    session = None
    lock_fd = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK | os.X_OK):
            raise OSError(f"{root} is not writable")
        free = shutil.disk_usage(root).free
        if free < MIN_FREE_BYTES:
            raise OSError(f"only {free // (1024 * 1024)} MB free in {root} (< {MIN_FREE_BYTES // (1024 * 1024)} MB)")
        session = root / f"s{os.getpid()}_{uuid.uuid4().hex[:8]}"
        session.mkdir()
        lock_fd = os.open(session / _SESSION_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sub in CAPTURE_SUBDIRS:
            (session / sub).mkdir()
    except OSError as error:
        warn(f"capture staging 無法使用（{error}）；改為直接寫入 {archive_root}")
        if lock_fd is not None:
            os.close(lock_fd)
        if session is not None:
            shutil.rmtree(session, ignore_errors=True)
        return None
    staging = CaptureStaging(session, lock_fd, archive_root, log=log, warn=warn)
    if sweep:
        try:
            sweep_stale(root, archive_root, exclude=session, log=log, warn=warn)
        except Exception as error:  # noqa: BLE001 — a failed sweep must not block the capture
            warn(f"前次遺留 staging 的清掃失敗: {type(error).__name__}: {error}")
    return staging
