# Repository Cloning & Pruning

```
clone_repository(github_url, dest_dir)      [repository_clone.py]
  → shallow `git clone --depth=1`, wiping dest_dir first
list_unwanted_files(repo_path)              [repository_parser.py]
  → git ls-files minus list_source_files's keep-list
prune_unwanted_files(repo_path, unwanted)   [repository_clone.py]
  → deletes unwanted files + resulting empty dirs, never touches .git/
```

`repository_parser.py` decides (pure, no disk mutation); `repository_clone.py` acts (executes deletion, no allowlist/denylist logic of its own — never calls `git ls-files`). `list_source_files`/`list_unwanted_files` share one predicate (`_is_wanted`) so they can't disagree.

## `repository_clone.py`

* `clone_repository(github_url, dest_dir) -> Path` — wipes `dest_dir` if present, `git clone --depth=1` via `subprocess.run` (arg list, not a shell string — `github_url` is external input). Works for any git host, not just GitHub. Raises `CloneError` (with stderr) on failure.
* `get_current_commit_sha(repo_path) -> str` — `git rev-parse HEAD`. Captured for a future Refresh & Sync component.
* `prune_unwanted_files(repo_path, unwanted_files) -> list[Path]` — deletes each given path if it exists, removes now-empty dirs, skips anything under `.git/` even if given. Returns what it removed.
* `delete_repository(repo_path) -> None` — `shutil.rmtree`. Called after a successful upsert; left alone on failure for inspection.

**Clone depth is shallow** — no history needed, since a future resync process keeps its own persistent clone and `fetch`/`diff`/`show`s incrementally rather than reusing this one.

**Why `subprocess` + `git`, not a provider API**: git's clone/fetch protocol is already host-agnostic (GitHub/GitLab/Bitbucket/self-hosted all speak it identically); a REST API isn't (different schema/auth/rate-limits per host). If multi-source or private-repo support is needed later, the seam is a thin per-host auth/URL adapter in front of these same calls, not a provider class replacing git.

## `repository_parser.py`

* `list_tracked_files(repo_path) -> list[Path]` — raw `git ls-files`.
* `list_source_files(repo_path) -> list[Path]` — kept files: `list_tracked_files` filtered by `_is_wanted`.
* `list_unwanted_files(repo_path) -> list[Path]` — the inverse; feeds `prune_unwanted_files`.
* `_is_wanted(repo_path, path)` — extension allowlist (`languages.LANGUAGE_CONFIG` + `PROSE_EXTENSIONS`) AND NOT `_is_denied_filename` AND `_passes_size_cutoff` (`MAX_FILE_SIZE_BYTES = 1_000_000`, placeholder).
* `DENIED_FILENAMES` — lockfiles that'd otherwise pass the extension check (`package-lock.json`, `pnpm-lock.yaml`, etc.)

## Destination & lifecycle

* Single fixed dir, `repository_files/` — one repo at a time, wiped and re-cloned fresh every run.
* Scratch space, not state: deleted after a successful upsert (carries nothing Qdrant doesn't already have). Refresh & Sync is a separate process (maybe a different machine) with no guaranteed access to it — keeps its own persistent clone.

## Testing strategy

* `clone_repository` — real local fixture repo (`git init` in `tmp_path`): successful clone, wipe-before-reclone, `CloneError` on a bad source.
* `get_current_commit_sha` — matches `git rev-parse HEAD` directly.
* `prune_unwanted_files` — explicit deletion list against a fixture with extra files: only untouched files remain, `.git/` untouched, empty dirs cleaned up.
* `delete_repository` — directory gone afterward.
* `list_source_files`/`list_unwanted_files`/predicates — fixture repo with wanted/lockfile/oversized files; assert the two lists partition tracked files with no overlap/gaps.

## Open decisions

* Private repos / auth — not handled.
* `MAX_FILE_SIZE_BYTES` — placeholder default, not derived from a real constraint.
* Symlinks / submodules — unaddressed.
* Where the ingested commit sha durably lives for a future resync process (no guaranteed filesystem access) — likely Qdrant repo-level metadata, not decided.
