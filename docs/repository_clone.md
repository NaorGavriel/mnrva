# Repository Cloning & Pruning

Expands on the brief sketch in `docs/code_parser.md` §1.4/§1.5 and supersedes it
on clone depth and where filtering happens. Covers the process that takes a GitHub repository
link and produces `repository_files/` on disk containing only the files
ingestion cares about.

## 1.1 Process overview

```
github_url
   │
   ▼
clone_repository(github_url, dest_dir)     [repository_clone.py]
   │  wipe dest_dir if it exists, shallow `git clone --depth=1`
   ▼
repo_path (= dest_dir, e.g. repository_files/)
   │
   ▼
list_source_files(repo_path)               [repository_parser.py]
   │  git ls-files, filtered by extension allowlist + filename
   │  denylist + size cutoff — pure decision, no disk mutation
   ▼
wanted_files: list[Path]
   │
   ▼
prune_unwanted_files(repo_path, wanted_files)  [repository_clone.py]
   │  deletes every tracked file not in wanted_files; cleans up
   │  now-empty directories
   ▼
repository_files/ now contains only wanted_files on disk
   (still a full git checkout — .git/ untouched)
```

Two modules, two concerns:

- **`repository_parser.py` decides** what's wanted. Pure function of a repo
  checkout on disk → a list of paths. No mutation, easy to unit test.
- **`repository_clone.py` acts** — gets the repo onto disk, and (given the
  parser's decision as input) physically removes what isn't wanted. Never
  contains extension/denylist/size-cutoff logic itself.

`prune_unwanted_files` takes `wanted_files` as a parameter rather than
importing `list_source_files` directly — the two modules stay decoupled, and
the composition happens in the orchestrator (`repository_ingester.py`,
component 1's entry point, not yet built).

## 1.2 `repository_clone.py`

* `clone_repository(github_url: str, dest_dir: Path) -> Path`
  - If `dest_dir` already exists, `shutil.rmtree(dest_dir)` first — always a
    fresh clone, no reuse/pull logic. Simple, no stale-state bugs.
  - Shallow clone: `subprocess.run(["git", "clone", "--depth=1", github_url,
    str(dest_dir)], check=True, capture_output=True, text=True)`. Argument
    list, never a shell string, since `github_url` is user-provided input.
  - Despite the parameter name, nothing here is GitHub-specific — any URL
    `git clone` accepts works (GitLab, Bitbucket, local paths). No GitHub
    REST API calls.
  - On `CalledProcessError`, re-raise as a `CloneError(RuntimeError)`
    carrying the captured stderr.
  - Returns `dest_dir`.

* `get_current_commit_sha(repo_path: Path) -> str`
  - `git rev-parse HEAD`, `cwd=repo_path`, stripped stdout. Captured at
    ingestion time and stored alongside the repo's chunks for the future
    Refresh & Sync component to diff against.

* `prune_unwanted_files(repo_path: Path, wanted_files: list[Path]) -> list[Path]`
  - `tracked = git ls-files` (all tracked paths, relative to `repo_path`).
  - Delete every path in `tracked` not in `set(wanted_files)`
    (`(repo_path / path).unlink()`).
  - Walk `repo_path` bottom-up (skip `.git/`) and `rmdir` any directory left
    empty by the deletions.
  - Returns the list of removed paths — useful for logging and for tests to
    assert on.
  - Never touches `.git/` — pruning only ever acts on `git ls-files` output,
    which never includes `.git` internals.

* `CloneError(RuntimeError)` — raised by `clone_repository` on any git
  failure, message includes the captured stderr.

* `delete_repository(repo_path: Path) -> None`
  - `shutil.rmtree(repo_path)`. Called by the orchestrator
    (`repository_ingester.py`) after chunks are successfully embedded and
    upserted into Qdrant — see §1.6. Deliberately *not* called on failure,
    so a broken run leaves `repository_files/` on disk for inspection
    instead of silently vanishing.

## 1.3 Clone depth: shallow

`--depth=1`, there's no prior state to diff.

## 1.4 Multi-source support (GitHub/GitLab/Bitbucket) or private repos

Later on multi-source support and private repos will be supported,
the solution is a thin wrapper sitting in front of the same `clone_repository`/`fetch`/`diff`/`show`
calls.

## 1.5 `repository_parser.py`

* `list_source_files(repo_path: Path) -> list[Path]` — `git ls-files`,
  filtered through:
  - the extension allowlist (`languages.py`'s `LANGUAGE_CONFIG` keys + `PROSE_EXTENSIONS`),
  - `DENIED_FILENAMES`, a filename denylist for files that would
    otherwise pass the extension allowlist (e.g. `package-lock.json`) -
    `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `poetry.lock`,
    `Pipfile.lock`, `Cargo.lock`, `composer.lock`, `Gemfile.lock`, `go.sum`,
    `mix.lock`, `flake.lock`,
  - a file-size cutoff (`MAX_FILE_SIZE_BYTES`)
* `_is_denied_filename(path: Path) -> bool`
* `_passes_size_cutoff(path: Path, max_bytes: int) -> bool`

This is the single place that decides "wanted or not" — `repository_clone.py` never re-derives this logic, it only executes the deletion.

## 1.6 Destination layout, re-run behavior, and lifecycle

- Single fixed working directory, `repository_files/` — one repo indexed at
  a time.
- Every ingestion run wipes and re-clones fresh (§1.2). Simplicity over incremental reuse.
- **`repository_files/` is scratch space, not state.** It exists only to
  give `list_source_files`/`parse_code_file` something to read from disk
  during one ingestion run. It carries no information Qdrant doesn't also
  end up with, so the orchestrator deletes it (`delete_repository`, §1.2)
  once that run's chunks are successfully upserted. Refresh & Sync is a
  separate process — possibly on a different machine — with no guaranteed
  access to this directory, and maintains its own persistent clone
  independently.

## 1.7 Testing strategy

* `clone_repository` — integration test against a small real local git repo
  (created via `git init` in `tmp_path`, no network dependency), asserting
  the clone lands at `dest_dir` and `.git/` is present. A second test
  asserts an existing `dest_dir` gets wiped before the clone runs. Failure
  path: point at a non-existent local path and assert `CloneError`.
* `get_current_commit_sha` — assert it matches `git rev-parse HEAD` run
  directly against the same fixture repo.
* `prune_unwanted_files` — fixture repo with a mix of wanted and unwanted
  tracked files (including a lockfile and an oversized file); assert only
  `wanted_files` remain on disk, `.git/` is untouched, and now-empty
  directories were removed.
* `delete_repository` — assert the directory is gone afterward; trivial but
  worth a smoke test given it's a destructive filesystem call.
* `list_source_files` / `_is_denied_filename` / `_passes_size_cutoff` — as
  already planned in `docs/code_parser.md` §1.7: predicates get plain unit
  tests, `list_source_files` gets an integration test against a real
  `git init`'d fixture repo, not a mocked `subprocess`.

## 1.8 Open decisions / known limitations

* **Private repos / auth are not handled.** `clone_repository` assumes a
  publicly cloneable URL. No token/SSH-key plumbing yet — add if/when
  needed.
* **`MAX_FILE_SIZE_BYTES` exact value is TBD** — needs a concrete default
  before `repository_parser.py` is implemented.
* **Symlinks and submodules** inside a cloned repo aren't addressed — how
  `list_source_files`/pruning should treat them isn't decided.

## Changes from the original sketch (`docs/code_parser.md` §1.4/§1.5)

- Filtering moved from "list-only, nothing deleted" to "prune in place":
  `repository_clone.py` now physically deletes unwanted files from
  `repository_files/` after cloning, using `repository_parser.py`'s
  `list_source_files` output as the "keep" list. `repository_parser.py`'s
  own responsibilities are otherwise unchanged.
- New: `prune_unwanted_files`, `CloneError`, fixed single-folder
  destination (`repository_files/`), always-wipe-and-reclone re-run policy.
