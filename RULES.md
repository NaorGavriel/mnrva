# RULES.md

Guardrails for AI coding agents (Claude Code or otherwise) working in this repository.

## Execution & environment

- **Never execute this project's own code** — no running scripts, no
  `python -c`, no invoking the test suite, no starting the app. Read and edit source; don't run it.
- **Never install, upgrade, or remove a dependency** (`pip install`,
  editing `pyproject.toml`'s `dependencies`).
- **Never start or stop a local service**.

## Network & external systems

- **Never pull data from the web into the sandbox** (`git clone`,
  `curl`, `wget` against external hosts) without asking first — even for
  diagnostics or research.
- **Never make a live call to a paid or external API/service** (OpenAI,
  a real Qdrant/Postgres instance), that includes
  smoke-testing (e.g. `tester.py`'s live tests) or "just checking"
  something works. These cost money and touch real infrastructure.
- **Never push to a git remote, change a remote's URL, or force-push**.

## Secrets & credentials

- **Never edit `.env`.** If a new environment variable is needed, tell
  the user its name and let them add it.
- **Never print, echo, or quote the *values* inside `.env`** or any
  other credential file, even if asked to "check" it. Reporting which
  variable names exist is fine; their values are not for chat output.

## Destructive/irreversible actions

- **Never delete a cloned repository directory or run `git clean`/
  `git reset --hard`/`rm -rf`**.
- **Never silently rewrite a document that represents accumulated,
  user-approved design decisions** (`docs/*.md`, `CLAUDE.md`, this file).

## Design & scope discipline

- **When a design decision is genuinely open** Don't present a proposal as a decision.
- **Don't refactor, rename, or restructure beyond what was asked**, even if a "better" version seems obvious mid-task. Flag it and ask instead of doing it unprompted.
