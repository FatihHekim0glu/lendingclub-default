# Contributing

Thanks for your interest in `lendingclub-default`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the data + viz + dev extras.
uv venv
uv pip install -e '.[data,viz,dev]'
```

Prefix commands with `uv run` to use the project env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src                                                            # lint
uv run mypy src                                                                  # types (strict)
uv run pytest -q --cov=lendingclub_default --cov-report=term --cov-fail-under=85 # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) must pass. It runs as a blocking step on every PR.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Correctness bars specific to this project

- **No leakage.** Any post-funding column listed in
  `lendingclub_default.LEAKAGE_COLS` must be dropped before features are built. A
  property test asserts none survive the pipeline, do not weaken it.
- **No look-ahead.** The train/test split is temporal by `issue_d` (train ≤
  cutoff, test on later vintages). Never substitute a random K-fold.
- **Honest headline.** Report ROC-AUC / PR-AUC / Brier. Never report accuracy or
  profit/ROI as a headline; the model ranks risk, it does not predict individuals.

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers, no `Co-Authored-By: Claude`,
  no "Generated with Claude", no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
