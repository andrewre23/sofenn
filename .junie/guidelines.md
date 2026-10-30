# Junie Agent Development Guidelines

These instructions define operational rules, quality standards, and boundaries for Junie when working within the `sofenn` repository.

---

## 1. Core Operating Principles

- **Follow AGENTS.md & Ticket Scope**: Always consult `AGENTS.md` and adhere strictly to the objectives, boundaries, and acceptance criteria specified in the active ticket.
- **Execute Only Authorized Changes**: Limit changes strictly to what is required to satisfy the assigned task. Never make unauthorized modifications to production code, configuration, or documentation.
- **Inspect Before Editing**: Inspect and understand relevant production code, architectural components, and existing test suites before authoring or modifying any files.
- **Small, Focused Implementations**: Prefer concise, high-utility, and targeted edits. Avoid broad, unsolicited refactorings, style sweeps, or code churn.
- **Preserve Public APIs**: Maintain backward compatibility for public classes, interfaces, and function signatures unless explicitly authorized by the ticket.

---

## 2. Research & Mathematical Rigor

- **Algorithmic Integrity**: Do not silently modify core mathematical logic, membership function formulations, fuzzy inference rules, or self-organizing heuristics.
- **Paper-Based Evidence**: Any mathematical adjustments or bug fixes must be backed by explicit citations from Leng, G., Prasad, G., & McGinnity, T. M. (2004), *An on-line algorithm for creating self-organizing fuzzy neural networks*, Neural Networks, 17, 1477–1493. DOI: 10.1016/j.neunet.2004.07.009.
- **Losses & Metrics**: Do not standardize or alter distinct error metrics or loss functions without research-backed justification.

---

## 3. Testing & Verification Discipline

- **Targeted First, Broad Second**: Run focused unit tests (`tests/unit/`) on modified modules before running broader integration suites (`tests/integration/`).
- **Reproducibility & Determinism**: Ensure all tests use fixed random seeds (`tf.random.set_seed`, `np.random.seed`, `random.seed`) where stochasticity is involved.
- **Honest & Transparent Reporting**: Report test results truthfully. Never weaken assertions, widen tolerances without mathematical basis, disable `@pytest.mark` decorators, or stub implementations to bypass failing tests.

---

## 4. Version Control & Release Boundaries

- **Developer Ownership**: All git commits, pushes, pull requests, merges, version bumps, git tagging, and package releases are exclusively performed by the human developer.
- **Agent Restriction**: Do not run `git commit`, `git push`, `git merge`, `git tag`, open PRs, or modify release versions unless explicitly directed by the ticket instructions.
- **Branching Awareness**:
  - `develop`: Integration branch (target branch for planned automated patch increments).
  - `master`: Production release baseline (target branch for planned automated minor increments).
  - Feature branches (e.g., `feature/SFNN-baseline-modernization`) are used across multiple modernization tickets.

---

## 5. Baseline Modernization Context

Junie should be aware that the repository is undergoing a sequential baseline modernization effort covering:
1. Python runtime and modern Keras compatibility validation.
2. Metadata and dependency consolidation into `pyproject.toml`.
3. Transition from CircleCI (`.circleci/config.yml`) to GitHub Actions workflows.
4. Deterministic testing infrastructure setup.
5. Guarded semantic versioning automation (`develop` -> patch, `master` -> minor, major -> manual, PyPI -> gated; planned capabilities).
6. End-to-end mathematical verification against Leng et al. (2004).

---

## 6. Task Completion Protocol

Every completed task must conclude with a clear, structured summary containing:
- **Files Modified / Created**: Explicit list of files touched.
- **Tests Executed & Outcomes**: Targeted and regression test commands run along with pass/fail counts.
- **Outstanding Findings & Observations**: Any discrepancies between repository state and conventions, potential risks, or notes for subsequent tickets.
