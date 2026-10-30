# Continuous Integration (CI) Architecture & Validation (SFNN-12, SFNN-35)

## 1. Overview

This document describes the GitHub Actions continuous integration architecture for `sofenn`, established under **SFNN-12** (replacing legacy CircleCI) and optimized under **SFNN-35** (fast development feedback).

The CI system balances rapid iteration speed during local and feature-branch development against rigorous multi-platform compatibility, packaging integrity, and coverage quality gates prior to integration.

---

## 2. Workflow Architecture

The CI pipeline is structured into a modular, reusable validation workflow and an entry-point trigger workflow:

```text
.github/workflows/
├── ci.yml                 # Entry-point workflow for pushes and pull requests
├── python-validation.yml  # Reusable workflow (workflow_call) implementing validation jobs
└── dependency-review.yml  # Pull request dependency vulnerability scanner
```

### 2.1 Reusable Workflow (`.github/workflows/python-validation.yml`)
The reusable workflow is invoked via `workflow_call` and accepts an optional parameter:
- **`fast-feedback`** (`boolean`, default: `false`): When `true`, restricts execution to fast unit tests on the primary development platform (`ubuntu-latest`, Python 3.11) and skips cross-platform matrix runs, integration tests, coverage collection, and package building.

The workflow defines five parallelizable jobs:

1. **`fast-unit-tests`**:
   - Executes the targeted unit test suite (`pytest tests/unit -v`).
   - Runs exclusively in fast feedback mode (`fast-feedback: true`) on the primary development platform (`ubuntu-latest`, Python 3.11).
   - Utilizes pip dependency caching keyed on `pyproject.toml`.
2. **`unit-tests`**:
   - Executes the targeted unit test suite (`pytest tests/unit -v`).
   - In full validation (`fast-feedback: false`), runs across the multi-platform Python matrix (Ubuntu, Windows, macOS across Python 3.9, 3.11, 3.12).
   - Skipped during fast feedback (`if: ${{ !inputs.fast-feedback }}`).
   - Utilizes pip dependency caching keyed on `pyproject.toml`.
3. **`integration-tests`**:
   - Executes the end-to-end integration test suite (`pytest tests/integration -v`).
   - Runs across the multi-platform Python matrix in parallel with unit tests.
   - Skipped during fast feedback (`if: ${{ !inputs.fast-feedback }}`).
   - Utilizes pip dependency caching keyed on `pyproject.toml`.
4. **`coverage`**:
   - Executes the full test suite with branch measurement enabled (`coverage run -m pytest`).
   - Generates interactive HTML (`coverage_report/`) and XML (`coverage.xml`) reports.
   - Enforces the 80% overall branch/statement coverage quality gate via `.coveragerc` and `coverage report -m`.
   - On pull requests targeting `develop` or `master`, enforces an 80% changed-line coverage quality gate via `diff-cover coverage.xml --compare-branch origin/<base_ref> --fail-under 80`.
   - Uploads `coverage-report` as an artifact containing `coverage_report/`, `coverage.xml`, and `.coverage`.
   - Executes on the primary development environment (`ubuntu-latest`, Python 3.11).
   - Skipped during fast feedback (`if: ${{ !inputs.fast-feedback }}`).
5. **`package-build`**:
   - Installs modern build tooling (`build`, `twine`, `check-wheel-contents`).
   - Builds both source distribution (`sdist`) and binary wheel (`bdist_wheel`) via `python -m build`.
   - Validates package metadata strictly with `twine check --strict dist/*`.
   - Analyzes wheel file contents with `check-wheel-contents dist/`.
   - Executes packaging regression tests (`pytest tests/unit/test_packaging.py -v`).
   - Verifies wheel installability and metadata loading in an isolated step.
   - Uploads distribution packages as a workflow artifact (`dist-packages`).
   - Skipped during fast feedback (`if: ${{ !inputs.fast-feedback }}`).

### 2.2 Entry-Point Workflow (`.github/workflows/ci.yml`)
`ci.yml` defines event triggers, concurrency controls, and computes the `fast-feedback` input:

```yaml
jobs:
  validate:
    name: Package Validation
    uses: ./.github/workflows/python-validation.yml
    with:
      fast-feedback: ${{ github.event_name == 'push' && github.ref_name != 'develop' && github.ref_name != 'master' }}
```

---

## 3. Trigger Behavior & Concurrency

### 3.1 Workflow Execution Profiles

| Trigger Event | Target Branch | Mode (`fast-feedback`) | Executed Jobs | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`push`** | `feature/**` | `true` (Fast Feedback) | `Fast Unit Tests (Ubuntu - Py3.11)` | Rapid feedback during active development; minimal runner consumption |
| **`pull_request`** | `develop` or `master` | `false` (Full Validation) | All 6 `unit-tests`, all 6 `integration-tests`, `coverage`, `package-build` | Comprehensive pre-merge quality gate across OS and Python matrix |
| **`push`** | `develop` or `master` | `false` (Full Validation) | All 6 `unit-tests`, all 6 `integration-tests`, `coverage`, `package-build` | Post-merge verification on protected release and integration branches |

### 3.2 Redundancy Prevention & Concurrency
- **PR Update Optimization**: When pushing new commits to an open PR branch (`feature/**`), the `push` event triggers fast unit tests (1 job), while the `pull_request` event triggers full validation (14 parallel jobs). This avoids running duplicate full 14-job suites across both event triggers.
- **In-Progress Cancellation**: Concurrency groups ensure outdated in-flight builds are automatically cancelled when new commits are pushed:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

*Note*: Release tagging, version increments, and PyPI publication are deliberately excluded from CI triggers (deferred to **SFNN-34**).

---

## 4. GitHub Branch Protection & Required Status Checks

When configuring GitHub branch protection rules for protected branches (`develop` and `master`):

1. **Required Status Checks on Protected Branches**:
   - Because `pull_request` events targeting `develop` and `master` always execute with `fast-feedback: false`, all matrix jobs and quality gates run on every PR.
   - Recommended required status checks in GitHub Branch Protection rules:
     - `Package Validation / Unit Tests (ubuntu-latest - Py3.11)`
     - `Package Validation / Unit Tests (ubuntu-latest - Py3.9)`
     - `Package Validation / Unit Tests (ubuntu-latest - Py3.12)`
     - `Package Validation / Unit Tests (windows-latest - Py3.9)`
     - `Package Validation / Unit Tests (windows-latest - Py3.11)`
     - `Package Validation / Unit Tests (macos-latest - Py3.11)`
     - `Package Validation / Integration Tests (ubuntu-latest - Py3.11)`
     - `Package Validation / Coverage Collection (Ubuntu - Py3.11)`
     - `Package Validation / Package Build & Metadata Validation`
     - `Dependency review / dependency-review`
2. **Feature Branch Non-Interference**:
   - Feature branches (`feature/**`) do not have branch protection rules requiring the full matrix. Pushes to feature branches run the dedicated `fast-unit-tests` job (`Package Validation / Fast Unit Tests (Ubuntu - Py3.11)`) without blocking developers or generating false failure states for skipped jobs.

---

## 5. Test Matrix & Platform Support

### 5.1 Multi-Platform Matrix (Full Validation)

The CI matrix balances coverage across operating systems and Python versions without creating an expensive full cross-product:

| Platform | Python Version | Role / Status |
| :--- | :--- | :--- |
| **Ubuntu (`ubuntu-latest`)** | `3.9` | Legacy baseline runtime |
| **Ubuntu (`ubuntu-latest`)** | `3.11` | Primary development target (fast-feedback host, coverage & build host) |
| **Ubuntu (`ubuntu-latest`)** | `3.12` | Modernization target |
| **Windows (`windows-latest`)** | `3.9` | Legacy Windows baseline (locally verified) |
| **Windows (`windows-latest`)** | `3.11` | Modern Windows target |
| **macOS (`macos-latest`)** | `3.11` | Modern macOS target (validating `tensorflow==2.16.2` darwin marker) |

### 5.2 Platform-Specific Dependencies
- **Linux / Windows**: Installs `tensorflow==2.19.0` and `keras==3.10.0`.
- **macOS (`darwin`)**: Installs `tensorflow==2.16.2` and `keras==3.10.0` per environment markers declared in `pyproject.toml`.

### 5.3 Known Limitations & Unvalidated Configurations
1. **Python 3.13**: Blocked upstream due to the absence of official TensorFlow 2.19.0 binary wheels (`cp313`) on PyPI. Python 3.13 is excluded from the CI matrix until upstream TensorFlow provides wheels.
2. **Alternative Keras 3 Backends**: Only the default TensorFlow backend is validated. PyTorch and JAX backends are not currently part of the CI matrix.
3. **Post-Push GitHub Verification Factors**:
   - Execution on GitHub-hosted remote runners across macOS and Ubuntu environments (local baseline verification is on Windows).
   - UI presentation of conditionally skipped jobs in GitHub Actions workflow visualization.
   - Enforcement of specific branch protection check name strings in repository settings.

---

## 6. Local Equivalents of CI Commands

Developers can run the exact commands executed by CI workflows locally:

### 6.1 Installation
```bash
# Upgrade pip and install package in editable mode with test dependencies
python -m pip install --upgrade pip
pip install -e ".[test]"
```

### 6.2 Unit and Integration Tests
```bash
# Run unit tests (fast development feedback)
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run entire test suite
pytest -v
```

### 6.3 Coverage Collection
```bash
# Run test suite with branch coverage and generate reports enforcing quality gate
coverage run -m pytest
coverage html -d coverage_report
coverage xml -o coverage.xml
coverage report -m

# Run diff coverage locally against develop (or target PR base branch)
diff-cover coverage.xml --compare-branch origin/develop --fail-under 80
```

### 6.4 Package Build & Validation
```bash
# Install packaging tools
pip install build twine check-wheel-contents

# Build sdist and wheel
python -m build

# Validate distribution metadata
twine check --strict dist/*

# Inspect wheel contents
check-wheel-contents dist/

# Run packaging regression tests
pytest tests/unit/test_packaging.py -v
```

---

## 7. Legacy CircleCI Decommissioning

Under **SFNN-12**, `.circleci/config.yml` has been decommissioned and removed from the repository.

### 7.1 Mapped Responsibilities (SFNN-12)
- CircleCI `build` job -> GitHub Actions `package-build` job (`python -m build`, `twine check`, `check-wheel-contents`, `test_packaging.py`).
- CircleCI `test` job -> GitHub Actions `unit-tests`, `integration-tests`, and `coverage` jobs.
- CircleCI `store_artifacts` -> GitHub Actions `actions/upload-artifact@v4` for coverage reports and distribution packages.

### 7.2 Deferred Responsibilities (SFNN-34)
The following CircleCI release capabilities are intentionally not implemented in SFNN-12 and are deferred to **SFNN-34 (Guarded Semantic Versioning & Release Automation)**:
- CircleCI `hold` manual approval gate.
- CircleCI `bump-version` job (bump2version, automated commit, git tagging, and branch pushing).
- CircleCI `publish` job (uploading distributions to PyPI via Twine on release tags).
