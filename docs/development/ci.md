# Continuous Integration (CI) Architecture & Validation (SFNN-12)

## 1. Overview

This document describes the GitHub Actions continuous integration architecture for `sofenn`, implemented under **SFNN-12** to replace legacy CircleCI validation (`.circleci/config.yml`).

The CI system validates package buildability, distribution metadata, unit and integration test suites, and code coverage across supported operating systems and Python runtimes.

---

## 2. Workflow Architecture

The CI pipeline is structured into a modular, reusable validation workflow and a lightweight entry-point trigger workflow:

```text
.github/workflows/
├── ci.yml                 # Entry-point workflow for pushes and pull requests
└── python-validation.yml  # Reusable workflow (workflow_call) implementing validation jobs
```

### 2.1 Reusable Workflow (`.github/workflows/python-validation.yml`)
The reusable workflow is invoked via `workflow_call` and contains four isolated, parallelizable jobs:

1. **`unit-tests`**:
   - Executes the targeted unit test suite (`pytest tests/unit -v`).
   - Runs across the multi-platform Python matrix.
   - Utilizes pip dependency caching keyed on `pyproject.toml`.
2. **`integration-tests`**:
   - Executes the end-to-end integration test suite (`pytest tests/integration -v`).
   - Runs across the multi-platform Python matrix in parallel with unit tests.
   - Utilizes pip dependency caching keyed on `pyproject.toml`.
3. **`coverage`**:
   - Executes the full test suite with branch measurement enabled (`coverage run -m pytest`).
   - Generates console, HTML, and XML coverage reports adhering to `.coveragerc`.
   - Uploads `coverage-report` as an artifact containing `coverage_report/`, `coverage.xml`, and `.coverage`.
   - Executes on the primary development environment (`ubuntu-latest`, Python 3.11).
4. **`package-build`**:
   - Installs modern build tooling (`build`, `twine`, `check-wheel-contents`).
   - Builds both source distribution (`sdist`) and binary wheel (`bdist_wheel`) via `python -m build`.
   - Validates package metadata strictly with `twine check --strict dist/*`.
   - Analyzes wheel file contents with `check-wheel-contents dist/`.
   - Executes packaging regression tests (`pytest tests/unit/test_packaging.py -v`).
   - Verifies wheel installability and metadata loading in an isolated step.
   - Uploads distribution packages as a workflow artifact (`dist-packages`).

### 2.2 Entry-Point Workflow (`.github/workflows/ci.yml`)
`ci.yml` defines event triggers and concurrency controls, delegating all validation tasks to `./.github/workflows/python-validation.yml`.

---

## 3. Trigger Behavior & Concurrency

### 3.1 Supported Triggers
The CI workflow runs on:
- **Pushes to integration branches**: `develop`
- **Pushes to production release branch**: `master`
- **Pushes to feature branches**: `feature/**`
- **Pull Requests**: Targeting `develop` or `master`

### 3.2 Concurrency & Deduplication
To prevent redundant or overlapping executions during rapid pushes while preserving necessary feature branch validation, concurrency is configured with cancellation enabled:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

*Note*: Release tagging, version increments, and PyPI publication are deliberately excluded from CI triggers (deferred to **SFNN-34**).

---

## 4. Test Matrix & Platform Support

### 4.1 Initial Multi-Platform Matrix

The CI matrix balances coverage across operating systems and Python versions without creating an expensive full cross-product:

| Platform | Python Version | Role / Status |
| :--- | :--- | :--- |
| **Ubuntu (`ubuntu-latest`)** | `3.9` | Legacy baseline runtime |
| **Ubuntu (`ubuntu-latest`)** | `3.11` | Preferred modern development target (coverage & build host) |
| **Ubuntu (`ubuntu-latest`)** | `3.12` | Modernization target |
| **Windows (`windows-latest`)** | `3.9` | Legacy Windows baseline (locally verified) |
| **Windows (`windows-latest`)** | `3.11` | Modern Windows target |
| **macOS (`macos-latest`)** | `3.11` | Modern macOS target (validating `tensorflow==2.16.2` darwin marker) |

### 4.2 Platform-Specific Dependencies
- **Linux / Windows**: Installs `tensorflow==2.19.0` and `keras==3.10.0`.
- **macOS (`darwin`)**: Installs `tensorflow==2.16.2` and `keras==3.10.0` per environment markers declared in `pyproject.toml`.

### 4.3 Known Limitations & Unvalidated Configurations
1. **Python 3.13**: Blocked upstream due to the absence of official TensorFlow 2.19.0 binary wheels (`cp313`) on PyPI. Python 3.13 is excluded from the CI matrix until upstream TensorFlow provides wheels.
2. **Alternative Keras 3 Backends**: Only the default TensorFlow backend is validated. PyTorch and JAX backends are not currently part of the CI matrix.

---

## 5. Local Equivalents of CI Commands

Developers can run the exact commands executed by CI workflows locally:

### 5.1 Installation
```bash
# Upgrade pip and install package in editable mode with test dependencies
python -m pip install --upgrade pip
pip install -e ".[test]"
```

### 5.2 Unit and Integration Tests
```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run entire test suite
pytest -v
```

### 5.3 Coverage Collection
```bash
# Run test suite with branch coverage and generate reports
coverage run -m pytest
coverage report -m
coverage html -d coverage_report
coverage xml -o coverage.xml
```

### 5.4 Package Build & Validation
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

## 6. Legacy CircleCI Decommissioning

Under **SFNN-12**, `.circleci/config.yml` has been decommissioned and removed from the repository.

### 6.1 Mapped Responsibilities (SFNN-12)
- CircleCI `build` job -> GitHub Actions `package-build` job (`python -m build`, `twine check`, `check-wheel-contents`, `test_packaging.py`).
- CircleCI `test` job -> GitHub Actions `unit-tests`, `integration-tests`, and `coverage` jobs.
- CircleCI `store_artifacts` -> GitHub Actions `actions/upload-artifact@v4` for coverage reports and distribution packages.

### 6.2 Deferred Responsibilities (SFNN-34)
The following CircleCI release capabilities are intentionally not implemented in SFNN-12 and are deferred to **SFNN-34 (Guarded Semantic Versioning & Release Automation)**:
- CircleCI `hold` manual approval gate.
- CircleCI `bump-version` job (bump2version, automated commit, git tagging, and branch pushing).
- CircleCI `publish` job (uploading distributions to PyPI via Twine on release tags).
