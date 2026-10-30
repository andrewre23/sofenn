# Python and Keras Compatibility Baseline

## 1. Overview

This document establishes the Python, Keras, TensorFlow, and platform compatibility baseline for `sofenn` as part of **SFNN-22**. It records the current declared vs. observed compatibility status, local baseline test execution outcomes, recommended version matrix for modernization, and actionable requirements for packaging (**SFNN-33**) and CI workflow implementation (**SFNN-12**).

---

## 2. Current Declared Configuration & Discrepancies

### 2.1 Configuration Files Audit

| File | Declared Configuration | Notes / Status |
| :--- | :--- | :--- |
| `pyproject.toml` | `requires-python = ">=3.9"`<br>Classifiers: Python 3, Python 3.9, MIT License, etc.<br>`dependencies`: `keras==3.10.0`, `tensorflow==2.19.0` (win32/linux), `tensorflow==2.16.2` (darwin)<br>`optional-dependencies`: `test`, `build`, `demo`, `dev` | Centralized single source of truth for packaging metadata and dependencies following PEP 517/518/621 (SFNN-33). Package discovery configured for `src/sofenn`. |
| `setup.py` | Removed (SFNN-33) | Superseded by declarative `pyproject.toml`. |
| `requirements.txt` | Removed (SFNN-33) | Core runtime dependencies consolidated into `pyproject.toml` `dependencies`. |
| `build-requirements.txt` | Removed (SFNN-33) | Test and build dependencies consolidated into `pyproject.toml` `[project.optional-dependencies.test]` and `build`. |
| `demo-requirements.txt` | Removed (SFNN-33) | Demo dependencies consolidated into `pyproject.toml` `[project.optional-dependencies.demo]`. |
| `pytest.ini` | `pythonpath = . src`<br>`markers = requires_trainable_backend` | Marker `requires_trainable_backend` registered but not actively applied in existing test files. |
| `tests/conftest.py` | Uses `tf.compat.v1.set_random_seed(seed_value)` | Uses legacy TensorFlow 1.x compatibility API for seeding instead of modern Keras 3 / TensorFlow 2.x seeding (`tf.random.set_seed` or `keras.utils.set_random_seed`). |
| `.circleci/config.yml` | Removed (SFNN-12) | Superseded by GitHub Actions modular workflows (`.github/workflows/ci.yml` and `.github/workflows/python-validation.yml`). |

### 2.2 Platform-Specific Findings & Upstream Verification

- **Windows (`win32`)**: Native TensorFlow wheels (`tensorflow==2.19.0`) run on CPU under Windows x86_64. GPU support on native Windows was discontinued after TensorFlow 2.10, but CPU inference and training are fully functional for `sofenn` test suites.
- **Linux (`linux`)**: TensorFlow 2.19.0 provides standard x86_64 and aarch64 wheels.
- **macOS (`darwin`)**: Pinned to TensorFlow 2.16.2 in `pyproject.toml` (migrated from legacy `requirements.txt`), while Linux/Windows use 2.19.0. TensorFlow 2.19.0 publishes wheels for macOS (Apple Silicon `arm64` and Intel `x86_64`), suggesting that platform version harmonization is viable pending CI matrix validation.
- **Python 3.13 Wheel Availability Verification**: Verified against authoritative PyPI package distribution metadata (`https://pypi.org/pypi/tensorflow/json` for release `2.19.0`). TensorFlow 2.19.0 publishes binary wheels exclusively for `cp39`, `cp310`, `cp311`, and `cp312`. No `cp313` distribution files exist on PyPI for TensorFlow 2.19.0. Consequently, Python 3.13 support is confirmed to be blocked upstream by TensorFlow binary distribution availability.

---

## 3. Tested Local Environment & Baseline Results

### 3.1 Tested Local Environment Profile

The only locally verified configuration in this baseline assessment is:

- **Operating System**: Windows 11 Pro (win32, AMD64 / x86_64)
- **Python Version**: `Python 3.9.13` (`tags/v3.9.13:6de2ca5`, May 17 2022)
- **Keras Version**: `3.10.0`
- **TensorFlow Version**: `2.19.0`
- **NumPy Version**: `2.0.2`
- **Test / Build Tooling**: `pytest 8.4.1`, `coverage 7.9.1`, `pandas 2.3.0`, `scikit-learn 1.5.2`, `scipy 1.13.1`

### 3.2 Test Execution Summary

Tests were executed using the local virtual environment Python executable without altering any production code or dependency specifications:

| Test Scope | Command | Total | Passed | Failed | Skipped | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unit Tests** | `python -m pytest tests/unit -v` | 163 | 163 | 0 | 0 | 51.53s |
| **Integration Tests** | `python -m pytest tests/integration -v` | 2 | 2 | 0 | 0 | 3.91s |
| **Full Suite** | `python -m pytest` | 165 | 165 | 0 | 0 | 50.49s |
| **Coverage Suite** | `coverage run -m pytest; coverage report` | 165 | 165 | 0 | 0 | 60.67s |

### 3.3 Code Coverage Metrics and Terminology

Executing `coverage run -m pytest; coverage report` with branch measurement enabled produced the following breakdown:

```text
Name                                              Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------------
src\sofenn\FuzzyNetwork.py                           67      0     14      0   100%
src\sofenn\SelfOrganizer.py                         206      0     56      1    99%
src\sofenn\callbacks\FuzzyWeightsInitializer.py      39      0     10      0   100%
src\sofenn\layers\FuzzyLayer.py                      42      0      4      0   100%
src\sofenn\layers\NormalizeLayer.py                  19      0      0      0   100%
src\sofenn\layers\OutputLayer.py                     34      0      4      1    97%
src\sofenn\layers\WeightedLayer.py                   33      0      2      0   100%
src\sofenn\losses\CustomLoss.py                      20      0      4      0   100%
src\sofenn\utils\layers.py                           22      0      4      0   100%
-----------------------------------------------------------------------------------
TOTAL                                               482      0     98      2    99%
```

To maintain technical accuracy, statement coverage is distinguished from combined statement/branch coverage:
- **Statement Coverage**: **100%** (482 executed / 482 total statements, 0 missed).
- **Branch Coverage**: **97.96%** (96 fully covered / 98 branch opportunities; 2 partial branch transitions in `SelfOrganizer.py` [line condition] and `OutputLayer.py` [activation check]).
- **Combined Statement and Branch Coverage**: **99.66%** (578/580 coverage opportunities), displayed as 99% by Coverage.py.

---

## 4. Compatibility Assessment & Proposed Matrix

### 4.1 Categorization of Compatibility Status

To prevent conflating locally tested baselines with proposed targets or unverified configurations:

1. **Locally Verified**:
   - Windows 11 (x86_64) + Python 3.9.13 + Keras 3.10.0 + TensorFlow 2.19.0 + NumPy 2.0.2. (165/165 tests passed, 100% statement coverage).
2. **Proposed Support Targets (Candidate Target Matrix)**:
   - Python 3.10, 3.11, 3.12 across Linux (`ubuntu-latest`), Windows (`windows-latest`), and macOS (`macos-latest`).
3. **Unverified Configurations (Require CI Matrix Validation under SFNN-12)**:
   - Linux and macOS environments under modern Python versions (3.10, 3.11, 3.12).
   - Windows environments running Python 3.10, 3.11, or 3.12.
   - Candidate dependency range lower bounds (`keras>=3.0.0`, `tensorflow>=2.16.0`).
   - Alternative Keras 3 backends (PyTorch / JAX), which are not currently configured or tested.

### 4.2 Minimum Python Version Reassessment & Tradeoffs

| Python Version | Lifecycle Status | Tradeoffs & Evaluation | Modernization Recommendation |
| :--- | :--- | :--- | :--- |
| **Python 3.9** | **EOL (Oct 2025)** | Legacy baseline. Fully functional in local test runs, but unsupported upstream. | Maintain as legacy transitional compatibility during initial migration; deprecate in subsequent cleanup. |
| **Python 3.10** | **EOL (Oct 2026)** | **Transitional Minimum Candidate**:<br>- *Pros*: Broad adoption across existing user environments and scientific stacks; full wheel coverage across all dependencies.<br>- *Cons*: Reached upstream EOL (October 2026); limits adoption of newer Python syntax and CPython optimizations. | Maintain transitional support in CI matrix; evaluate deprecation alongside Python 3.9 in future modernization milestones. |
| **Python 3.11** | **Active (EOL Oct 2027)** | **Longer-Term Minimum / Primary Dev Target**:<br>- *Pros*: Substantial runtime performance gains (CPython 3.11 specializing adaptive interpreter); mature, stable support in all ML libraries; active upstream support window.<br>- *Cons*: Transitional support for Python 3.9 and 3.10 must be maintained until deprecation milestones. | **Recommended primary development version** and future minimum once legacy Python versions (3.9/3.10) are formally deprecated. |
| **Python 3.12** | **Active (EOL Oct 2028)** | Supported by modern Keras 3 and TensorFlow 2.16+. | **Recommended active matrix target** for CI and user installations. |
| **Python 3.13** | **Active (EOL Oct 2029)** | Lacks upstream TensorFlow 2.19 wheels (verified on PyPI). | **Blocked upstream / Unsupported** until TensorFlow publishes compatible binary wheels. |

### 4.3 Proposed Dependency Range Candidates (Unverified Bounds)

*Note: The following version ranges are proposed candidates for evaluation in SFNN-33 and SFNN-12. They are not yet proven compatible across their full span.*

- **Keras Candidate Range**: `keras>=3.0.0,<4.0.0` (locally verified with `3.10.0`; earlier 3.x releases require matrix validation).
- **TensorFlow Candidate Range**: `tensorflow>=2.16.0,<2.20.0` (locally verified with `2.19.0` on Windows; `2.16.2` on macOS and Linux requires matrix validation).

---

## 5. Reproducible Local Setup & Verification Commands

### 5.1 Environment Setup

#### Windows (PowerShell)
```powershell
# Create virtual environment with target Python (e.g., 3.9, 3.10, 3.11, or 3.12)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install package in editable mode with development/test extras
pip install -e ".[test]"

# Optionally install demo dependencies
pip install -e ".[demo]"
```

#### Linux / macOS (Bash / Zsh)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install package in editable mode with development/test extras
pip install -e ".[test]"

# Optionally install demo dependencies
pip install -e ".[demo]"
```

### 5.2 Test Execution Commands

```bash
# Run unit tests only
pytest tests/unit -v

# Run integration tests only
pytest tests/integration -v

# Run complete test suite
pytest

# Run tests with branch coverage reporting
coverage run --branch -m pytest
coverage report
```

---

## 6. Known Limitations & Unresolved Questions

1. **Unverified Multi-OS / Multi-Python Combinations**: All non-Windows-3.9 combinations remain unverified until automated CI is established in SFNN-12.
2. **Upstream Python 3.13 Blocker**: Verified absence of `tensorflow==2.19.0` wheels for `cp313` on PyPI prevents supporting Python 3.13 in initial modernization.
3. **Random Seeding Modernization**: Modernized under SFNN-31 using `keras.utils.set_random_seed` and multi-framework environment seeding.
4. **Platform-Specific Requirement Declarations**: `pyproject.toml` currently splits TensorFlow versions (`2.19.0` vs. `2.16.2` via platform environment markers). Unifying these under candidate bounds requires multi-platform CI verification.
5. **Pytest Custom Markers**: The `requires_trainable_backend` marker in `pytest.ini` is unused across existing tests and will be standardized or cleaned up in subsequent test infrastructure tasks.

---

## 7. Coverage Quality Gate

As established under **SFNN-31**, test execution enforces a minimum combined branch and statement coverage threshold of **95%** in `.coveragerc` and CI. Baseline test coverage stands at **99.66%**. See [Testing Documentation](testing.md) for full details.

---

## 8. Roadmap Implications & Division of Responsibilities

To ensure a structured, non-breaking modernization process, responsibilities are partitioned between **SFNN-33** (packaging), **SFNN-12** (CI matrix validation), and **SFNN-31** (deterministic testing and coverage quality gates):

### 8.1 Implementations Completed under SFNN-33 (Packaging Modernization)

SFNN-33 has completed the declarative packaging consolidation into `pyproject.toml`:

- **Consolidated Packaging Metadata (PEP 517 / 518 / 621)**: Migrated all package metadata and dependencies from legacy `setup.py`, `requirements.txt`, `build-requirements.txt`, and `demo-requirements.txt` into an authoritative `pyproject.toml`.
- **Set Python Version Specification**: Declared `requires-python = ">=3.9"` to preserve backward compatibility during initial modernization.
- **Trove Classifiers**: Maintained standard Python 3 and Python 3.9 classifiers alongside research/development metadata.
- **Dependency Tables & Extras**:
  - `dependencies = ["keras==3.10.0", "tensorflow==2.19.0; sys_platform == 'win32' or sys_platform == 'linux'", "tensorflow==2.16.2; sys_platform == 'darwin'"]`
  - `[project.optional-dependencies]` defining `test`, `build` (backward-compatibility alias), `demo`, and `dev` extras.
- **Packaging Regression Tests**: Added `tests/unit/test_packaging.py` validating metadata, dependency tables, platform markers, and package discovery.

### 8.2 Implementations Completed under SFNN-12 (CI Pipeline & Matrix Validation)

SFNN-12 implemented the GitHub Actions CI pipeline and decommissioned legacy CircleCI:

- **Modular Reusable Workflow**: Created `.github/workflows/python-validation.yml` defining parallel `unit-tests`, `integration-tests`, `coverage`, and `package-build` jobs.
- **Entry-Point CI Trigger Workflow**: Created `.github/workflows/ci.yml` triggering on pushes to `develop`, `master`, `feature/**`, and pull requests targeting `develop` or `master`, with in-progress run cancellation.
- **Multi-Platform Matrix**: Established validation across Ubuntu (`3.9`, `3.11`, `3.12`), Windows (`3.9`, `3.11`), and macOS (`3.11`), respecting platform-specific TensorFlow markers.
- **Distribution & Metadata Validation**: Added package build validation (`build`, `twine check --strict`, `check-wheel-contents`, and packaging regression tests).
- **Branch-Aware Coverage Collection**: Configured `.coveragerc` execution and artifact upload.
- **CircleCI Decommissioning**: Removed `.circleci/config.yml`.
- **CI Documentation**: Authored `docs/development/ci.md`.

### 8.3 Implementations Completed under SFNN-31 (Deterministic Testing & Coverage Quality Gate)

SFNN-31 established reproducible testing and coverage quality gates:

- **Multi-Framework Random Seeding**: Standardized on modern `keras.utils.set_random_seed` and multi-framework seeding in `tests/conftest.py`.
- **Isolated Deterministic Fixtures**: Decoupled dataset generators in `tests/testing_utils.py` using explicit `RandomState(seed)`.
- **Numerical Tolerance Assertions**: Replaced floating-point equality checks with explicit tolerance comparisons.
- **Coverage Quality Gate**: Enforced 95% branch/statement coverage threshold in `.coveragerc` and CI.
- **Testing Documentation**: Authored `docs/development/testing.md`.

### 8.4 Version Isolation and Release Automation Considerations (SFNN-34)

Packaging regression tests in `tests/unit/test_packaging.py` verify that `pyproject.toml` adheres to Semantic Versioning and that runtime and optional dependency specifications do not couple with the package version. However, static packaging tests cannot independently prove that future version-bump tooling leaves dependencies untouched at execution time. SFNN-34 must implement before/after verification (e.g., automated diff assertion of dependency tables across version bumps) within release automation workflows to ensure only the version declaration is updated.
