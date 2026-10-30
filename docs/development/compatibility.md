# Python and Keras Compatibility Baseline (SFNN-13)

## 1. Overview

This document establishes the Python, Keras, TensorFlow, and platform compatibility baseline for `sofenn` as part of **SFNN-13**. It records the declared vs. observed compatibility status, local baseline test execution outcomes, version matrix for modernization, security vulnerability remediation rationale, and actionable requirements for packaging (**SFNN-33**) and CI workflow implementation (**SFNN-12**).

### Approved Support Policy
- **Python 3.9**: Unsupported / deprecated.
- **Python 3.10**: Unsupported / deprecated.
- **Python 3.11**: Primary development baseline.
- **Python 3.12**: Officially supported production environment.
- **Python 3.13+**: Not yet verified or advertised as supported (blocked upstream by TensorFlow binary wheel distribution availability).

---

## 2. Current Declared Configuration & Architecture

### 2.1 Configuration Files Audit

| File | Declared Configuration | Notes / Status |
| :--- | :--- | :--- |
| `pyproject.toml` | `requires-python = ">=3.11"`<br>Classifiers: Python 3, Python 3.11, Python 3.12, MIT License, etc.<br>`dependencies`: `keras==3.15.1`, `tensorflow==2.19.0` (win32/linux), `tensorflow==2.16.2` (darwin)<br>`optional-dependencies`: `test`, `build`, `demo`, `dev` | Centralized single source of truth for packaging metadata and dependencies following PEP 517/518/621 (SFNN-33, SFNN-13). Package discovery configured for `src/sofenn`. |
| `setup.py` | Removed (SFNN-33) | Superseded by declarative `pyproject.toml`. |
| `requirements.txt` | Removed (SFNN-33) | Core runtime dependencies consolidated into `pyproject.toml` `dependencies`. |
| `build-requirements.txt` | Removed (SFNN-33) | Test and build dependencies consolidated into `pyproject.toml` `[project.optional-dependencies.test]` and `build`. |
| `demo-requirements.txt` | Removed (SFNN-33) | Demo dependencies consolidated into `pyproject.toml` `[project.optional-dependencies.demo]`. |
| `pytest.ini` | `pythonpath = . src`<br>`markers = requires_trainable_backend` | Marker `requires_trainable_backend` registered for backend-specific test execution. |
| `tests/conftest.py` | Modern multi-backend seeding | Synchronizes Python `random`, `numpy.random`, `tf.random`, and `keras.utils.set_random_seed`. |
| `.circleci/config.yml` | Removed (SFNN-12) | Superseded by GitHub Actions modular workflows (`.github/workflows/ci.yml` and `.github/workflows/python-validation.yml`). |

### 2.2 Security Remediation Rationale (Keras 3.15.1 Upgrade)

The dependency baseline was upgraded from Keras 3.10.0 to **Keras 3.15.1** under SFNN-13 to resolve multiple high-severity Common Vulnerabilities and Exposures (CVEs) and GitHub Security Advisories (GHSA) affecting earlier Keras releases:

1. **GHSA-26c4-7vv6-867j / CVE-2024-55459**: Arbitrary file read / disclosure via HDF5 virtual datasets.
2. **GHSA-5gwj-m78q-7pq3**: Remote Code Execution (RCE) via unsafe `Lambda` deserialization in legacy model loading.
3. **GHSA-74m6-m3xx-3vmj**: Denial of Service (DoS) via crafted tensor shape specifications ("shape bomb").
4. **GHSA-gh82-f9x8-5frx**: Path traversal in `DiskIOStore` caching.
5. **GHSA-m8wh-29wm-52mv**: Information disclosure via HDF5 external link resolution.
6. **GHSA-v2w2-w228-c444**: Arbitrary code execution via `TorchModuleWrapper` deserialization.
7. **GHSA-hqp4-2352-xf5r**: Directory traversal during model archive extraction.

Upgrading to Keras 3.15.1 resolves these advisories without introducing breaking API changes, preserving complete mathematical fidelity and API compatibility across all SOFNN models, custom layers, and self-organization routines.

### 2.3 Platform-Specific Findings & Upstream Wheel Verification

- **Windows (`win32`)**: Native TensorFlow wheels (`tensorflow==2.19.0`) run on CPU under Windows x86_64. Full compatibility verified locally with Keras 3.15.1.
- **Linux (`linux`)**: TensorFlow 2.19.0 provides standard x86_64 and aarch64 wheels, fully compatible with Keras 3.15.1.
- **macOS (`darwin`) Architecture Strategy**:
  - `pyproject.toml` declares `tensorflow==2.16.2; sys_platform == 'darwin'`.
  - *Architecture rationale*: TensorFlow 2.19.0 on PyPI only publishes wheels for `macosx_12_0_arm64` (Apple Silicon). In contrast, TensorFlow 2.16.2 publishes wheels for both `macosx_10_15_x86_64` (Intel Mac) and `macosx_12_0_arm64` (Apple Silicon). Retaining TensorFlow 2.16.2 on macOS ensures broad architecture compatibility across both Intel and Apple Silicon hardware while remaining fully compatible with Keras 3.15.1.
- **Python 3.13 Wheel Availability Verification**:
  - Verified against authoritative PyPI package distribution metadata (`https://pypi.org/pypi/tensorflow/json` for release `2.19.0`). TensorFlow 2.19.0 publishes binary wheels exclusively for `cp39`, `cp310`, `cp311`, and `cp312`. No `cp313` distribution files exist on PyPI for TensorFlow 2.19.0.
  - Python 3.13 is therefore classified as unverified and not yet advertised as supported until upstream TensorFlow publishes compatible binary distributions.

---

## 3. Tested Local Environment & Baseline Results

### 3.1 Tested Local Environment Profile

- **Operating System**: Windows 11 Pro (win32, AMD64 / x86_64)
- **Python Version**: `Python 3.12.10`
- **Keras Version**: `3.15.1`
- **TensorFlow Version**: `2.19.0`
- **NumPy Version**: `2.0.2`
- **Test / Build Tooling**: `pytest 8.4.1`, `coverage 7.9.1`, `pandas 2.3.0`, `scikit-learn 1.5.2`

### 3.2 Test Execution Summary

| Test Scope | Command | Total | Passed | Failed | Skipped | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unit Tests** | `python -m pytest tests/unit -v` | 171 | 171 | 0 | 0 | 58.40s |
| **Integration Tests** | `python -m pytest tests/integration -v` | 2 | 2 | 0 | 0 | 1.03s |
| **Full Suite & Coverage** | `coverage run -m pytest; coverage report` | 173 | 173 | 0 | 0 | 80.32s |

### 3.3 Code Coverage Metrics

Executing `coverage run -m pytest; coverage report` with branch measurement enabled produced the following breakdown:

```text
Name                                              Stmts   Miss Branch BrPart   Cover   Missing
----------------------------------------------------------------------------------------------
src\sofenn\FuzzyNetwork.py                           67      0     14      0 100.00%
src\sofenn\SelfOrganizer.py                         206      0     56      1  99.62%   187->196
src\sofenn\callbacks\FuzzyWeightsInitializer.py      39      0     10      0 100.00%
src\sofenn\layers\FuzzyLayer.py                      42      0      4      0 100.00%
src\sofenn\layers\NormalizeLayer.py                  19      0      0      0 100.00%
src\sofenn\layers\OutputLayer.py                     34      0      4      1  97.37%   117->120
src\sofenn\layers\WeightedLayer.py                   33      0      2      0 100.00%
src\sofenn\losses\CustomLoss.py                      20      0      4      0 100.00%
src\sofenn\utils\layers.py                           22      0      4      0 100.00%
----------------------------------------------------------------------------------------------
TOTAL                                               482      0     98      2  99.66%
```

- **Statement Coverage**: **100.00%** (482 executed / 482 total statements, 0 missed).
- **Branch Coverage**: **97.96%** (96 fully covered / 98 branch opportunities; 2 partial branch transitions).
- **Combined Statement and Branch Coverage**: **99.66%** (578/580 coverage opportunities).

---

## 4. Compatibility Matrix & CI Validation Strategy

### 4.1 Categorization of Compatibility Status

1. **Locally Verified**:
   - Windows 11 (x86_64) + Python 3.12.10 + Keras 3.15.1 + TensorFlow 2.19.0 (173/173 tests passed, 100% statement coverage).
2. **Supported Target Matrix (Gated via CI in SFNN-12 / SFNN-13)**:
   - Ubuntu (`ubuntu-latest`): Python 3.11, Python 3.12
   - Windows (`windows-latest`): Python 3.11, Python 3.12
   - macOS (`macos-latest`): Python 3.11
3. **Unsupported / Unverified Configurations**:
   - Python 3.9 and 3.10: Unsupported (deprecated).
   - Python 3.13+: Blocked upstream by TensorFlow binary distribution availability.

### 4.2 CI Matrix Configuration

The CI matrix in `.github/workflows/python-validation.yml` validates all supported OS and Python runtime combinations:

```yaml
matrix:
  os: [ubuntu-latest, windows-latest]
  python-version: ['3.11', '3.12']
  include:
    - os: macos-latest
      python-version: '3.11'
```

---

## 5. Reproducible Local Setup & Verification Commands

### 5.1 Environment Setup (Python 3.11+)

#### Windows (PowerShell)
```powershell
# Create virtual environment with Python 3.11 or 3.12
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install package with test dependencies
python -m pip install --upgrade pip
pip install -e ".[test]"
```

#### Linux / macOS (Bash / Zsh)
```bash
# Create virtual environment with Python 3.11 or 3.12
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install package with test dependencies
python3 -m pip install --upgrade pip
pip install -e ".[test]"
```

### 5.2 Test Execution Commands

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run complete test suite with coverage
coverage run -m pytest
coverage report -m
```

---

## 6. Known Limitations & Unresolved Questions

1. **GitHub-Hosted CI Confirmation**: Full validation across all matrix configurations (Ubuntu 3.11/3.12, Windows 3.11/3.12, macOS 3.11) requires execution on GitHub Actions runners.
2. **Upstream Python 3.13 Blocker**: Absence of `tensorflow==2.19.0` wheels for `cp313` on PyPI prevents supporting Python 3.13 until upstream builds are published.
3. **Random Seeding Modernization**: Modernized under SFNN-31 using `keras.utils.set_random_seed` and multi-framework environment seeding.
4. **Platform-Specific Requirement Declarations**: `pyproject.toml` pins `tensorflow==2.16.2` on macOS to retain dual Intel (`x86_64`) and Apple Silicon (`arm64`) wheel compatibility, while Windows and Linux utilize `tensorflow==2.19.0`.
5. **Pytest Custom Markers**: The `requires_trainable_backend` marker in `pytest.ini` is registered for backend-specific test execution.

---

## 7. Coverage Quality Gates

As established under **SFNN-31**, test execution enforces a minimum combined branch and statement coverage threshold of **80%** in `.coveragerc` and an **80%** changed-line coverage threshold on pull requests via `diff-cover`. Baseline test coverage stands at **99.66%**. See [Testing Documentation](testing.md) for full details.

---

## 8. Roadmap Implications & Division of Responsibilities

To ensure a structured, non-breaking modernization process, responsibilities are partitioned between **SFNN-33** (packaging), **SFNN-12** (CI matrix validation), **SFNN-31** (deterministic testing and coverage quality gates), and **SFNN-13** (security upgrade & Python 3.11+ baseline):

### 8.1 Implementations Completed under SFNN-33 & SFNN-13 (Packaging Modernization)

SFNN-33 and SFNN-13 completed the declarative packaging consolidation into `pyproject.toml`:

- **Consolidated Packaging Metadata (PEP 517 / 518 / 621)**: Migrated all package metadata and dependencies from legacy `setup.py`, `requirements.txt`, `build-requirements.txt`, and `demo-requirements.txt` into an authoritative `pyproject.toml`.
- **Set Python Version Specification**: Declared `requires-python = ">=3.11"` (SFNN-13).
- **Trove Classifiers**: Maintained standard Python 3, Python 3.11, and Python 3.12 classifiers alongside research/development metadata.
- **Dependency Tables & Extras**:
  - `dependencies = ["keras==3.15.1", "tensorflow==2.19.0; sys_platform == 'win32' or sys_platform == 'linux'", "tensorflow==2.16.2; sys_platform == 'darwin'"]`
  - `[project.optional-dependencies]` defining `test`, `build` (backward-compatibility alias), `demo`, and `dev` extras (with obsolete `tomli` condition removed).
- **Packaging Regression Tests**: Added `tests/unit/test_packaging.py` validating metadata, dependency tables, platform markers, and package discovery.

### 8.2 Implementations Completed under SFNN-12 & SFNN-13 (CI Pipeline & Matrix Validation)

SFNN-12 and SFNN-13 established the GitHub Actions CI pipeline and revised test matrix:

- **Modular Reusable Workflow**: Created `.github/workflows/python-validation.yml` defining parallel `unit-tests`, `integration-tests`, `coverage`, and `package-build` jobs.
- **Entry-Point CI Trigger Workflow**: Created `.github/workflows/ci.yml` triggering on pushes to `develop`, `master`, `feature/**`, and pull requests targeting `develop` or `master`, with in-progress run cancellation.
- **Multi-Platform Matrix**: Established validation across Ubuntu (`3.11`, `3.12`), Windows (`3.11`, `3.12`), and macOS (`3.11`), respecting platform-specific TensorFlow markers.
- **Distribution & Metadata Validation**: Added package build validation (`build`, `twine check --strict`, `check-wheel-contents`, and packaging regression tests).
- **Branch-Aware Coverage Collection**: Configured `.coveragerc` execution and artifact upload.
- **CircleCI Decommissioning**: Removed `.circleci/config.yml`.
- **CI Documentation**: Authored `docs/development/ci.md`.

### 8.3 Implementations Completed under SFNN-31 (Deterministic Testing & Coverage Quality Gate)

SFNN-31 established reproducible testing and coverage quality gates:

- **Multi-Framework Random Seeding**: Standardized on modern `keras.utils.set_random_seed` and multi-framework seeding in `tests/conftest.py`.
- **Isolated Deterministic Fixtures**: Decoupled dataset generators in `tests/testing_utils.py` using explicit `RandomState(seed)`.
- **Numerical Tolerance Assertions**: Replaced floating-point equality checks with explicit tolerance comparisons.
- **Coverage Quality Gates**: Enforced 80% overall branch/statement coverage threshold in `.coveragerc` and 80% PR changed-line coverage gate via `diff-cover` in CI.
- **Testing Documentation**: Authored `docs/development/testing.md`.

### 8.4 Version Isolation and Release Automation Considerations (SFNN-34)

Packaging regression tests in `tests/unit/test_packaging.py` verify that `pyproject.toml` adheres to Semantic Versioning and that runtime and optional dependency specifications do not couple with the package version. However, static packaging tests cannot independently prove that future version-bump tooling leaves dependencies untouched at execution time. SFNN-34 must implement before/after verification (e.g., automated diff assertion of dependency tables across version bumps) within release automation workflows to ensure only the version declaration is updated.
