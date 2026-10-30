# AI Agent Development Guidelines (AGENTS.md)

## 1. Project Overview & Architecture

`sofenn` (pronounced *soften*) is an open-source Python library implementing Self-Organizing Fuzzy Neural Networks using Keras/TensorFlow. The framework provides an adaptive neuro-fuzzy architecture that dynamically optimizes both network structure and continuous parameters during online training.

### Core Architecture
- **`FuzzyNetwork` (`src/sofenn/FuzzyNetwork.py`)**: The underlying parametric neural network model composed of:
  - **Fuzzy Layer (`src/sofenn/layers/FuzzyLayer.py`)**: Radial/ellipsoidal basis membership function (MF) layer representing fuzzy rule premises with Gaussian activation functions parameterized by centers ($c_{ij}$) and widths ($\sigma_{ij}$).
  - **Normalize Layer (`src/sofenn/layers/NormalizeLayer.py`)**: Firing strength normalization across active rule neurons.
  - **Weighted Layer (`src/sofenn/layers/WeightedLayer.py`)**: Rule consequence layer calculating affine linear combinations of input features weighted by rule firing strengths ($w_{2j} = A_j \cdot B$).
  - **Output Layer (`src/sofenn/layers/OutputLayer.py`)**: Aggregation layer computing the unweighted sum of rule outputs, followed by a trainable `Dense` transformation with task-specific activation functions (linear regression, softmax classification). *(Note: The mathematical fidelity of the trainable Dense layer following aggregation is not characterized as established and will be investigated separately under SFNN-18).*
- **`FuzzySelfOrganizer` (`src/sofenn/SelfOrganizer.py`)**: Metamodel that monitors training dynamics and orchestrates online architectural self-organization, including neuron allocation (rule generation) and neuron pruning (rule elimination).
- **Callbacks & Utilities (`src/sofenn/callbacks/`, `src/sofenn/utils/`, `src/sofenn/losses/`)**: Parameter initializers, layer helpers, and custom training objectives.

### Repository Layout
- `src/sofenn/`: Production library source code.
- `tests/`: Comprehensive test suite containing `unit/` and `integration/` tests along with test fixtures in `tests/data/`.
- `demos/`: Jupyter demonstration notebooks for regression and classification workflows.
- `pytest.ini`: Pytest configuration configuring module search paths and test markers.

---

## 2. Research Fidelity & Algorithmic Integrity

### Reference Paper
The theoretical foundation of `sofenn` is defined in:
> **Leng, G., Prasad, G., & McGinnity, T. M. (2004).** *An on-line algorithm for creating self-organizing fuzzy neural networks.* Neural Networks, 17, 1477–1493. DOI: 10.1016/j.neunet.2004.07.009.

### Mathematical Fidelity
- All core algorithms—including Gaussian membership computation, normalization, consequent parameter updates, rule generation criteria (system error and premise distance thresholds), and pruning criteria—must strictly adhere to the mathematical formulations in Leng et al. (2004).
- **Distinguishing Mathematical Defects from Intentional Design Decisions**:
  - *Mathematical Defects*: Algorithmic errors where implementation formulas deviate from the published equations, leading to incorrect gradient computation, invalid normalization, or divergence from theoretical bounds. Any defect identification must cite the specific equation/section from Leng et al. (2004).
  - *Intentional Design Decisions*: Adaptations required for modern deep learning frameworks (e.g., vectorized batching, Keras 3 compatibility, modular layer subclasses, training callbacks).
- Do not silently alter mathematical equations, loss functions, or error metrics without explicit paper-backed justification.

---

## 3. API Stability & Backward Compatibility

- **Public Interface Preservation**: The public API—including `FuzzyNetwork`, `FuzzySelfOrganizer`, and standard layer interfaces—must maintain backward compatibility.
- Do not rename public methods, alter parameter defaults, or change return types without explicit authorization.
- Internal refactorings must preserve external caller expectations.

---

## 4. Testing & Determinism Standards

- **Deterministic Numerical Testing**: Set explicit random seeds across all numerical backends (`tf.random.set_seed`, `np.random.seed`, Python `random.seed`) to ensure reproducible test executions.
- **Test Discipline**:
  - Run fast, targeted unit tests (`tests/unit/`) prior to executing slower integration tests (`tests/integration/`).
  - Maintain regression coverage for both regression and classification model configurations.
  - Honest reporting: Never weaken test assertions, loosen numerical tolerances arbitrarily, disable test cases, or stub failing logic to achieve artificial pass states.

---

## 5. Dependency Management & Single Source of Truth

- **Single Source of Truth**: All package configuration and dependency definitions are centralized in `pyproject.toml` (PEP 517/518/621) to avoid drift across configuration files.
- **Current State**: Package metadata, core dependencies, and optional test/demo/build extras are consolidated in `pyproject.toml`. Legacy `setup.py` and loose requirement files have been removed.

---

## 6. Branching, CI/CD, & Version Control Conventions

### Repository Branching Model
- **`develop`**: Active integration branch for ongoing feature modernization and development (target branch for planned automated patch version increments).
- **`master`**: Production-ready release branch representing stable published baselines (target branch for planned automated minor version increments).
  *(Note: The repository uses `master` rather than `main` as its primary default branch).*
- **`feature/*`**: Topic branches created off `develop` for specific modernization initiatives (e.g., `feature/SFNN-baseline-modernization`).

### Release Strategy
*(Note: Automated versioning and release automation are planned future capabilities to be introduced in upcoming modernization tickets and are not currently implemented).*
- **`develop`**: Automated patch bumps on integration (planned).
- **`master`**: Automated minor bumps upon release promotion (planned).
- **Major Releases**: Manually initiated and gated.
- **PyPI Publishing**: Strictly gated, isolated, and verified via dedicated release automation (planned).

### CI/CD Platform
- **Target Platform**: GitHub Actions CI workflows.
- **Legacy Platform**: Existing CircleCI pipeline (`.circleci/config.yml`) will be decommissioned as part of modernization.

### Version Control Ownership
- **Developer Ownership**: All commits, pushes, pull requests, merges, git tags, and release publications are strictly owned and executed by the human developer.
- **Agent Constraint**: AI agents must not commit, push, merge, create tags, open PRs, or bump version numbers unless explicitly directed.

### Commit Message Convention
When preparing changes or describing commits, follow the ticket-prefixed format:
```text
SFNN-<ticket>: <type>: <description>
```
*Examples*:
- `SFNN-32: docs: create agent development guidelines`
- `SFNN-33: build: consolidate package metadata and dependencies in pyproject.toml`

---

## 7. Baseline Modernization Roadmap

The following modernization roadmap defines upcoming sequential initiatives:

1. **Python / Keras Compatibility**: Validate and ensure compatibility across supported Python runtimes and modern Keras backends.
2. **Metadata & Dependency Consolidation**: Consolidate package metadata, build requirements, and test extras into `pyproject.toml`.
3. **CI Pipeline Migration**: Replace legacy CircleCI with modular GitHub Actions workflows.
4. **Deterministic Test Infrastructure**: Standardize test suites, random seeding, and regression fixtures.
5. **Guarded Semantic Versioning**: Introduce automated, guarded semantic versioning workflows aligned with branch strategies.
6. **Mathematical Verification**: Conduct end-to-end mathematical validation of core learning algorithms against Leng et al. (2004).
