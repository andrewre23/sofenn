# Testing Infrastructure, Determinism, and Quality Gates (SFNN-31)

## 1. Overview

This document details the testing architecture, reproducibility standards, and coverage quality gates established under **SFNN-31**.

The testing framework ensures that tests are deterministic, isolated from execution order, reproducible across supported platforms and Python runtimes, and gated by automated coverage standards in CI.

---

## 2. Test Architecture & Structure

The repository test suite is organized into unit and integration tiers located under `tests/`:

```text
tests/
├── conftest.py                             # Global pytest configuration, autouse seeding, and fixtures
├── testing_utils.py                        # Reusable test helpers, problem configurations, and data generators
├── data/                                   # Fixed datasets and pre-trained model fixtures
│   ├── iris/                               # Features and target CSVs for classification
│   └── models/                             # Pre-trained Keras model fixtures (.keras)
├── integration/
│   └── test_sofenn.py                      # End-to-end integration tests for FuzzySelfOrganizer
└── unit/
    ├── test_fuzzy_network.py               # FuzzyNetwork model architecture, fitting, and serialization
    ├── test_self_organizer.py              # FuzzySelfOrganizer heuristics, rule allocation, and pruning
    ├── test_packaging.py                  # Declarative packaging and metadata regression tests
    ├── callbacks/
    │   └── test_fuzzy_weights_initializer.py # Center and width initialization callback tests
    ├── layers/                             # Unit tests for custom fuzzy and aggregation layers
    │   ├── test_fuzzy.py
    │   ├── test_normalize.py
    │   ├── test_output.py
    │   └── test_weighted.py
    ├── losses/
    │   └── test_custom.py                  # Custom loss functions and gradient compatibility
    └── utils/
        └── test_layer_utils.py             # Layer dimension and argument parsing utility tests
```

---

## 3. Determinism & Random Seeding Strategy

### 3.1 Startup Environment Configuration

Environment variables affecting hash randomization and backend CPU kernel dispatch must be established prior to Python interpreter startup or framework initialization:

- **`PYTHONHASHSEED=0`**: Set in the CI environment (`.github/workflows/python-validation.yml`) and initialized at module level in `tests/conftest.py` before runtime imports to ensure consistent dictionary and set iteration order.
- **`TF_ENABLE_ONEDNN_OPTS=0`**: Set in CI environment variables and top-level `tests/conftest.py` prior to importing TensorFlow to disable non-associative oneDNN floating-point kernel optimizations across CPU architectures.

### 3.2 Per-Test Seeding Fixture

To ensure reproducible test execution order independence across Python, NumPy, TensorFlow, and modern Keras backends, `tests/conftest.py` implements an `autouse=True` fixture that executes prior to each test case:

```python
DEFAULT_SEED = 0

@pytest.fixture(autouse=True)
def set_random_seed():
    """Establish deterministic test environment across Python, NumPy, TensorFlow, and Keras."""
    seed_value = DEFAULT_SEED

    # 1. Modern Keras seeding API (synchronizes Python, NumPy, and active backend RNGs)
    keras.utils.set_random_seed(seed_value)

    # 2. Explicit backend seeding for comprehensive coverage
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    tf.random.set_seed(seed_value)
```

---

## 4. Reusable Fixtures & Isolated Data Generation

### 4.1 Pytest Fixtures

`tests/conftest.py` provides clean, reusable fixtures for common test data requirements:

- **`classification_data`**: Provides a deterministic (90/10 train/test split) of the Iris dataset.
- **`regression_data`**: Provides a deterministic synthetic multi-feature regression dataset.
- **`problem_defaults`**: Exposes isolated deep copies (`copy.deepcopy(PROBLEM_DEFAULTS)`) of configuration defaults for classification and regression tasks.

### 4.2 Decoupled Data Generators & Caching (`tests/testing_utils.py`)

To eliminate cross-test state leakage and shared mutable state:
- **Cached Raw Loaders**: Raw disk reads and dataset splitting (`_load_raw_classification_data`, `_load_raw_regression_data`) use `@lru_cache` to minimize repeated I/O overhead.
- **Independent Array Copies**: Generator functions (`_get_classification_data`, `_get_regression_data`, `_get_training_data`) return fresh copies (`tuple(arr.copy() for arr in ...)`) of the cached arrays. Any in-place mutation performed by a test does not corrupt subsequent test executions.
- **Isolated Synthetic RNG**: `_load_raw_regression_data(seed=0)` utilizes an explicit `numpy.random.RandomState(seed)` instance, ensuring exact numeric data generation independent of prior global RNG state.
- **Fresh Model Loading**: `_load_saved_model` loads clean model instances from `tests/data/models/` to ensure isolated state for each caller.

---

## 5. Model Serialization vs. Historical Fixture Compatibility

### 5.1 Serialization Round-Trip Verification

Complete model persistence round-trip tests (`test_fit_classification`, `test_fit_regression` in `tests/unit/test_fuzzy_network.py`) verify full end-to-end serialization fidelity:
1. Train a model instance in memory.
2. Save the model to a temporary directory (`tempfile.TemporaryDirectory()`) using `Path(temp_dir) / 'filename.keras'`.
3. Reload the saved model with `keras.saving.load_model(..., custom_objects={'FuzzyNetwork': FuzzyNetwork})`.
4. Assert numerical identity between the in-memory trained model and the reloaded model predictions using `self.assertAllClose(trained_model.predict(X_test), loaded_model.predict(X_test))`.

### 5.2 Model Save-Only Verification

Targeted save-only tests (`test_saving_model` in `tests/unit/test_fuzzy_network.py`) isolate and verify the `.save()` operation independently across parameterized problem types (classification and regression):
1. Fit a compiled model instance on training data.
2. Save the model artifact to an isolated temporary directory path (`Path(temp_directory) / 'model.keras'`).
3. Confirm clean serialization completion without exceptions, decoupling the write phase from subsequent deserialization checks.

### 5.3 Historical Model Fixture Compatibility

Historical pre-trained `.keras` fixtures stored in `tests/data/models/` (`iris_classification.keras`, `regression.keras`, and their `-deep` counterparts) serve as static baselines:
- **Decoupled Verification**: Historical fixtures are verified via `test_saved_model_fixture_compatibility` and used as input to `FuzzySelfOrganizer` unit tests (`tests/unit/test_self_organizer.py`).
- **No Training Conflation**: Predictions from newly trained models are never asserted against static historical model files. Multi-epoch gradient descent (e.g., 25 epochs of RMSprop) naturally accumulates small floating-point divergences across different framework versions (e.g., TensorFlow 2.16 vs. 2.19) and CPU architectures, which must not be conflated with serialization failure.

---

## 6. Numerical Tolerance Conventions

Because neuro-fuzzy networks and deep learning models rely on continuous numerical operations across diverse hardware architectures:

1. **No Exact Equality on Floats/Tensors**: Floating-point values and Tensor outputs must not be compared using raw `==` or `self.assertEqual` unless validating integer indices or categorical labels.
2. **Explicit Tolerances**:
   - NumPy arrays: Use `numpy.allclose(actual, expected, rtol=1e-5, atol=1e-8)`.
   - Keras/TensorFlow Tensors: Use `self.assertAllClose(actual, expected, rtol=1e-6, atol=1e-6)` inheriting from `keras.src.testing.TestCase`.

---

## 7. Coverage Quality Gate

### 7.1 Threshold Configuration (`.coveragerc`)

A minimum branch/statement coverage threshold is enforced via `.coveragerc`:

```ini
[run]
branch = True

[report]
include = */src/sofenn/*
omit = */src/*/**/__init__.py
fail_under = 95
precision = 2
show_missing = True
```

### 7.2 Threshold Rationale (95% Quality Gate)

- **Observed Baseline**: The test suite achieves **99.66%** combined statement and branch coverage (100% statement, 97.96% branch).
- **Safety Margin**: A **95.00%** quality gate strictly protects the codebase from unverified logic while providing a sensible buffer for legitimate structural adjustments and defensive error handling without triggering false CI failures.
- **No Exclusion Abuse**: Enforcing this threshold ensures tests cover actual execution paths without relying on artificial `pragma: no cover` exclusions on production code.

### 7.3 Report Preservation & CI Integration

The GitHub Actions CI pipeline (`.github/workflows/python-validation.yml`) executes:
1. `coverage run -m pytest`: Runs the full test suite with branch coverage tracking.
2. `coverage html -d coverage_report`: Generates interactive HTML report (`coverage_report/index.html`).
3. `coverage xml -o coverage.xml`: Produces standard Cobertura XML report for tooling integration.
4. `coverage report -m`: Prints detailed console report with missing line numbers and enforces the 95% quality gate (`fail_under = 95`).

HTML, XML, and raw `.coverage` files are uploaded as CI build artifacts on every run.

---

## 8. Reproducible Test Execution Commands

### 8.1 Running Targeted Tests

```bash
# Run unit tests only
pytest tests/unit -v

# Run integration tests only
pytest tests/integration -v

# Run packaging regression tests
pytest tests/unit/test_packaging.py -v

# Run a specific test file
pytest tests/unit/layers/test_fuzzy.py -v
```

### 8.2 Running Full Suite & Coverage Collection

```bash
# Execute entire test suite
pytest -v

# Collect branch coverage and generate reports enforcing quality gate
coverage run -m pytest
coverage html -d coverage_report
coverage xml -o coverage.xml
coverage report -m
```

---

## 9. Limitations of Numerical Reproducibility & Cross-Platform Factors

While software-level seeding, isolated fixtures, and environment variables provide consistent and deterministic test execution within a given environment, developers should note remaining cross-platform factors:
- **No Guarantee of Bitwise Cross-Platform Reproducibility**: Setting `PYTHONHASHSEED`, `TF_ENABLE_ONEDNN_OPTS=0`, and `set_random_seed(0)` does not guarantee bitwise-identical floating-point results across different operating systems, CPU architectures (e.g., x86 AVX2/AVX-512 vs. ARM NEON), or different TensorFlow minor versions.
- **Non-Associative Floating-Point Operations**: Multithreaded BLAS/Eigen matrix multiplication and parallel reduction algorithms execute floating-point operations in non-deterministic order across threads.
- **Multi-Epoch Optimization Drift**: Even minor single-step numerical discrepancies ($< 10^{-7}$) compound non-linearly over dozens of optimization epochs, leading to slight parameter differences between platforms.
- **Mitigation Standard**: Tests avoid asserting exact equality across independent training runs or against static historical weights, relying instead on explicit tolerance assertions (`assertAllClose`, `allclose`) and dedicated serialization round-trips.
