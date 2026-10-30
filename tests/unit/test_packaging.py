import importlib
import importlib.metadata
import pathlib
import re
import sys
import unittest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from setuptools import find_packages


class PackagingRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = pathlib.Path(__file__).resolve().parents[2]
        cls.pyproject_path = cls.project_root / "pyproject.toml"
        with open(cls.pyproject_path, "rb") as f:
            cls.pyproject_data = tomllib.load(f)

    def test_pyproject_metadata(self):
        project = self.pyproject_data.get("project", {})
        self.assertEqual(project.get("name"), "sofenn")

        # Validate that version conforms to Semantic Versioning (SemVer 2.0.0)
        # without hardcoding a specific release version string
        version = project.get("version")
        self.assertIsInstance(version, str)
        semver_pattern = (
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
            r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        )
        self.assertRegex(
            version,
            semver_pattern,
            f"Project version '{version}' does not conform to semantic versioning",
        )

        self.assertEqual(project.get("description"), "Keras model of a Self-Organizing Fuzzy Network")
        self.assertEqual(project.get("readme"), "README.md")
        self.assertEqual(project.get("requires-python"), ">=3.9")
        self.assertEqual(project.get("license"), {"text": "MIT"})

        authors = project.get("authors", [])
        self.assertEqual(len(authors), 1)
        self.assertEqual(authors[0].get("name"), "Andrew Edmonds")
        self.assertEqual(authors[0].get("email"), "andrewre23@gmail.com")

        urls = self.pyproject_data.get("project", {}).get("urls", {})
        self.assertEqual(urls.get("Homepage"), "https://github.com/andrewre23/sofenn")
        self.assertEqual(urls.get("Repository"), "https://github.com/andrewre23/sofenn")

    def test_runtime_dependencies_and_platform_markers(self):
        dependencies = self.pyproject_data.get("project", {}).get("dependencies", [])
        expected_dependencies = [
            "keras==3.10.0",
            "tensorflow==2.19.0; sys_platform == 'win32' or sys_platform == 'linux'",
            "tensorflow==2.16.2; sys_platform == 'darwin'",
        ]
        self.assertEqual(dependencies, expected_dependencies)

    def test_optional_dependencies_extras(self):
        """
        Verify optional extras definitions and consistency.

        PEP 621 flat metadata does not support internal table inheritance without recursive
        self-referential extras (e.g. `sofenn[test]`). To avoid cyclic resolution quirks across
        different package installers, dependencies are explicitly declared per extra.
        This test enforces consistency across shared dependencies and extras aliases.
        """
        optional_deps = self.pyproject_data.get("project", {}).get("optional-dependencies", {})
        self.assertIn("test", optional_deps)
        self.assertIn("build", optional_deps)
        self.assertIn("demo", optional_deps)
        self.assertIn("dev", optional_deps)

        # Legacy 'build' extra must strictly mirror 'test' for backward compatibility
        self.assertEqual(
            optional_deps["build"],
            optional_deps["test"],
            "Legacy 'build' extra must remain an exact alias of 'test'",
        )

        # 'test' extra must contain required testing tools and conditional Python 3.9/3.10 tomli
        test_deps = optional_deps["test"]
        self.assertIn("pytest==8.4.1", test_deps)
        self.assertIn("coverage==7.9.1", test_deps)
        self.assertIn("diff-cover==10.0.0", test_deps)
        self.assertIn("pandas==2.3.0", test_deps)
        self.assertIn("scikit-learn==1.5.2", test_deps)
        self.assertTrue(
            any("tomli" in dep and "python_version" in dep for dep in test_deps),
            "Expected conditional 'tomli' dependency for python_version < '3.11' in 'test' extra",
        )

        # 'demo' extra must contain demonstration packages
        demo_deps = optional_deps["demo"]
        self.assertIn("matplotlib==3.9.4", demo_deps)
        self.assertIn("jupyter==1.1.1", demo_deps)
        self.assertIn("pandas==2.3.0", demo_deps)
        self.assertIn("scikit-learn==1.5.2", demo_deps)

        # 'dev' extra must contain all requirements from 'test' and 'demo'
        dev_deps = optional_deps["dev"]
        for dep in test_deps:
            self.assertIn(dep, dev_deps, f"Missing test dependency '{dep}' in 'dev' extra")
        for dep in demo_deps:
            self.assertIn(dep, dev_deps, f"Missing demo dependency '{dep}' in 'dev' extra")

        # Consistency check: verify that any duplicate package across extras uses identical requirement specifications
        package_spec_map = {}
        for extra_name, deps in optional_deps.items():
            for dep in deps:
                pkg_name = re.split(r"[=<>!~; ]", dep)[0].strip()
                if pkg_name in package_spec_map:
                    self.assertEqual(
                        dep,
                        package_spec_map[pkg_name],
                        f"Inconsistent dependency specification for '{pkg_name}' across extras: "
                        f"'{dep}' vs '{package_spec_map[pkg_name]}'",
                    )
                else:
                    package_spec_map[pkg_name] = dep

    def test_package_discovery_and_modules(self):
        packages = find_packages(where=str(self.project_root / "src"))
        expected_packages = [
            "sofenn",
            "sofenn.callbacks",
            "sofenn.layers",
            "sofenn.losses",
            "sofenn.utils",
        ]
        self.assertEqual(sorted(packages), sorted(expected_packages))

        expected_modules = [
            "sofenn",
            "sofenn.FuzzyNetwork",
            "sofenn.SelfOrganizer",
            "sofenn.callbacks",
            "sofenn.callbacks.FuzzyWeightsInitializer",
            "sofenn.layers",
            "sofenn.layers.FuzzyLayer",
            "sofenn.layers.NormalizeLayer",
            "sofenn.layers.WeightedLayer",
            "sofenn.layers.OutputLayer",
            "sofenn.losses",
            "sofenn.losses.CustomLoss",
            "sofenn.utils",
            "sofenn.utils.layers",
        ]
        for mod in expected_modules:
            imported = importlib.import_module(mod)
            self.assertIsNotNone(imported)

    def test_version_isolation_for_future_automation(self):
        """
        Verify that package version is isolated from dependency declarations.

        NOTE FOR SFNN-34 (Release Automation):
        Static packaging tests verify that dependency specifications do not embed the package
        version string or project package name. However, static analysis cannot independently prove
        that an external version-bump tool (e.g. bumpversion / bump-my-version) leaves dependencies
        untouched at execution time. SFNN-34 must implement before/after verification in release
        workflows (e.g. validating git diff or dependency tables before and after bumping) to guarantee
        that only the project version field is modified.
        """
        project = self.pyproject_data.get("project", {})
        version = project.get("version")
        self.assertIsInstance(version, str)

        # Ensure dependencies and optional-dependencies do not reference sofenn or embed version
        all_declared_deps = list(project.get("dependencies", []))
        for extra_deps in project.get("optional-dependencies", {}).values():
            all_declared_deps.extend(extra_deps)

        for dep in all_declared_deps:
            self.assertFalse(
                dep.startswith("sofenn"),
                f"Dependency '{dep}' unexpectedly references the project package itself",
            )
            self.assertNotIn(
                f"=={version}",
                dep,
                f"Dependency '{dep}' unexpectedly contains project version '{version}'",
            )

        # Simulate bumping the version in pyproject text and ensure dependency declarations are unaffected
        pyproject_text = self.pyproject_path.read_text(encoding="utf-8")
        simulated_version = "99.99.99"
        simulated_text = re.sub(
            r'^version\s*=\s*["\'].*?["\']',
            f'version = "{simulated_version}"',
            pyproject_text,
            flags=re.MULTILINE,
        )
        if sys.version_info >= (3, 11):
            simulated_data = tomllib.loads(simulated_text)
        else:
            simulated_data = tomllib.loads(simulated_text)

        self.assertEqual(simulated_data["project"]["version"], simulated_version)
        self.assertEqual(
            simulated_data["project"]["dependencies"],
            self.pyproject_data["project"]["dependencies"],
            "Simulated version bump altered core runtime dependencies",
        )
        self.assertEqual(
            simulated_data["project"]["optional-dependencies"],
            self.pyproject_data["project"]["optional-dependencies"],
            "Simulated version bump altered optional dependencies",
        )

    def test_installed_distribution_metadata(self):
        dist = importlib.metadata.distribution("sofenn")
        metadata = dist.metadata
        self.assertEqual(metadata["Name"], "sofenn")

        # Distribution version must dynamically match pyproject.toml declaration
        project_version = self.pyproject_data.get("project", {}).get("version")
        self.assertEqual(
            metadata["Version"],
            project_version,
            f"Installed distribution version '{metadata['Version']}' does not match pyproject.toml version '{project_version}'",
        )
        self.assertEqual(metadata["Summary"], "Keras model of a Self-Organizing Fuzzy Network")


if __name__ == "__main__":
    unittest.main()
