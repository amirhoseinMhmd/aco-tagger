"""The layering, enforced.

A layer diagram in a README rots the first time someone adds a convenient
import. This walks the actual source with :mod:`ast` and fails if the
dependency direction is ever violated, so the architecture is a property of
the code rather than a claim about it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "pos_aco"

# A layer may import itself and anything to its right. Nothing else.
ALLOWED = {
    "domain": {"domain"},
    "search": {"search", "domain"},
    "application": {"application", "search", "domain"},
    "adapters": {"adapters", "application", "domain"},
}

# Third-party imports are allowed only where they are an implementation
# detail of talking to the outside world.
THIRD_PARTY_ALLOWED_IN = {"adapters"}
STANDARD_LIBRARY_EXCEPTIONS = {"__future__"}


def modules_of(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).glob("*.py"))


def imports_in(path: Path) -> list[tuple[str, int]]:
    """Every ``pos_aco`` module this file imports, as (layer, line)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:  # absolute: not an intra-package import
            continue
        # level 1 == same layer, level 2 == "..other_layer.module"
        if node.level == 1:
            found.append((path.parent.name, node.lineno))
        elif node.module:
            found.append((node.module.split(".")[0], node.lineno))
    return found


class LayerDependencyTest(unittest.TestCase):
    def test_every_layer_directory_exists(self):
        for layer in ALLOWED:
            self.assertTrue((PACKAGE / layer).is_dir(), f"missing layer: {layer}")

    def test_no_layer_imports_one_below_it(self):
        for layer, permitted in ALLOWED.items():
            for path in modules_of(layer):
                for imported, line in imports_in(path):
                    with self.subTest(module=f"{layer}/{path.name}", line=line):
                        self.assertIn(
                            imported,
                            permitted,
                            f"{layer}/{path.name}:{line} imports '{imported}'. "
                            f"{layer} may only import {sorted(permitted)}.",
                        )

    def test_the_domain_is_self_contained(self):
        """The strictest rule, called out separately for a clearer failure."""
        for path in modules_of("domain"):
            for imported, line in imports_in(path):
                self.assertEqual(
                    imported,
                    "domain",
                    f"domain/{path.name}:{line} reaches out to '{imported}'",
                )

    def test_the_application_never_reaches_for_an_adapter(self):
        """Ports are declared inward and implemented outward."""
        for path in modules_of("application"):
            for imported, line in imports_in(path):
                self.assertNotEqual(
                    imported,
                    "adapters",
                    f"application/{path.name}:{line} imports an adapter; "
                    "declare a protocol in application/ports.py instead",
                )


class PurityTest(unittest.TestCase):
    """The inner layers must stay free of I/O and third-party packages."""

    def top_level_imports(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        return names - STANDARD_LIBRARY_EXCEPTIONS

    def test_only_adapters_may_touch_third_party_packages(self):
        for layer in ALLOWED:
            if layer in THIRD_PARTY_ALLOWED_IN:
                continue
            for path in modules_of(layer):
                with self.subTest(module=f"{layer}/{path.name}"):
                    self.assertNotIn("nltk", self.top_level_imports(path))

    def test_the_domain_does_no_file_or_path_work(self):
        for path in modules_of("domain"):
            imported = self.top_level_imports(path)
            with self.subTest(module=path.name):
                self.assertNotIn("csv", imported)
                self.assertNotIn("pathlib", imported)


class PublicApiTest(unittest.TestCase):
    def test_the_facade_re_exports_every_name_it_lists(self):
        import pos_aco

        for name in pos_aco.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pos_aco, name))

    def test_each_layer_exposes_its_own_facade(self):
        import importlib

        for layer in ALLOWED:
            module = importlib.import_module(f"pos_aco.{layer}")
            with self.subTest(layer=layer):
                self.assertTrue(module.__all__)
                for name in module.__all__:
                    self.assertTrue(hasattr(module, name), f"{layer}.{name}")


if __name__ == "__main__":
    unittest.main()
