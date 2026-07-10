from __future__ import annotations

import ast
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

from flightrecorder.path_safety import (
    assert_safe_output_directory,
    path_has_symlink_component,
    replace_owned_output_directory,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPTS = (
    ROOT / "scripts" / "external_verification_smoke.py",
    ROOT / "scripts" / "live_verifier_smoke.py",
    ROOT / "scripts" / "live_hermes_smoke.py",
    ROOT / "scripts" / "live_openclaw_smoke.py",
    ROOT / "scripts" / "live_coven_smoke.py",
)


class PathHasSymlinkComponentTests(unittest.TestCase):
    def test_relative_parent_walk_detects_sibling_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            working_dir = root / "working"
            target_dir = root / "target"
            working_dir.mkdir()
            target_dir.mkdir()
            sibling_link = root / "sibling-link"
            try:
                sibling_link.symlink_to(target_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with chdir(working_dir):
                candidate = Path("../sibling-link/artifact.json")
                self.assertTrue(path_has_symlink_component(candidate, include_leaf=False))


class SafeOutputDirectoryTests(unittest.TestCase):
    def test_owned_replacement_refuses_unowned_nonempty_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "unrelated.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to replace unrecognized"):
                replace_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=True,
                    label="test output",
                    is_owned=lambda _path: False,
                )

            self.assertTrue((target / "unrelated.txt").exists())

    def test_owned_replacement_requires_force_and_deletes_only_owned_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "output"
            target.mkdir()
            (target / "marker.json").write_text("{}", encoding="utf-8")
            def owned(path: Path) -> bool:
                return (path / "marker.json").is_file()

            with self.assertRaisesRegex(ValueError, "pass --force"):
                replace_owned_output_directory(
                    target,
                    repo_root=root / "repo",
                    force=False,
                    label="test output",
                    is_owned=owned,
                )

            replace_owned_output_directory(
                target,
                repo_root=root / "repo",
                force=True,
                label="test output",
                is_owned=owned,
            )
            self.assertFalse(target.exists())

    def test_allows_nested_relative_output_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_dir = repo_root / "runs" / "smoke"
            output_dir.mkdir(parents=True)

            with chdir(repo_root):
                assert_safe_output_directory(Path("runs/smoke"), repo_root=repo_root)

    def test_rejects_filesystem_root_and_protected_roots_or_ancestors(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo_root = base / "repo"
            cwd = base / "workspace" / "nested"
            repo_root.mkdir()
            cwd.mkdir(parents=True)

            unsafe_targets = {
                "filesystem root": Path(repo_root.anchor),
                "repository root": repo_root,
                "repository ancestor": repo_root.parent,
                "working directory": cwd,
                "working directory ancestor": cwd.parent,
            }
            for case_name, target in unsafe_targets.items():
                with self.subTest(case=case_name):
                    with self.assertRaisesRegex(ValueError, "protected|filesystem root"):
                        assert_safe_output_directory(target, repo_root=repo_root, cwd=cwd)

    def test_rejects_symlinked_parent_and_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo_root = base / "repo"
            cwd = base / "workspace"
            external = base / "external"
            repo_root.mkdir()
            cwd.mkdir()
            (external / "nested").mkdir(parents=True)
            linked_parent = repo_root / "linked-parent"
            linked_destination = repo_root / "linked-destination"
            try:
                linked_parent.symlink_to(external, target_is_directory=True)
                linked_destination.symlink_to(external / "nested", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            for target in (linked_parent / "nested", linked_destination):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        assert_safe_output_directory(target, repo_root=repo_root, cwd=cwd)


class SmokeScriptGuardTests(unittest.TestCase):
    def test_every_rmtree_is_immediately_preceded_by_shared_guard(self) -> None:
        for script in SMOKE_SCRIPTS:
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            rmtree_count = 0
            owned_replacement_count = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "replace_owned_output_directory"
            )
            for node in ast.walk(tree):
                for field_name in ("body", "orelse", "finalbody"):
                    statements = getattr(node, field_name, None)
                    if not isinstance(statements, list):
                        continue
                    for index, statement in enumerate(statements):
                        rmtree_call = _expression_call(statement, owner="shutil", name="rmtree")
                        if rmtree_call is None:
                            continue
                        rmtree_count += 1
                        self.assertGreater(index, 0, f"unguarded rmtree in {script}")
                        guard_call = _expression_call(
                            statements[index - 1],
                            owner=None,
                            name="assert_safe_output_directory",
                        )
                        self.assertIsNotNone(guard_call, f"unguarded rmtree in {script}")
                        assert guard_call is not None
                        self.assertEqual(
                            ast.dump(guard_call.args[0]),
                            ast.dump(rmtree_call.args[0]),
                            f"guard checks a different target in {script}",
                        )
            self.assertGreater(
                rmtree_count + owned_replacement_count,
                0,
                f"expected a guarded removal path in {script}",
            )


def _expression_call(statement: ast.stmt, *, owner: str | None, name: str) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if owner is None:
        return call if isinstance(call.func, ast.Name) and call.func.id == name else None
    if not isinstance(call.func, ast.Attribute) or call.func.attr != name:
        return None
    return call if isinstance(call.func.value, ast.Name) and call.func.value.id == owner else None


if __name__ == "__main__":
    unittest.main()
