import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_check", ROOT / "tools/check_release.py")
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


class ReleaseTests(unittest.TestCase):
    def test_publication_tree_has_no_flagged_content(self):
        _, errors, _ = release_check.scan(ROOT)
        self.assertEqual(errors, [])

    def test_scanner_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = "hf_" + "a" * 30
            (root / "example.txt").write_text(token)
            _, errors, _ = release_check.scan(root)
            self.assertTrue(any("credential" in error for error in errors))
            self.assertTrue(all(token not in error for error in errors))

    def test_scanner_detects_private_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.txt").write_text("/" + "home" + "/" + "private-user" + "/models")
            _, errors, _ = release_check.scan(root)
            self.assertTrue(any("user path" in error for error in errors))

    def test_license_is_an_explicit_release_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            _, errors, warnings = release_check.scan(Path(directory))
            self.assertEqual(errors, [])
            self.assertTrue(any("LICENSE" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
