import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
import zipfile

from fold2reason.data.download import safe_unzip
from fold2reason.data.install_archive import install

ROOT = Path(__file__).resolve().parents[1]


class ResourceTests(unittest.TestCase):
    def test_manifest_has_nine_pinned_external_datasets(self):
        data = json.loads((ROOT / "configs/resources.json").read_text())["resources"]
        self.assertEqual(len(data), 10)
        self.assertEqual(sum(v["repo_type"] == "dataset" for v in data.values()), 9)
        for item in data.values():
            self.assertRegex(item["revision"], r"^[a-f0-9]{40}$")
        self.assertEqual(data["spatialviz_bench"]["extract_destinations"]["SpatialViz_Bench_images.zip"],
                         "SpatialViz_Bench_images")

    def test_zip_roundtrip_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            with zipfile.ZipFile(archive, "w") as out:
                out.writestr("images/example.txt", "synthetic")
            safe_unzip(archive, root / "out")
            safe_unzip(archive, root / "out")
            self.assertEqual((root / "out/images/example.txt").read_text(), "synthetic")
            (root / "out/images/example.txt").write_text("different")
            with self.assertRaises(FileExistsError):
                safe_unzip(archive, root / "out")

    def test_zip_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "bad.zip", "w") as out:
                out.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                safe_unzip(root / "bad.zip", root / "out")

    def test_zip_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(root / "bad.zip", "w") as out:
                out.writestr(info, "outside")
            with self.assertRaises(ValueError):
                safe_unzip(root / "bad.zip", root / "out")

    def archive(self, root, name="artifacts/cache/example.txt"):
        path = root / "resource.tar.gz"
        with tarfile.open(path, "w:gz") as out:
            member = tarfile.TarInfo(name)
            member.size = 3
            out.addfile(member, io.BytesIO(b"abc"))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_install_verifies_hash_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self.archive(root)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                install(archive, root / "out", "0" * 64)
            self.assertEqual(install(archive, root / "out", digest), 1)
            self.assertEqual((root / "out/artifacts/cache/example.txt").read_bytes(), b"abc")
            with self.assertRaises(FileExistsError):
                install(archive, root / "out", digest)

    def test_install_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self.archive(root, "artifacts/../../escape.txt")
            with self.assertRaises(ValueError):
                install(archive, root / "out", digest)

    def test_all_upload_archives_have_real_checksums(self):
        catalog = json.loads((ROOT / "docs/release_assets.json").read_text())
        self.assertEqual(catalog["distribution_scope"], "data-decoder-and-qwen35-9b-main-weights")
        self.assertEqual(len(catalog["archives"]), 8)
        self.assertEqual(sum("decoder-heads" in item["filename"] for item in catalog["archives"]), 1)
        main = [item for item in catalog["archives"] if "main-lora-workspace" in item["filename"]]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(main[0]["seeds"], [20260729, 20260803, 20260804])
        self.assertEqual((main[0]["epoch"], main[0]["step"]), (3, 375))
        for item in catalog["archives"]:
            self.assertNotIn("pretrained", item["filename"])
            self.assertRegex(item["sha256"], r"^[a-f0-9]{64}$")
            self.assertGreater(item["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
