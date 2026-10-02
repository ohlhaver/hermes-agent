"""Exercise installed desktop assets through the real wheel/sdist backend.

The OCI image installs a wheel and removes the checkout. A source-tree import
cannot detect a missing launcher beside the installed runtime module.
"""
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("launcher.sh", "wallpaper.png")


class BotDesktopPackagingTests(unittest.TestCase):
    def test_installed_launcher_survives_wheel_and_sdist_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            source = scratch / "source"
            source.mkdir()
            # Keep the production build metadata and actual desktop package.
            # Other runtime packages/dependencies are unnecessary for building
            # and resolving these files; no provider or desktop is started.
            for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "hermes_constants.py"):
                shutil.copy2(ROOT / name, source / name)
            for name in ("tools", "locales", "optional-mcps"):
                shutil.copytree(ROOT / name, source / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            output = scratch / "artifacts"
            output.mkdir()

            def build(tree, method):
                result = subprocess.run(
                    [sys.executable, "-c", "import setuptools.build_meta as b; "
                     f"b.{method}({str(output)!r})"], cwd=tree,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-3000:])

            build(source, "build_wheel")
            direct_wheel = next(output.glob("*.whl"))
            self.check_installed_assets(direct_wheel, scratch / "direct")
            direct_wheel.unlink()
            build(source, "build_sdist")
            with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
                archive.extractall(scratch / "sdist", filter="data")
            sdist = next((scratch / "sdist").iterdir())
            build(sdist, "build_wheel")
            self.check_installed_assets(next(output.glob("*.whl")), scratch / "from-sdist")

    def check_installed_assets(self, wheel, installed):
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed)
        for name in ASSETS:
            path = installed / "tools" / "bot_desktop" / name
            self.assertTrue(path.is_file(), f"Installed wheel omits desktop asset: {name}")
            self.assertEqual(path.read_bytes(), (ROOT / "tools" / "bot_desktop" / name).read_bytes())
        # Import from the wheel outside the source checkout. This is the path
        # runtime.start() actually passes to bash in the packaged OCI image.
        check = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from tools.bot_desktop.runtime import _LAUNCHER; "
             "assert _LAUNCHER.is_file(); "
             "assert _LAUNCHER.parent == __import__('pathlib').Path(sys.argv[1]) / 'tools/bot_desktop'",
             str(installed)], cwd=installed, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        syntax = subprocess.run(["bash", "-n", str(installed / "tools/bot_desktop/launcher.sh")],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
