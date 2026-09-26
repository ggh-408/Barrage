from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from barrage_rl.artifacts import _replace_with_retry


class ArtifactTests(TestCase):
    def test_replace_retries_transient_permission_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tmp"
            destination = root / "destination.json"
            source.write_text("new", encoding="utf-8")
            destination.write_text("old", encoding="utf-8")

            from barrage_rl import artifacts

            real_replace = artifacts.os.replace
            calls = 0

            def flaky_replace(src: Path, dst: Path) -> None:
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError("temporarily locked")
                real_replace(src, dst)

            with (
                patch.object(artifacts.os, "replace", side_effect=flaky_replace),
                patch.object(artifacts.time, "sleep"),
            ):
                _replace_with_retry(source, destination)

            self.assertEqual(calls, 3)
            self.assertEqual(destination.read_text(encoding="utf-8"), "new")
