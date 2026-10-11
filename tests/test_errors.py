import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from src.rag.config import load_config
from src.rag.data_models import Source, load_source
from src.rag.dataset import load_dataset, load_results
from src.rag.errors import IndexingError, RagError, SearchError
from src.rag.logger import Logger

ROOT = Path(__file__).resolve().parent.parent


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()


class SourceTest(TempDirTestCase):
    def source(self, content: str, chunk: str) -> Source:
        path = self.tmp / "file.py"
        path.write_text(content, encoding="utf-8")
        return Source(
            file_path=str(path),
            first_character_index=0,
            last_character_index=len(chunk),
            chunk_hash=sha256(chunk.encode("utf-8")).hexdigest(),
        )

    def test_valid_source(self) -> None:
        source = self.source("hello world", "hello")
        self.assertEqual(load_source(source), "hello")

    def test_missing_file(self) -> None:
        source = self.source("hello world", "hello")
        Path(source.file_path).unlink()
        with self.assertRaises(RagError) as ctx:
            load_source(source)
        self.assertIn("no longer exists", str(ctx.exception))
        self.assertIn(source.file_path, str(ctx.exception))

    def test_modified_content(self) -> None:
        source = self.source("HELLO world", "hello")
        with self.assertRaises(RagError) as ctx:
            load_source(source)
        self.assertIn("chunk hash mismatch", str(ctx.exception))

    def test_truncated_file(self) -> None:
        source = self.source("he", "hello")
        with self.assertRaises(RagError) as ctx:
            load_source(source)
        self.assertIn("shorter than the indexed range", str(ctx.exception))


class DatasetTest(TempDirTestCase):
    def test_missing_dataset(self) -> None:
        with self.assertRaises(RagError) as ctx:
            load_dataset(self.tmp / "nope.json")
        self.assertIn("file not found", str(ctx.exception))

    def test_invalid_json(self) -> None:
        path = self.tmp / "bad.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(RagError) as ctx:
            load_dataset(path)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_invalid_search_results(self) -> None:
        path = self.tmp / "results.json"
        path.write_text(json.dumps({"k": 3}), encoding="utf-8")
        with self.assertRaises(SearchError) as ctx:
            load_results(path)
        self.assertIn(
            "'search_results': required field", str(ctx.exception)
        )


class LoggerTest(unittest.TestCase):
    def test_redirected_output_is_plain_text_on_stderr(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            logger = Logger()
            Logger.last_stream = None
            logger.warning("first line\nsecond line")
            logger.result("path/file.py [0:10]")
            logger.success("done")
        self.assertEqual(
            stderr.getvalue(),
            "WARNING  first line\n         second line\n\n\nSUCCESS  done\n"
        )
        self.assertEqual(stdout.getvalue(), "path/file.py [0:10]\n")


class IndexStateTest(TempDirTestCase):
    def test_interrupted_index_is_reported_and_kept(self) -> None:
        from src.rag.retrieval.backends import BM25Backend

        cwd = Path.cwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, cwd)
        config = load_config("default.yaml", ROOT / "config")
        config.paths.processed = self.tmp
        backend = BM25Backend(config, Logger())

        with self.assertRaises(SearchError) as search_ctx:
            with backend.search_session():
                pass
        self.assertIn("no index found", str(search_ctx.exception))

        backend.index_path.mkdir(parents=True)
        backend.marker.touch()
        with self.assertRaises(IndexingError) as ctx:
            with backend.index_session(2000):
                pass
        message = str(ctx.exception)
        self.assertIn("bm25 index", message)
        self.assertIn(str(backend.index_path), message)
        self.assertIn(".indexing", message)
        self.assertIn("manually", message)
        self.assertTrue(backend.marker.is_file())


class CliTest(TempDirTestCase):
    def run_cli(self, *args: str) -> "subprocess.CompletedProcess[str]":
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run(
            [sys.executable, "-m", "src", *args],
            cwd=self.tmp, env=env, capture_output=True, text=True,
        )

    def test_expected_errors_exit_with_status_1(self) -> None:
        (self.tmp / "config").mkdir()
        result = self.run_cli(
            "search", "--query", "x", "--config_file", "missing.yaml"
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("ERROR", result.stderr)
        self.assertIn("'missing.yaml' not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("\x1b[", result.stderr)

        result = self.run_cli("search", "--query", "x", "--k", "abc")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--k", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
