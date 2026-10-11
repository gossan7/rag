import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.rag.config import load_config
from src.rag.errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent


class ConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.config_dir = self.tmp / "config"
        self.config_dir.mkdir()

    def raw_config(self) -> dict[str, Any]:
        raw: dict[str, Any] = yaml.safe_load(
            (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
        )
        raw["paths"] = {
            "raw": str(self.tmp / "data" / "raw"),
            "processed": str(self.tmp / "data" / "processed"),
            "datasets": str(self.tmp / "data" / "datasets"),
            "output": str(self.tmp / "data" / "output"),
        }
        return raw

    def load(self, raw: Any) -> None:
        path = self.config_dir / "test.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        load_config("test.yaml", self.config_dir)

    def assert_rejected(self, raw: Any, *expected: str) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(raw)
        message = str(ctx.exception)
        self.assertNotIn("Traceback", message)
        self.assertNotIn("pydantic", message)
        for text in expected:
            self.assertIn(text, message)


class ValidConfigTest(ConfigTestCase):
    def test_default_config_loads(self) -> None:
        cwd = Path.cwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, cwd)
        config = load_config("default.yaml", ROOT / "config")
        self.assertTrue(config.retrieval.backends)

    def test_missing_directories_and_shared_parent_are_accepted(self) -> None:
        self.assertFalse((self.tmp / "data").exists())
        self.load(self.raw_config())

    def test_nested_shared_paths_are_accepted(self) -> None:
        raw = self.raw_config()
        raw["paths"]["raw"] = str(self.tmp / "data" / "processed" / "raw")
        raw["paths"]["output"] = str(self.tmp / "data" / "datasets" / "out")
        self.load(raw)


class FieldValidationTest(ConfigTestCase):
    def test_unknown_field(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["wieght"] = 1
        self.assert_rejected(
            raw, "'retrieval.backends.bm25.wieght': unknown field"
        )

    def test_unknown_section(self) -> None:
        raw = self.raw_config()
        raw["extra"] = {}
        self.assert_rejected(raw, "'extra': unknown field")

    def test_missing_field(self) -> None:
        raw = self.raw_config()
        del raw["answer"]["model_name"]
        self.assert_rejected(raw, "'answer.model_name': required field")

    def test_out_of_range_values(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["tfidf"]["weight"] = 3
        raw["answer"]["temperature"] = 0
        raw["answer"]["max_output_tokens"] = 1
        raw["evaluation"]["recall"] = [0, 5]
        raw["evaluation"]["min_iou"] = 2
        self.assert_rejected(
            raw,
            "'retrieval.backends.tfidf.weight'",
            "'answer.temperature'",
            "'answer.max_output_tokens'",
            "'evaluation.recall.0'",
            "'evaluation.min_iou'",
        )

    def test_max_input_tokens_must_be_positive(self) -> None:
        for value in (0, -1):
            raw = self.raw_config()
            raw["answer"]["max_input_tokens"] = value
            self.assert_rejected(raw, "'answer.max_input_tokens'")

        raw = self.raw_config()
        del raw["answer"]["max_input_tokens"]
        self.assert_rejected(raw, "'answer.max_input_tokens': required")

    def test_empty_recall(self) -> None:
        raw = self.raw_config()
        raw["evaluation"]["recall"] = []
        self.assert_rejected(raw, "'evaluation.recall'")

    def test_unknown_backend_name(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["faiss"] = {
            "enabled": True, "weight": 1.0, "path": "faiss"
        }
        self.assert_rejected(raw, "'retrieval.backends.faiss'", "'bm25'")

    def test_unknown_file_extension(self) -> None:
        raw = self.raw_config()
        raw["indexing"]["file_types"][".rst"] = {
            "chunking_method": "window", "enabled": True
        }
        self.assert_rejected(raw, "'indexing.file_types..rst'", "'.md'")

    def test_unknown_chunking_method(self) -> None:
        raw = self.raw_config()
        raw["indexing"]["file_types"][".md"]["chunking_method"] = "magic"
        self.assert_rejected(
            raw, "'indexing.file_types..md.chunking_method'"
        )

    def test_missing_file_type(self) -> None:
        raw = self.raw_config()
        del raw["indexing"]["file_types"][".txt"]
        self.assert_rejected(raw, "no entry for: .txt")

    def test_ast_requires_python(self) -> None:
        raw = self.raw_config()
        raw["indexing"]["file_types"][".md"]["chunking_method"] = "ast"
        self.assert_rejected(raw, "'ast' can only be used with .py")

    def test_no_backend_enabled(self) -> None:
        raw = self.raw_config()
        for backend in raw["retrieval"]["backends"].values():
            backend["enabled"] = False
        self.assert_rejected(raw, "At least one retrieval backend")

    def test_exclude_must_be_a_directory_name(self) -> None:
        raw = self.raw_config()
        raw["indexing"]["excludes"] = ["docs/build"]
        self.assert_rejected(raw, "'indexing.excludes'", "'docs/build'")

    def test_blank_backend_path(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["path"] = "  "
        self.assert_rejected(raw, "'retrieval.backends.bm25.path'")

    def test_path_is_a_file(self) -> None:
        raw = self.raw_config()
        file_path = self.tmp / "file.txt"
        file_path.write_text("x", encoding="utf-8")
        raw["paths"]["raw"] = str(file_path)
        self.assert_rejected(raw, "'paths.raw'", "not a directory")

    def test_not_a_mapping(self) -> None:
        self.assert_rejected(["paths"], "expected a mapping")

    def test_invalid_yaml(self) -> None:
        (self.config_dir / "bad.yaml").write_text("a: [1", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_config("bad.yaml", self.config_dir)
        self.assertIn("not valid YAML", str(ctx.exception))

    def test_missing_file(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config("nope.yaml", self.config_dir)
        self.assertIn("'nope.yaml' not found", str(ctx.exception))


class PathConflictTest(ConfigTestCase):
    def test_duplicate_backend_paths(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["path"] = "tfidf"
        self.assert_rejected(
            raw,
            "'retrieval.backends.bm25.path' and "
            "'retrieval.backends.tfidf.path' resolve to the same directory",
            str(self.tmp / "data" / "processed" / "tfidf"),
        )

    def test_normalized_alias(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["path"] = "../processed/tfidf"
        self.assert_rejected(raw, "resolve to the same directory")

    def test_absolute_alias(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["path"] = str(
            self.tmp / "data" / "processed" / "minilm"
        )
        self.assert_rejected(
            raw,
            "'retrieval.backends.bm25.path' and "
            "'retrieval.backends.minilm.path'",
        )

    def test_symlink_alias(self) -> None:
        processed = self.tmp / "data" / "processed"
        (processed / "tfidf").mkdir(parents=True)
        (processed / "alias").symlink_to(
            processed / "tfidf", target_is_directory=True
        )
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["path"] = "alias"
        self.assert_rejected(raw, "resolve to the same directory")

    def test_disabled_backend_still_conflicts(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["bm25"]["enabled"] = False
        raw["retrieval"]["backends"]["bm25"]["path"] = "tfidf"
        self.assert_rejected(raw, "resolve to the same directory")

    def test_answer_cache_shares_backend_directory(self) -> None:
        raw = self.raw_config()
        raw["answer"]["path"] = "minilm"
        self.assert_rejected(
            raw, "'retrieval.backends.minilm.path' and 'answer.path'"
        )

    def test_answer_cache_inside_backend_directory(self) -> None:
        raw = self.raw_config()
        raw["answer"]["path"] = "bm25/data"
        self.assert_rejected(
            raw, "'answer.path'", "is inside 'retrieval.backends.bm25.path'"
        )

    def test_backend_inside_other_backend(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["tfidf"]["path"] = "bm25/search_cache"
        self.assert_rejected(
            raw,
            "'retrieval.backends.tfidf.path'",
            "is inside 'retrieval.backends.bm25.path'",
        )

    def test_backend_is_processed_root(self) -> None:
        raw = self.raw_config()
        raw["retrieval"]["backends"]["tfidf"]["path"] = "."
        self.assert_rejected(
            raw,
            "'retrieval.backends.tfidf.path' resolves to 'paths.processed'",
        )

    def test_indexes_inside_corpus(self) -> None:
        raw = self.raw_config()
        raw["paths"]["processed"] = str(self.tmp / "data" / "raw" / "idx")
        self.assert_rejected(raw, "is inside 'paths.raw'")

    def test_corpus_inside_index(self) -> None:
        raw = self.raw_config()
        raw["paths"]["raw"] = str(
            self.tmp / "data" / "processed" / "bm25" / "corpus"
        )
        self.assert_rejected(
            raw, "'paths.raw'", "is inside 'retrieval.backends.bm25.path'"
        )

    def test_datasets_inside_index(self) -> None:
        raw = self.raw_config()
        raw["paths"]["datasets"] = str(
            self.tmp / "data" / "processed" / "tfidf" / "datasets"
        )
        self.assert_rejected(
            raw,
            "'paths.datasets'",
            "is inside 'retrieval.backends.tfidf.path'",
        )

    def test_output_is_answer_cache(self) -> None:
        raw = self.raw_config()
        raw["paths"]["output"] = str(
            self.tmp / "data" / "processed" / "generation"
        )
        self.assert_rejected(raw, "'answer.path' and 'paths.output'")

    def test_shared_paths_must_differ(self) -> None:
        raw = self.raw_config()
        raw["paths"]["output"] = str(
            self.tmp / "data" / "other" / ".." / "datasets"
        )
        self.assert_rejected(
            raw, "'datasets' and 'output' resolve to the same directory"
        )


if __name__ == "__main__":
    unittest.main()
