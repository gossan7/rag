import json
import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import transformers

from src.rag.config import Config, load_config
from src.rag.data_models import SearchResults, Source
from src.rag.errors import AnswerError
from src.rag.generation import Answerer
from src.rag.generation.llm import LLM
from src.rag.logger import Logger

ROOT = Path(__file__).resolve().parent.parent
THINK_END_ID = 7
UNSET = int(1e30)


class FakeTokenizer:
    def __init__(self, model_max_length: Any, thinking: bool) -> None:
        self.model_max_length = model_max_length
        self.thinking = thinking
        self.calls: list[dict[str, Any]] = []

    def get_added_vocab(self) -> dict[str, int]:
        if self.thinking:
            return {"<think>": THINK_END_ID - 1, "</think>": THINK_END_ID}
        return {"<|im_end|>": 1}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.get_added_vocab().get(token)

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        for token, known_id in self.get_added_vocab().items():
            if known_id == token_id:
                return token
        return None

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, list[int]]:
        self.calls.append(kwargs)
        n_tokens = sum(len(m["content"].split()) for m in messages)
        return {"input_ids": [0] * n_tokens}


class FakePipeline:
    def __init__(
        self,
        output: str,
        max_position_embeddings: Any = 100,
        model_max_length: Any = UNSET,
        thinking: bool = True,
    ) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []
        self.tokenizer = FakeTokenizer(model_max_length, thinking)
        model_config = SimpleNamespace()
        if max_position_embeddings is not None:
            model_config.max_position_embeddings = max_position_embeddings
        self.model = SimpleNamespace(
            config=model_config, generation_config=SimpleNamespace()
        )

    def __call__(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> list[dict[str, str]]:
        self.calls.append(kwargs)
        return [{"generated_text": self.output}]


def default_config() -> Config:
    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        return load_config("default.yaml", ROOT / "config")
    finally:
        os.chdir(cwd)


def messages(n_words: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": " ".join(["word"] * n_words)}]


class LLMTestCase(unittest.TestCase):
    def make_llm(
        self,
        output: str = "final answer",
        thinking_mode: bool = False,
        max_input_tokens: int = 50,
        max_output_tokens: int = 30,
        **pipeline_kwargs: Any,
    ) -> tuple[LLM, FakePipeline]:
        config = default_config()
        config.answer.thinking_mode = thinking_mode
        config.answer.max_input_tokens = max_input_tokens
        config.answer.max_output_tokens = max_output_tokens
        fake = FakePipeline(output, **pipeline_kwargs)
        with mock.patch.dict(
            vars(transformers), {"pipeline": mock.Mock(return_value=fake)}
        ), mock.patch.object(Logger, "info"):
            llm = LLM(config, Logger())
        self.assertIs(llm.generator, fake)
        return llm, fake


class ContextCheckTest(LLMTestCase):
    def test_prompt_below_limits_is_generated(self) -> None:
        llm, fake = self.make_llm()
        self.assertEqual(llm.answer(messages(50)), "final answer")
        self.assertEqual(len(fake.calls), 1)

    def test_counts_tokens_like_the_generation_call(self) -> None:
        llm, fake = self.make_llm(thinking_mode=True, output="a</think>b")
        llm.answer(messages(5))
        count_kwargs = fake.tokenizer.calls[0]
        self.assertTrue(count_kwargs["tokenize"])
        self.assertTrue(count_kwargs["add_generation_prompt"])
        self.assertIs(count_kwargs["enable_thinking"], True)
        self.assertEqual(
            fake.calls[0]["tokenizer_encode_kwargs"],
            {"enable_thinking": True},
        )

    def test_prompt_above_max_input_tokens(self) -> None:
        llm, fake = self.make_llm()
        with self.assertRaises(AnswerError) as ctx:
            llm.answer(messages(51))
        message = str(ctx.exception)
        self.assertIn("exceeds 'answer.max_input_tokens'", message)
        self.assertIn("Qwen/Qwen3-0.6B", message)
        self.assertIn("input: 51 tokens", message)
        self.assertIn("'answer.max_input_tokens': 50", message)
        self.assertIn("'answer.max_output_tokens': 30", message)
        self.assertIn("model context limit: 100 tokens", message)
        self.assertIn("--k", message)
        self.assertEqual(fake.calls, [])

    def test_prompt_and_output_above_model_context(self) -> None:
        llm, fake = self.make_llm(max_input_tokens=90)
        self.assertEqual(llm.answer(messages(70)), "final answer")
        with self.assertRaises(AnswerError) as ctx:
            llm.answer(messages(71))
        message = str(ctx.exception)
        self.assertIn("need 101 tokens", message)
        self.assertIn("model context limit: 100 tokens", message)
        self.assertIn("'answer.max_output_tokens'", message)
        self.assertEqual(len(fake.calls), 1)

    def test_context_limit_sources(self) -> None:
        cases: list[tuple[dict[str, Any], int | None]] = [
            ({"max_position_embeddings": 100, "model_max_length": 60}, 60),
            ({"max_position_embeddings": 100, "model_max_length": UNSET}, 100),
            ({"max_position_embeddings": None, "model_max_length": 80}, 80),
            ({"max_position_embeddings": 0, "model_max_length": None}, None),
            ({"max_position_embeddings": None, "model_max_length": UNSET},
             None),
        ]
        for kwargs, expected in cases:
            llm, _ = self.make_llm(**kwargs)
            self.assertEqual(llm.context_limit, expected, kwargs)

    def test_unknown_model_context(self) -> None:
        llm, fake = self.make_llm(
            max_position_embeddings=None, model_max_length=UNSET
        )
        with self.assertRaises(AnswerError) as ctx:
            llm.answer(messages(5))
        self.assertIn("cannot be determined", str(ctx.exception))
        self.assertIn("model context limit: unknown", str(ctx.exception))
        self.assertEqual(fake.calls, [])


class ThinkingTest(LLMTestCase):
    def test_marker_comes_from_the_tokenizer(self) -> None:
        llm, _ = self.make_llm()
        self.assertEqual(llm.think_end, "</think>")
        llm, _ = self.make_llm(thinking=False)
        self.assertIsNone(llm.think_end)

    def test_complete_thinking_returns_only_the_answer(self) -> None:
        llm, _ = self.make_llm(
            "<think>\nstep </think> again\n</think>\n\n The answer. \n",
            thinking_mode=True,
        )
        self.assertEqual(llm.answer(messages(5)), "The answer.")

    def test_truncated_thinking_is_rejected(self) -> None:
        llm, _ = self.make_llm(
            "<think>\nstill reasoning about", thinking_mode=True
        )
        with self.assertRaises(AnswerError) as ctx:
            llm.answer(messages(5))
        message = str(ctx.exception)
        self.assertIn("did not finish thinking", message)
        self.assertIn("'answer.max_output_tokens'", message)
        self.assertIn("'answer.thinking_mode'", message)

    def test_non_thinking_answer_without_marker(self) -> None:
        llm, _ = self.make_llm("\n\nPlain answer.\n", thinking_mode=False)
        self.assertEqual(llm.answer(messages(5)), "Plain answer.")

    def test_empty_final_answer_is_rejected(self) -> None:
        for output, thinking_mode in (
            ("<think>\nreasoning\n</think>\n\n", True),
            ("  \n", False),
        ):
            llm, _ = self.make_llm(output, thinking_mode=thinking_mode)
            with self.assertRaises(AnswerError) as ctx:
                llm.answer(messages(5))
            self.assertIn("empty answer", str(ctx.exception))

    def test_model_without_thinking_marker(self) -> None:
        llm, _ = self.make_llm(
            "Use </think> literally.", thinking_mode=True, thinking=False
        )
        self.assertEqual(llm.answer(messages(5)), "Use </think> literally.")


class AnswerCacheTest(unittest.TestCase):
    def test_cache_hit_skips_the_model_and_depends_on_the_limit(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source_file = root / "file.py"
        source_file.write_text("hello world", encoding="utf-8")
        query = SearchResults(
            question_id="q1",
            question="what?",
            retrieved_sources=[Source(
                file_path=str(source_file),
                first_character_index=0,
                last_character_index=5,
                chunk_hash=sha256(b"hello").hexdigest(),
            )],
        )

        config = default_config()
        config.paths.processed = root / "processed"
        self.assertIn(
            '"max_input_tokens": 20000',
            json.dumps(config.answer.model_dump(mode="json")),
        )

        answerer = Answerer(config, Logger())
        self.addCleanup(answerer.answer_cache.close)
        model = mock.Mock()
        model.answer.return_value = "generated"
        answerer.__dict__["model"] = model

        self.assertEqual(answerer.answer(query).answer, "generated")
        self.assertFalse(answerer.last_from_cache)
        self.assertEqual(answerer.answer(query).answer, "generated")
        self.assertTrue(answerer.last_from_cache)
        self.assertEqual(model.answer.call_count, 1)

        config.answer.max_input_tokens = 10000
        answerer.answer(query)
        self.assertFalse(answerer.last_from_cache)
        self.assertEqual(model.answer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
