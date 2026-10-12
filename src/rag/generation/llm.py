from copy import deepcopy
from typing import Any
import logging

from ..config import Config
from ..logger import Logger
from ..errors import AnswerError, ModelLoadError

THINK_END_MARKERS = ("</think>",)
CONTEXT_ATTRIBUTES = ("max_position_embeddings", "n_positions", "n_ctx")
MAX_PLAUSIBLE_CONTEXT = 10_000_000

logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)


class LLM:
    """Local text-generation model run through Transformers.

    Wraps a Hugging Face ``text-generation`` pipeline, checks each
    prompt against the configured and model token limits, and
    extracts the final answer from the generated text.
    """

    def __init__(self, config: Config, logger: Logger) -> None:
        """Load the generation pipeline and derive its settings.

        The model runs on CUDA when available and on CPU otherwise, and
        may be downloaded from the Hugging Face Hub on first use. The
        generation settings are a copy of the model defaults with
        sampling enabled and the configured temperature and maximum
        number of new tokens.

        Args:
            config: Validated application configuration.
            logger: Logger used to report the model loading.

        Raises:
            ModelLoadError: If the model cannot be downloaded or loaded.
        """
        self.config = config
        self.logger = logger

        from transformers import pipeline, GenerationConfig
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = config.answer.model_name
        if device == "cuda":
            self.logger.info(
                f"Loading generation model '{model_name}' on CUDA..."
            )
        else:
            self.logger.info(
                f"Loading generation model '{model_name}' on CPU "
                "(CUDA unavailable)..."
            )

        try:
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=device,
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise ModelLoadError(
                f"Cannot load generation model '{model_name}': {e}\n"
                "Check 'answer.model_name' in the configuration, your "
                "network connection and the Hugging Face cache."
            ) from e

        self.generation_config: GenerationConfig = deepcopy(
            self.generator.model.generation_config
        )

        self.generation_config.temperature = self.config.answer.temperature
        self.generation_config.max_new_tokens = (
            config.answer.max_output_tokens
        )

        self.generation_config.do_sample = True
        self.generation_config.max_length = None

        self.tokenizer: Any = self.generator.tokenizer
        self.encode_kwargs: dict[str, bool] = {
            "enable_thinking": self.config.answer.thinking_mode
        }
        self.context_limit: int | None = self._get_context_limit()
        self.think_end: str | None = self._get_think_end()

    def _get_context_limit(self) -> int | None:
        """Find the context window of the loaded model.

        Returns:
            The smallest plausible value among the known context
            attributes of the model configuration and the maximum length
            of the tokenizer, or None when none is declared.
        """
        model_config = self.generator.model.config
        limits = [
            getattr(model_config, name, None) for name in CONTEXT_ATTRIBUTES
        ]
        limits.append(getattr(self.tokenizer, "model_max_length", None))
        known = [
            limit for limit in limits
            if isinstance(limit, int) and not isinstance(limit, bool)
            and 0 < limit <= MAX_PLAUSIBLE_CONTEXT
        ]
        return min(known) if known else None

    def _get_think_end(self) -> str | None:
        """Find the token that closes the thinking part of an output.

        Returns:
            The end-of-thinking marker when the tokenizer defines it as
            an added token, otherwise None.
        """
        tokenizer = self.tokenizer
        added_vocab = tokenizer.get_added_vocab()
        for marker in THINK_END_MARKERS:
            if marker not in added_vocab:
                continue
            token = tokenizer.convert_ids_to_tokens(
                tokenizer.convert_tokens_to_ids(marker)
            )
            if isinstance(token, str) and token:
                return token
        return None

    def _check_context(self, messages: list[dict[str, str]]) -> None:
        """Verify that a prompt fits the token budgets before generation.

        The prompt is tokenized with the chat template exactly as it
        will be for generation.

        Args:
            messages: Chat messages of the prompt.

        Raises:
            AnswerError: If the chat template cannot be applied, if the
                prompt exceeds ``answer.max_input_tokens``, if the model
                context window is unknown, or if the prompt plus
                ``answer.max_output_tokens`` exceeds that window.
        """
        model_name = self.config.answer.model_name
        max_input = self.config.answer.max_input_tokens
        max_output = self.config.answer.max_output_tokens

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.encode_kwargs,
            )
        except ValueError as e:
            raise AnswerError(
                f"Cannot build the prompt for model '{model_name}': {e}"
            ) from e
        n_tokens = len(encoded["input_ids"])

        if self.context_limit is None:
            context = "model context limit: unknown"
        else:
            context = f"model context limit: {self.context_limit} tokens"
        details = (
            f"(input: {n_tokens} tokens, 'answer.max_input_tokens': "
            f"{max_input}, 'answer.max_output_tokens': {max_output}, "
            f"{context})"
        )

        if n_tokens > max_input:
            raise AnswerError(
                f"Cannot generate an answer with model '{model_name}': the "
                f"prompt exceeds 'answer.max_input_tokens' {details}.\n"
                "Reduce --k, or increase 'answer.max_input_tokens' in the "
                "configuration."
            )
        if self.context_limit is None:
            raise AnswerError(
                f"Cannot generate an answer with model '{model_name}': its "
                "context window cannot be determined from its configuration "
                "or tokenizer, so the prompt cannot be checked against it "
                f"{details}.\n"
                "Use a model that declares its context size in "
                "'answer.model_name'."
            )
        if n_tokens + max_output > self.context_limit:
            raise AnswerError(
                f"Cannot generate an answer with model '{model_name}': the "
                "prompt and the reserved output need "
                f"{n_tokens + max_output} tokens, more than the model "
                f"context window {details}.\n"
                "Reduce --k, or decrease 'answer.max_output_tokens' or "
                "'answer.max_input_tokens' in the configuration."
            )

    def _get_final_answer(self, text: str) -> str:
        """Extract the final answer from the generated text.

        Everything up to the last end-of-thinking marker is discarded
        when the model defines such a marker.

        Args:
            text: Text generated by the model, without the prompt.

        Returns:
            The stripped final answer.

        Raises:
            AnswerError: If thinking mode is enabled and the marker is
                absent, which means generation stopped while thinking, or
                if the final answer is empty.
        """
        model_name = self.config.answer.model_name
        thinking = self.config.answer.thinking_mode
        hint = (
            "Generation may have stopped before the final answer: "
            "increase 'answer.max_output_tokens' or set "
            "'answer.thinking_mode' to false in the configuration."
        )

        if self.think_end is not None:
            _, marker, final = text.rpartition(self.think_end)
            if marker:
                text = final
            elif thinking:
                raise AnswerError(
                    f"Model '{model_name}' did not finish thinking: no "
                    f"closing '{self.think_end}' marker in its output.\n"
                    f"{hint}"
                )

        answer = text.strip()
        if not answer:
            if self.think_end is not None and thinking:
                raise AnswerError(
                    f"Model '{model_name}' returned an empty answer after "
                    f"thinking.\n{hint}"
                )
            raise AnswerError(
                f"Model '{model_name}' returned an empty answer."
            )
        return answer

    def answer(self, messages: list[dict[str, str]]) -> str:
        """Generate the answer to a chat prompt.

        Args:
            messages: Chat messages of the prompt.

        Returns:
            The final answer, without any thinking content.

        Raises:
            AnswerError: If the prompt does not fit the token budgets,
                if the generation fails, or if no final answer can be
                extracted from the output.
        """
        model_name = self.config.answer.model_name
        self._check_context(messages)
        try:
            result = self.generator(
                messages,
                tokenizer_encode_kwargs=self.encode_kwargs,
                generation_config=self.generation_config,
                return_full_text=False,
            )[0]["generated_text"]
        except RuntimeError as e:
            raise AnswerError(
                f"Generation failed with model '{model_name}': {e}"
            ) from e
        if not isinstance(result, str):
            raise AnswerError(
                f"Model '{model_name}' returned an unexpected output format."
            )

        return self._get_final_answer(result)
