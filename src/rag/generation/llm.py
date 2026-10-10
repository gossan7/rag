
from copy import deepcopy

from ..config import Config
from ..logger import Logger


class LLM:
    def __init__(self, config: Config, logger: Logger) -> None:
        self.config = config
        self.logger = logger

        from transformers import pipeline, GenerationConfig
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            self.logger.info("CUDA available.")
        else:
            self.logger.info("CUDA unavailable, will fall back to cpu")

        self.generator = pipeline(
            "text-generation",
            model=config.answer.model_name,
            device=device,
        )

        self.generation_config: GenerationConfig = deepcopy(
            self.generator.model.generation_config
        )

        self.generation_config.temperature = self.config.answer.temperature
        self.generation_config.max_new_tokens = (
            config.answer.max_output_tokens
        )

        self.generation_config.do_sample = True
        self.generation_config.max_length = None

    def answer(self, messages: list[dict[str, str]]) -> str:
        result = self.generator(
            messages,
            tokenizer_encode_kwargs={
                "enable_thinking": self.config.answer.thinking_mode
            },
            generation_config=self.generation_config,
            return_full_text=False,
        )[0]["generated_text"]

        assert isinstance(result, str)
        return result
