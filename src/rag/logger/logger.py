import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from collections.abc import Iterable


class Logger:
    def __init__(self) -> None:
        self.logger = logging.getLogger("rag")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s: %(message)s")
        )
        self.logger.addHandler(handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def loading_bar(
        self,
        name: str,
        iterable: Iterable,
        total: int | None = None,
    ) -> Iterable:
        with logging_redirect_tqdm(loggers=[self.logger]):
            yield from tqdm(
                iterable,
                desc=name,
                total=total,
            )
