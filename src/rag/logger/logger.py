import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from collections.abc import Iterable


class Logger:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def waring(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def loading_bar(
        self,
        iterable: Iterable,
        name: str | None = None
    ) -> Iterable:
        with logging_redirect_tqdm():
            for x in tqdm(iterable=iterable, desc=name):
                yield x
