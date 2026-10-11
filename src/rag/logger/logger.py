from rich.console import Console, RenderableType
from rich.text import Text
from tqdm import tqdm

from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, TypeVar
import sys

T = TypeVar("T")

LABEL_WIDTH = 9


class Logger:
    bars: ClassVar[dict[int, "tqdm[Any]"]] = {}
    last_stream: ClassVar[str | None] = None

    def __init__(self) -> None:
        self.console = Console(stderr=True, highlight=False)
        self.stdout = Console(highlight=False)

    def _separate(self, stream: str) -> None:
        if Logger.last_stream not in (None, stream):
            self.console.print()
        Logger.last_stream = stream

    def _log(self, label: str, style: str, msg: str) -> None:
        text = Text()
        text.append(label.ljust(LABEL_WIDTH), style=style)
        text.append(msg.replace("\n", "\n" + " " * LABEL_WIDTH))
        with tqdm.external_write_mode(file=sys.stderr):
            self._separate("stderr")
            self.console.print(text, soft_wrap=True)

    def info(self, msg: str) -> None:
        self._log("INFO", "cyan", msg)

    def success(self, msg: str) -> None:
        self._log("SUCCESS", "bold green", msg)

    def warning(self, msg: str) -> None:
        self._log("WARNING", "bold yellow", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", "bold red", msg)

    def result(self, content: RenderableType) -> None:
        if isinstance(content, str):
            content = Text(content)
        with tqdm.external_write_mode(file=sys.stderr):
            self._separate("stdout")
            self.stdout.print(content, soft_wrap=isinstance(content, Text))

    @staticmethod
    def merge_labels(entries: list[tuple[str, str]]) -> list[str]:
        merged: dict[str, list[str]] = {}
        for label, msg in entries:
            merged.setdefault(msg, []).append(label)
        return [
            f"{', '.join(labels)}: {msg}" for msg, labels in merged.items()
        ]

    @staticmethod
    def count(n: int, noun: str) -> str:
        return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    @staticmethod
    def duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rest = divmod(int(seconds), 60)
        return f"{minutes}m {rest:02d}s"

    def close_bars(self) -> None:
        for bar in list(Logger.bars.values()):
            bar.close()
        Logger.bars.clear()

    def loading_bar(
        self,
        name: str,
        iterable: Iterable[T],
        total: int | None = None,
    ) -> Iterator[T]:
        with self.console.capture() as capture:
            self.console.print(
                Text("INFO".ljust(LABEL_WIDTH), style="cyan"), end=""
            )
        self._separate("stderr")
        bar = tqdm(
            iterable,
            desc=capture.get() + name,
            total=total,
            disable=None,
            dynamic_ncols=True,
        )
        Logger.bars[id(bar)] = bar
        try:
            yield from bar
        finally:
            bar.close()
            Logger.bars.pop(id(bar), None)
