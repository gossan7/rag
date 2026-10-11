from rich.console import Console, RenderableType
from rich.text import Text
from tqdm import tqdm

from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, TypeVar
import sys

T = TypeVar("T")

LABEL_WIDTH = 9


class Logger:
    """Console reporter shared by the CLI and the HTTP API.

    Diagnostics and progress bars go to standard error, results go
    to standard output. The registry of open progress bars and the
    last stream written to are shared by every instance.

    Attributes:
        bars: Progress bars currently open, by object identifier.
        last_stream: Name of the stream that received the last
            output, used to separate diagnostics from results.
    """

    bars: ClassVar[dict[int, "tqdm[Any]"]] = {}
    last_stream: ClassVar[str | None] = None

    def __init__(self) -> None:
        """Create the standard error and standard output consoles."""
        self.console = Console(stderr=True, highlight=False)
        self.stdout = Console(highlight=False)

    def _separate(self, stream: str) -> None:
        """Print a blank line when the output moves to another stream.

        Args:
            stream: Name of the stream about to be written to.
        """
        if Logger.last_stream not in (None, stream):
            self.console.print()
        Logger.last_stream = stream

    def _log(self, label: str, style: str, msg: str) -> None:
        """Print a labelled message on standard error.

        Continuation lines are aligned under the message, and open
        progress bars are redrawn after the message.

        Args:
            label: Severity label shown before the message.
            style: Rich style applied to the label.
            msg: Message, possibly spanning several lines.
        """
        text = Text()
        text.append(label.ljust(LABEL_WIDTH), style=style)
        text.append(msg.replace("\n", "\n" + " " * LABEL_WIDTH))
        with tqdm.external_write_mode(file=sys.stderr):
            self._separate("stderr")
            self.console.print(text, soft_wrap=True)

    def info(self, msg: str) -> None:
        """Print an informational message on standard error.

        Args:
            msg: Message to print.
        """
        self._log("INFO", "cyan", msg)

    def success(self, msg: str) -> None:
        """Print a success message on standard error.

        Args:
            msg: Message to print.
        """
        self._log("SUCCESS", "bold green", msg)

    def warning(self, msg: str) -> None:
        """Print a warning message on standard error.

        Args:
            msg: Message to print.
        """
        self._log("WARNING", "bold yellow", msg)

    def error(self, msg: str) -> None:
        """Print an error message on standard error.

        Args:
            msg: Message to print.
        """
        self._log("ERROR", "bold red", msg)

    def result(self, content: RenderableType) -> None:
        """Print a command result on standard output.

        Args:
            content: Plain text, printed without markup interpretation,
                or any Rich renderable such as Markdown.
        """
        if isinstance(content, str):
            content = Text(content)
        with tqdm.external_write_mode(file=sys.stderr):
            self._separate("stdout")
            self.stdout.print(content, soft_wrap=isinstance(content, Text))

    @staticmethod
    def merge_labels(entries: list[tuple[str, str]]) -> list[str]:
        """Group identical messages emitted under different labels.

        Args:
            entries: Pairs made of a label and a message.

        Returns:
            One ``labels: message`` line per distinct message, with the
            labels joined by commas, in order of first appearance.
        """
        merged: dict[str, list[str]] = {}
        for label, msg in entries:
            merged.setdefault(msg, []).append(label)
        return [
            f"{', '.join(labels)}: {msg}" for msg, labels in merged.items()
        ]

    @staticmethod
    def count(n: int, noun: str) -> str:
        """Format a quantity with a regular English plural.

        Args:
            n: Quantity.
            noun: Singular noun.

        Returns:
            The quantity followed by the noun, with an ``s`` unless the
            quantity is exactly one.
        """
        return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    @staticmethod
    def duration(seconds: float) -> str:
        """Format a duration for display.

        Args:
            seconds: Duration in seconds.

        Returns:
            Seconds with one decimal below a minute, otherwise whole
            minutes and seconds.
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rest = divmod(int(seconds), 60)
        return f"{minutes}m {rest:02d}s"

    def close_bars(self) -> None:
        """Close every open progress bar and empty the registry."""
        for bar in list(Logger.bars.values()):
            bar.close()
        Logger.bars.clear()

    def loading_bar(
        self,
        name: str,
        iterable: Iterable[T],
        total: int | None = None,
    ) -> Iterator[T]:
        """Iterate over items while displaying a tqdm progress bar.

        The bar is written to standard error, is disabled when that
        stream is not a terminal, and is registered so that
        ``close_bars`` can close it when a command is aborted.

        Args:
            name: Description shown next to the bar.
            iterable: Items to iterate over.
            total: Number of items, when the iterable has no length.

        Yields:
            The items of the iterable, in order.
        """
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
