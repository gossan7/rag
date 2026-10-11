import yaml

from pathlib import Path

from ..errors import ConfigError
from .config import Config


def dump_config(
    config: Config,
    file_name: str,
    directory: Path
) -> None:
    """Write a configuration to a YAML file.

    The content is first written to a sibling ``.tmp`` file which
    then replaces the destination, so a failed write does not leave
    a truncated configuration behind. The directory is created when
    missing.

    Args:
        config: Configuration to serialize.
        file_name: Name of the destination file.
        directory: Directory receiving the file.

    Raises:
        ConfigError: If the directory or the file cannot be written.
    """
    raw_config = config.model_dump(mode="json")
    path = directory / Path(file_name)
    text_config = yaml.safe_dump(raw_config, sort_keys=False)

    try:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text_config, encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        raise ConfigError(
            f"Cannot write configuration file '{path}': {e}"
        ) from e
