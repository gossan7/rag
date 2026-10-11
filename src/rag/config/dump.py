import yaml

from pathlib import Path

from ..errors import ConfigError
from .config import Config


def dump_config(
    config: Config,
    file_name: str,
    directory: Path
) -> None:
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
