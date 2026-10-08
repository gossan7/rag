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
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(file_name)
    text_config = yaml.safe_dump(raw_config, sort_keys=False)

    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text_config, encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        raise ConfigError(
            f"Could not write the config at {path}:\n"
            f"{e}"
        )
