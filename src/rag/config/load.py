from pathlib import Path

from pydantic import ValidationError
import yaml

from ..errors import ConfigError
from .config import Config


CONFIG_DIR = Path("config/")


def load_config(file_path: str = "default.yaml") -> Config:
    try:
        config_file = next(CONFIG_DIR.rglob(file_path))
    except StopIteration as e:
        raise ConfigError(
            f"File not found in {CONFIG_DIR}: '{file_path}'"
        ) from e

    if not config_file.exists():
        raise ConfigError(f"File not found in {CONFIG_DIR}: '{file_path}'")
    if not config_file.is_file():
        raise ConfigError(f"File not found in {CONFIG_DIR}: '{file_path}'")

    try:
        raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ConfigError(
            f"Could not open '{file_path}': {e}"
        ) from e

    try:
        config = Config.model_validate(raw_config)
    except ValidationError as e:
        error_details = "\n"+"\n".join([
            err["msg"] + "\nWHERE: " + ":".join(
                map(str, err["loc"])
            ) for err in e.errors()
        ])
        raise ConfigError(
            f"Config file '{file_path}' malformed:{error_details}"
        ) from e

    return config


if __name__ == "__main__":
    load_config()
