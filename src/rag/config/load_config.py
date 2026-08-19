from pathlib import Path
import yaml

from ..errors import ConfigError
from .config import Config
from .verify_config import verify_config_schema
from .config_facotry import ConfigFactory


CONFIG_DIR = "config/"


def load_config(file_path: str = "default.yaml") -> Config:
    config_dir = Path(CONFIG_DIR)
    config_file = next(config_dir.rglob(file_path))

    if not config_file.exists():
        raise ConfigError(f"File not found in {CONFIG_DIR}: \'{file_path}\'")
    if not config_file.is_file():
        raise ConfigError(f"File not found in {CONFIG_DIR}: \'{file_path}\'")

    try:
        raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except BaseException as e:
        raise ConfigError(f"Could not open \'{file_path}\': {e}")

    try:
        verify_config_schema(raw_config=raw_config)
    except ConfigError as e:
        raise ConfigError(f"Config file \'{file_path}\' malformed:\n{e}")

    try:
        config = ConfigFactory.create(raw_config=raw_config)
    except ConfigError as e:
        raise ConfigError(f"Config file \'{file_path}\' maflormed:\n{e}")

    return config


if __name__ == "__main__":
    load_config()
