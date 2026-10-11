from pydantic import ValidationError

from pathlib import Path
import yaml

from ..errors import ConfigError, format_validation_error
from .config import Config


def load_config(
    file_name: str = "default.yaml",
    config_dir: Path = Path("config/")
) -> Config:
    """Load and validate a YAML configuration file.

    The file is searched recursively under ``config_dir`` and the
    first match is used.

    Args:
        file_name: Name of the configuration file to look for.
        config_dir: Directory searched for the file.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the directory or the file is missing, if the
            name is not a valid search pattern, if the file cannot be
            read or is not valid YAML, if its top level is not a
            mapping, or if it fails validation.
    """
    if not config_dir.is_dir():
        raise ConfigError(
            f"Cannot load configuration '{file_name}': "
            f"directory '{config_dir}' does not exist."
        )

    try:
        config_file = next(config_dir.rglob(file_name))
    except StopIteration as e:
        raise ConfigError(
            f"Configuration file '{file_name}' not found in '{config_dir}'."
        ) from e
    except (ValueError, NotImplementedError) as e:
        raise ConfigError(
            f"Invalid configuration file name '{file_name}': expected a "
            f"file name located in '{config_dir}'."
        ) from e

    if not config_file.is_file():
        raise ConfigError(
            f"Configuration '{config_file}' is not a regular file."
        )

    try:
        raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(
            f"Cannot read configuration file '{config_file}': {e}"
        ) from e
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Configuration file '{config_file}' is not valid YAML:\n{e}"
        ) from e

    if not isinstance(raw_config, dict):
        raise ConfigError(
            f"Invalid configuration in '{config_file}': expected a mapping "
            "of sections (paths, indexing, retrieval, evaluation, answer)."
        )

    try:
        config = Config.model_validate(raw_config)
    except ValidationError as e:
        raise ConfigError(
            f"Invalid configuration in '{config_file}':\n"
            f"{format_validation_error(e)}"
        ) from e

    return config
