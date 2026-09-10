import yaml

from pathlib import Path

from .config import Config


def dump_config(
    config: Config,
    file_name: str,
    config_dir: Path
) -> None:
    raw_config = config.model_dump(mode="json")
    config_dir.mkdir(parents=True, exist_ok=True)
    text_config = yaml.safe_dump(raw_config, sort_keys=False)

    with open(file_name, mode="w", encoding="utf-8") as f:
        f.write(text_config)
