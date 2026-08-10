import yaml

from .indexing import create_chunks


class RagCLI:
    def __init__(self, config_file: str = "config/config.yaml") -> None:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(
                f"An error has occured during config reading / loading.\n{e}"
            )

    def search(self, msg: str = "feur") -> None:
        print(self.config, msg, sep='\n')

    def index(self, max_chunk_size: int = 2000) -> None:
        if not 1 <= max_chunk_size <= 2000:
            raise ValueError(
                "max_chunk_size must be between 1 and 2000"
            )
        create_chunks(self.config, max_chunk_size)
