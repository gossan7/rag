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

    def index(self, max_chunk_length: int = 2000) -> None:
        chunks = create_chunks(self.config, max_chunk_length)
        # for ext in chunks:
        #     for chunk in chunks[ext]:
        #         print(chunk)
