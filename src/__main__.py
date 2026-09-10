from .rag.errors import RagError
from .rag.config.load import load_config
from .rag.retrieval.index import file_generator, index

import random

# import fire
# from .rag_cli import RagCLI


if __name__ == "__main__":
    try:
        config = load_config()
        index(config, 2000)

    except RagError as e:
        print(e)
    # try:
    # fire.Fire(RagCLI)
    # except Exception as e:
    # print(e)
