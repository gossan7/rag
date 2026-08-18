from rag.errors import RagError
from rag.config.load_config import load_config

# import fire
# from .rag_cli import RagCLI


if __name__ == "__main__":
    try:
        load_config()
    except RagError as e:
        print(e)
    # try:
    # fire.Fire(RagCLI)
    # except Exception as e:
    # print(e)
