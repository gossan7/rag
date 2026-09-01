from .rag.errors import RagError
from .rag.config.load_config import load_config
from .rag.indexing.load_files import iter_files

# import fire
# from .rag_cli import RagCLI


if __name__ == "__main__":
    try:
        config = load_config()
        gen = iter_files(config)

        ext, first_path, first_content = next(gen)
        print(ext, first_path, first_content[:min(len(first_content), 100)], sep="\n")
    except RagError as e:
        print(e)
    # try:
    # fire.Fire(RagCLI)
    # except Exception as e:
    # print(e)
