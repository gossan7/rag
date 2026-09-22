import fire

from .rag.rag_cli import RagCLI
from .rag.errors import RagError


if __name__ == "__main__":
    try:
        fire.Fire(RagCLI)
    except RagError as e:
        print(e)
