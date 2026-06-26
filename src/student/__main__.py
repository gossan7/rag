import fire
from .rag_cli import RagCLI


if __name__ == "__main__":
    try:
        fire.Fire(RagCLI)
    except Exception as e:
        print(e)
