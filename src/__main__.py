import fire

import sys

from .rag import RagCLI
from .rag import RagError
from .rag import Logger


if __name__ == "__main__":
    logger = Logger()
    try:
        fire.Fire(RagCLI)
    except RagError as e:
        logger.close_bars()
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.close_bars()
        logger.error("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}")
