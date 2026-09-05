from .rag.errors import RagError
from .rag.config.load_config import load_config
from .rag.indexing.load_files import file_iterator
from .rag.indexing.chunk_files import chunk_file

import random

# import fire
# from .rag_cli import RagCLI


if __name__ == "__main__":
    try:
        config = load_config()
        gen = file_iterator(config)

        for fdata in gen:
            chunks = chunk_file(config, fdata, 200)

            # if random.random() <= 0.01:
            #     break

        ext, file_path, file_content = fdata

        print(file_path)
        # for i, (depth, chunk) in enumerate(chunks):
        #     start = chunk.first_character_index
        #     end = chunk.last_character_index
        #     print(
        #         f"chunk n{i}, depth{depth}, len{end-start}\n",
        #         file_content[start:end],
        #         "\n\n\n\n\n",
        #         sep="\n"
        #     )

        for i, chunk in enumerate(chunks):
            start = chunk.first_character_index
            end = chunk.last_character_index
            print(
                f"chunk n{i}, len{end - start}\n",
                file_content[start:end],
                "\n\n\n\n\n",
                sep="\n"
            )

    except RagError as e:
        print(e)
    # try:
    # fire.Fire(RagCLI)
    # except Exception as e:
    # print(e)
