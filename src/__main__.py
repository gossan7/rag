from .rag.errors import RagError
from .rag.config.load_config import load_config
from .rag.indexing.load_files import iter_files
from .rag.indexing.chunk_files import chunk_file

# import fire
# from .rag_cli import RagCLI


if __name__ == "__main__":
    try:
        config = load_config()
        gen = iter_files(config)

        fdata = next(gen)
        fdata = next(gen)
        print(fdata[1])
        chunks = chunk_file(config, fdata, 2000)

        ext, file_path, file_content = fdata

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
                f"chunk n{i}, depth{0}, len{end - start}\n",
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
