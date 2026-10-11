.DEFAULT_GOAL := help

UV      := uv
PYTHON  := $(UV) run python
MODULE  := src
ARGS    ?=

.PHONY: help install run debug index search search_dataset answer answer_dataset evaluate \
        profile lint lint-strict clean dclean fclean re

help:
	@printf '%s\n' \
		'Usage: make <target> [ARGS="..."]' \
		'' \
		'Mandatory targets:' \
		'  help            Show this help' \
		'  install         Install/sync dependencies with uv' \
		'  run             Run the CLI (pass arguments with ARGS=...)' \
		'  debug           Run the CLI under pdb' \
		'  clean           Remove Python/tool caches' \
		'  lint            Run the required flake8 + mypy checks' \
		'' \
		'RAG targets:' \
		'  index           Build/update the indexes' \
		'  search          Search one query' \
		'  search_dataset  Search a dataset' \
		'  answer          Answer one query' \
		'  answer_dataset  Answer a dataset from search results' \
		'  evaluate        Evaluate recall@k' \
		'' \
		'Development targets:' \
		'  profile         Profile indexing with cProfile + snakeviz' \
		'  lint-strict     Run flake8 + mypy --strict' \
		'  dclean          Remove generated indexes/output' \
		'  fclean          clean + dclean + remove .venv' \
		'  re              Recreate the environment' \
		'' \
		'Examples:' \
		'  make' \
		'  make index ARGS="--max_chunk_size=2000"' \
		'  make search ARGS="--query=\"what is LoRA?\" --k=5"'

install:
	$(UV) sync

run:
	$(PYTHON) -m $(MODULE) $(ARGS)

debug:
	$(PYTHON) -m pdb -m $(MODULE) $(ARGS)

index:
	$(PYTHON) -m $(MODULE) index $(ARGS)

search:
	$(PYTHON) -m $(MODULE) search $(ARGS)

search_dataset:
	$(PYTHON) -m $(MODULE) search_dataset $(ARGS)

answer:
	$(PYTHON) -m $(MODULE) answer $(ARGS)

answer_dataset:
	$(PYTHON) -m $(MODULE) answer_dataset $(ARGS)

evaluate:
	$(PYTHON) -m $(MODULE) evaluate $(ARGS)

profile:
	$(PYTHON) -m cProfile -o profile.prof -m $(MODULE) index $(ARGS)
	$(UV) run snakeviz profile.prof

lint:
	$(UV) run flake8 src/
	$(UV) run mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV) run flake8 src/
	$(UV) run mypy src/ --strict

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache profile.prof
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

dclean:
	find data -maxdepth 1 -mindepth 1 ! -name raw ! -name datasets -exec rm -rf {} +

fclean: dclean
	rm -rf .venv

re: fclean install
