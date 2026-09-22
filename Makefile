UV 			:= uv
PYTHON 		:= $(UV) run python
SRC 		:= ./src/
ARGS		?=



run:
	$(PYTHON) -m src index $(ARGS)

profile:
	$(PYTHON) -m cProfile -o profile.prof -m src index $(ARGS)
	$(UV) run snakeviz profile.prof

install:
	$(UV) sync

lint:
	$(PYTHON) -m mypy $(SRC)

clean:
	rm -rf .mypy_cache
	rm -rf src/rag.egg-info/
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

dclean:
	rm -rf data/processed/

fclean: clean dclean
	rm -rf .venv
