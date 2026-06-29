UV 			:= uv
PYTHON 		:= $(UV) run python
SRC 		:= ./src/
ARGS		?=



run:
	$(PYTHON) -m student index $(ARGS)

install:
	$(UV) sync

lint:
	$(PYTHON) -m mypy $(SRC)

clean:
	rm -rf .mypy_cache
	rm -rf src/rag.egg-info/
	rm -rf data/processed
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

fclean: clean
	@rm -rf .venv
