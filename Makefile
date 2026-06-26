UV 			:= uv
PYTHON 		:= $(UV) run python
ARGS		?=



run:
	$(PYTHON) -m student index $(ARGS)

install:
	$(UV) sync

clean:
	@rm -rf .mypy_cache src/rag.egg-info/
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

fclean: clean
	@rm -rf .venv
