UV = uv



install:
	$(UV) sync

clean:
	@rm -rf .mypy_cache
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

fclean: clean
	@rm -rf .venv
