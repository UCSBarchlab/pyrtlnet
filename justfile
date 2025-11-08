# pyrtlnet uses `just` instead of `make` because:
# * `make` is not installed by default on Windows.
# * `uv` can install `just` on all supported platforms from PyPI.
presubmit: tests docs

tests:
        # Run `pytest` to run all unit tests.
        uv run pytest

        # Run `ruff format` to check that code is formatted properly.
        #
        # If this fails, try running
        #
        # $ uv run ruff format
        #
        # to automatically reformat any changed code.
        uv run ruff format --diff

        # Run `ruff check` to check for lint errors.
        #
        # If this fails, try running
        #
        # $ uv run ruff check --fix
        #
        # to automatically apply fixes, when possible.
        uv run ruff check

docs:
        # Run `sphinx-build` to generate documentation.
        #
        # Output: docs/_build/html/index.html
        uv run sphinx-build -M html docs/ docs/_build
