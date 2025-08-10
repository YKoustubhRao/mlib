#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

poetry run ruff check --fix .
poetry run ruff format .
poetry run ruff check --fix .
pytest
./run_examples.sh
