#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running format ..."

uv run ruff check --fix .
uv run ruff format .
