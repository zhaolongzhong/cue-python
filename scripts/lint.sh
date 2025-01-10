#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
pwd

echo "Running lint ..."

uv run ruff check .
