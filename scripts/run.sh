#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

export ENVIRONMENT="development"
export CUE_LOG="debug"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run with uv command, it runs src.cue.cli._cli_async
# uv run cue -r
uv run python -W ignore::RuntimeWarning -m src.cue.cli._cli_async -r
