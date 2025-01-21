#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

# Default to development environment
ENVIRONMENT="development"
LOG_LEVEL="debug"

# Parse command-line options
while getopts ":pdh" opt; do
  case ${opt} in
    p )
      ENVIRONMENT="production"
      ;;
    d )
      ENVIRONMENT="development"
      ;;
    h )
      echo "Usage: $0 [-p|-d] [-h]"
      echo "  -p  Run in production environment"
      echo "  -d  Run in development environment (default)"
      echo "  -h  Show this help message"
      exit 0
      ;;
  esac
done

export ENVIRONMENT="${ENVIRONMENT}"
export CUE_LOG="${LOG_LEVEL}"

echo "Running in ${ENVIRONMENT} environment"

# Run with uv command, it runs src.cue.cli._cli_async
# uv run cue -r
uv run python -W ignore::RuntimeWarning -m src.cue.cli._cli_async -r
