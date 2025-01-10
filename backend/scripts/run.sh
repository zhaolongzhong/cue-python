#!/usr/bin/env bash

set -uo pipefail

# Default environment
ENVIRONMENT="development"

# Parse command-line options
while getopts "dpt" opt; do
  case $opt in
    d)
      ENVIRONMENT="development"
      ;;
    p)
      ENVIRONMENT="production"
      ;;
    t)
      ENVIRONMENT="testing"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done


# Kill any process running on port 8000 if it exists
if lsof -t -i tcp:8000 >/dev/null 2>&1; then
    echo "Killing process on port 8000..."
    kill -9 $(lsof -t -i tcp:8000)
else
    echo "No process running on port 8000"
fi

ENVIRONMENT=$ENVIRONMENT uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
