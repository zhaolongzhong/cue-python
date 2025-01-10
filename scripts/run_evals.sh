#!/usr/bin/env bash

set -euo pipefail

# Change to the project root directory
cd "$(dirname "$0")/.."

# Function to load environment variables from a file
load_env() {
  local env_file="$1"
  if [ -f "$env_file" ]; then
    # Export variables without echoing them
    set -a
    # Use grep to ignore comments and empty lines
    # shellcheck disable=SC1091
    grep -v '^\s*#' "$env_file" | grep -v '^\s*$' > /tmp/temp_env_file
    source /tmp/temp_env_file
    rm /tmp/temp_env_file
    set +a
  fi
}


echo "Running evals ..."

# Load the main .env file to get the API keys
load_env "src/cue/.env"

# Check if OPENAI_API_KEY is set
if [ -z "${OPENAI_API_KEY:-}" ]; then
echo "Error: OPENAI_API_KEY is not set. Please set it in the .env file or as an environment variable." >&2
exit 1
fi

uv run pytest -s evals

echo "All requested tests have been executed successfully."
