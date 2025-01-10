#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Start building ..."

# Set the API URL in config.py
CONFIG_FILE="src/cue/config.py"
if [ -z "${API_URL:-}" ]; then
    echo "Error: API_URL environment variable is not set"
    exit 1
fi

# Update the API_URL in config.py
sed -i.bak "s|API_URL: str = \"\"|API_URL: str = \"$API_URL\"|" "$CONFIG_FILE"
rm -f "$CONFIG_FILE.bak"

# Print the change for verification (without showing the full URL)
echo "Updated API_URL in config.py"
grep "API_URL: str" "$CONFIG_FILE" | sed 's|\(http[s]*://[^/]*\)/.*|\1/...|'

uv sync
uv build
