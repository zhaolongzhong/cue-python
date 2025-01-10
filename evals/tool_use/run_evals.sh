#!/usr/bin/env bash

set -euo pipefail

# Navigate to the root directory of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Build the .whl file
uv build

# Find the generated .whl file dynamically
WHL_FILE=$(ls dist/*.whl | head -n 1)

# Check if the .whl file exists
if [[ -f "$WHL_FILE" ]]; then
    echo "Copying $WHL_FILE to evals/tool_use/assets"
    cp "$WHL_FILE" evals/tool_use/assets
else
    echo "Error: No .whl file found in the dist directory."
    exit 1
fi

# Navigate to the evals/tool_use directory
cd evals/tool_use

# Run the evaluation script
uv run python -m run_evals
