#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Start building ..."

# Verify environment variables
if [ -z "${version_name:-}" ]; then
    echo "Error: version_name environment variable is not set"
    exit 1
fi

if [ -z "${API_URL:-}" ]; then
    echo "Error: API_URL environment variable is not set"
    exit 1
fi

# Update version in all locations
echo "Updating version to: $version_name"

# 1. Update _version.py
cat > src/cue/_version.py << EOF
# src/cue/_version.py
__title__ = "cue"
__version__ = "$version_name"
EOF

# 2. Update pyproject.toml
sed -i.bak "s/version = \"[0-9]*\.[0-9]*\.[0-9]*\"/version = \"$version_name\"/" pyproject.toml
rm -f pyproject.toml.bak

# 3. Update API_URL in config.py
CONFIG_FILE="src/cue/config.py"
sed -i.bak "s|API_URL: str = \"\"|API_URL: str = \"$API_URL\"|" "$CONFIG_FILE"
rm -f "$CONFIG_FILE.bak"

# Print changes for verification
echo "Updated _version.py:"
cat src/cue/_version.py

echo "Updated version in pyproject.toml:"
grep "version = " pyproject.toml

echo "Updated API_URL in config.py"
grep "API_URL: str" "$CONFIG_FILE" | sed 's|\(http[s]*://[^/]*\)/.*|\1/...|'

uv sync
uv build
