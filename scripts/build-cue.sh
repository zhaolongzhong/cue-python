#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Start building ..."

uv sync

uv build

