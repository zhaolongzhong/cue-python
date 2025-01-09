#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make sure we're in the root directory of the project
cd "${SCRIPT_DIR}/../.." || exit 1

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy the pre-commit hook
cp "${SCRIPT_DIR}/pre-commit" .git/hooks/pre-commit

# Make the hook executable
chmod +x .git/hooks/pre-commit

# Verify installation
if [ -x .git/hooks/pre-commit ]; then
    echo -e "${GREEN}✓ Git hooks installed successfully${NC}"
    echo "Pre-commit hook is now active and will:"
    echo "  • Prevent direct commits to protected branches"
    echo "  • Enforce branch naming conventions"
    echo "  • Run lint checks (if available)"
else
    echo -e "${RED}✗ Failed to install git hooks${NC}"
    exit 1
fi
