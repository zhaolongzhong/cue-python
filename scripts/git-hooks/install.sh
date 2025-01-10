#!/bin/bash

# Get script directory and move to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.." || exit 1

# Define hooks to install
HOOKS=("pre-commit" "commit-msg")
HOOKS_DIR=".git/hooks"

# Ensure hooks directory exists
mkdir -p "${HOOKS_DIR}"

# Install hooks
for hook in "${HOOKS[@]}"; do
    # Copy hook file
    if cp "${SCRIPT_DIR}/${hook}" "${HOOKS_DIR}/${hook}"; then
        # Set executable permission
        chmod +x "${HOOKS_DIR}/${hook}"
        echo "Installed ${hook} hook"
    else
        echo "Failed to install ${hook} hook"
        exit 1
    fi

    # Verify installation
    if [[ ! -x "${HOOKS_DIR}/${hook}" ]]; then
        echo "Hook ${hook} is not executable"
        exit 1
    fi
done

echo "All git hooks installed successfully"
