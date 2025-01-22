#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

# Function to check if UV is installed
check_uv_installed() {
    if ! command -v uv &> /dev/null; then
        echo "UV is not installed. Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | bash
        
        # Source the environment immediately after installation
        if [ -f "$HOME/.local/bin/env" ]; then
            echo "Sourcing UV environment..."
            source "$HOME/.local/bin/env"
        else
            echo "Error: UV environment file not found at $HOME/.local/bin/env"
            exit 1
        fi
    else
        echo "UV is already installed."
    fi
}

# Check and install UV
check_uv_installed

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

echo "Installing dependencies..."
# Install both main and dev dependencies
uv pip install -e ".[dev]"

echo "Setup complete! Virtual environment is activated."