#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

# Function to check if UV is installed
check_uv_installed() {
    if ! command -v uv &> /dev/null; then
        echo "UV is not installed. Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | bash

        # Add UV to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
            echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bash_profile
            export PATH="$HOME/.cargo/bin:$PATH"
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
