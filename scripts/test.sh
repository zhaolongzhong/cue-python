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

# Load the .env.test file if it exists
load_env ".env.test"

# Initialize flags for each test type
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_EVALUATION=false
RUN_SINGLE_TEST=false
TEST_PATH=""
PYTEST_ARGS=""

# Function to display usage instructions
usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -u    Run unit tests only"
  echo "  -i    Run integration tests only"
  echo "  -a    Run unit, integration and evaluation tests"
  echo "  -e    Run evaluation tests (requires OPENAI_API_KEY)"
  echo "  -t <test_path> Run a single test specified by the test path"
  echo "  -p <pytest_args> Additional pytest arguments (e.g., '-s -v -x')"
  echo "  -h    Show this help message and exit"
  echo ""
  echo "Examples:"
  echo "  $0 -u          # Run unit tests only"
  echo "  $0 -i          # Run integration tests only"
  echo "  $0 -a          # Run both unit and integration tests"
  echo "  $0 -e          # Run evaluation tests"
  echo "  $0 -t tests/evaluation/test_async_client.py::TestClientManager::test_async_client  # Run a single test, e.g. ./scripts/test.sh -t tests/evaluation/test_basic_tool_use.py"
  echo "  $0 -u -e       # Run unit and evaluation tests"
  echo "  $0 -u -p '-s -v -x'  # Run unit tests with debug log, verbose output and exit on first failure"
  exit 1
}

# Parse command-line options
while getopts ":uiaet:p:h" opt; do
  case ${opt} in
    u )
      RUN_UNIT=true
      ;;
    i )
      RUN_INTEGRATION=true
      ;;
    a )
      RUN_UNIT=true
      RUN_INTEGRATION=true
      RUN_EVALUATION=true
      ;;
    e )
      RUN_EVALUATION=true
      ;;
    t )
      RUN_SINGLE_TEST=true
      TEST_PATH="$OPTARG"
      ;;
    p )
      PYTEST_ARGS="$OPTARG"
      ;;
    h )
      usage
      ;;
    \? )
      echo "Invalid Option: -$OPTARG" >&2
      usage
      ;;
    : )
      echo "Invalid Option: -$OPTARG requires an argument" >&2
      usage
      ;;
  esac
done

# Shift off the options and optional --
shift $((OPTIND -1))

# If no options are provided, default to running both unit and integration tests
if ! $RUN_UNIT && ! $RUN_INTEGRATION && ! $RUN_EVALUATION && ! $RUN_SINGLE_TEST; then
  RUN_UNIT=true
  RUN_INTEGRATION=true
fi

# Function to run pytest and handle empty test suites
run_pytest() {
  local test_type="$1"
  local pytest_args="$2"
  local extra_args="${3:-}"
  
  echo "Running $test_type tests ..."
  # Use set +e to prevent script from exiting if pytest returns non-zero
  set +e
  uv run pytest --ignore="backend" $pytest_args $extra_args
  local exit_code=$?
  set -e
  
  # Exit codes:
  # 0 = tests passed
  # 1 = tests failed
  # 5 = no tests collected
  if [ $exit_code -eq 5 ]; then
    echo "No $test_type tests found (this is OK if you haven't created any yet)"
    return 0
  elif [ $exit_code -ne 0 ]; then
    echo "$test_type tests failed with exit code $exit_code"
    exit $exit_code
  fi
}

# Run unit tests if flagged
if $RUN_UNIT; then
  run_pytest "unit" "$PYTEST_ARGS" "-m unit"
fi

# Run integration tests if flagged
if $RUN_INTEGRATION; then
load_env ".env.test"
  run_pytest "integration" "$PYTEST_ARGS" "-m integration"
fi

# Run evaluation tests if flagged
if $RUN_EVALUATION; then
  
  # Load the main .env file to get the API keys
  load_env ".env"
  
  # Check if OPENAI_API_KEY is set
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is not set. Please set it in the .env file or as an environment variable." >&2
    exit 1
  fi
  
  run_pytest "evaluation" "$PYTEST_ARGS" "tests/evaluation"
fi

# Run a single test if flagged
if $RUN_SINGLE_TEST; then
  echo "Running single test: $TEST_PATH ..."
  
  # Check if TEST_PATH contains "evaluation" and load the .env file if necessary
  if [[ "$TEST_PATH" == *"evaluation"* ]]; then
    echo "Loading environment variables for evaluation tests..."
    load_env ".env"
  fi
  
  run_pytest "single" "$PYTEST_ARGS" "$TEST_PATH"
fi

echo "All requested tests have been executed successfully."