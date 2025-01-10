# Tool Use Evaluations

This guide explains how to build and run the agent client evaluation. Follow the steps below to build, copy, and run the evaluations either in a container or locally.

### 1. Build and Copy Agent Client to Assets Folder

Run the following commands to build the package and copy it to the assets folder:

```bash
cd evals/tool_use
./run_evals.sh
```

> The `run_evals.sh` script performs the following:
>
> - Builds the `.whl` file
> - Copies the built `.whl` file to the assets folder
> - Executes the evaluation process

Alternatively, you can run these commands manually:

```bash
uv build
cp dist/*.whl evals/tool_use/assets
```

### 2. Run Evaluations

You can run the evaluations in different environments:

#### Run in Container

```bash
cd evals/tool_use
python run_evals.py
```

#### Run Locally

Activate your virtual environment and run the evaluation script:

```bash
source .venv/bin/activate
cd evals/tool_use
python run_evals_locally.py
```
