# Git Hooks for Cue Development

This directory contains Git hooks for enforcing iOS development workflow standards.

## Installation

Run the installation script:

```bash
./scripts/git-hooks/install.sh
```

## Features

The pre-commit hook enforces:

1. Protected Branch Rules

   - Prevents direct commits to main/master/release/production/develop
   - Shows helpful error messages with instructions
   - Provides emergency override option (--no-verify)

2. Branch Naming Convention

   - feat/feature-name (for new features)
   - feature/feature-name (for new features)
   - bugfix/bug-name (for bug fixes)
   - docs/change-name (for documentation)
   - refactor/name (for code refactoring)
   - style/change-name (for styling changes)
   - test/suite-name (for testing changes)
   - chore/task-name (for maintenance tasks)

## Manual Override

In case of emergency, you can bypass the pre-commit hook using:

```bash
git commit --no-verify
```
