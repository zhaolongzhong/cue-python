# Backend Guide

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [API Docs](#api-docs)

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- UV package manager
- OpenAI API Key

## Setup

1. **Start Docker Services**

   ```bash
   docker-compose up -d
   ```

   Starts PostgreSQL service. To restart:

   ```bash
   docker-compose down && docker-compose up -d
   ```

2. **Environment Configuration**

   ```bash
   cd backend
   cp .env.example .env
   ```

   Set `OPENAI_API_KEY` for memory embedding.

3. **Database Initialization**

   ```bash
   uv sync
   source .venv/bin/activate
   alembic upgrade head
   ```

4. **Run Server**

   ```bash
   ./scripts/run.sh
   ```

## API Docs

Access OpenAPI UI: [http://localhost:8000/docs](http://localhost:8000/docs)
