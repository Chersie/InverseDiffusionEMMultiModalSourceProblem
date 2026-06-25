#!/usr/bin/env bash
# Start a local MLflow tracking server with SQLite backend and ./mlartifacts root.
# Backed by R3 in research/framework-rebuild/manifest.md (mlflow 3.12.0 server defaults).
set -euo pipefail

HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"
BACKEND="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
ARTIFACTS="${MLFLOW_ARTIFACT_ROOT:-./mlartifacts}"

mkdir -p "$(dirname "${BACKEND#sqlite:///}")"
mkdir -p "${ARTIFACTS}"

exec uv run mlflow server \
    --backend-store-uri "${BACKEND}" \
    --default-artifact-root "${ARTIFACTS}" \
    --host "${HOST}" \
    --port "${PORT}"
