#!/usr/bin/env sh
set -eu

FILES_ROOT_PATH="${FILES_ROOT:-files}"
DATA_STORE_FILE="${DATA_STORE_PATH:-data/store.json}"

mkdir -p "${FILES_ROOT_PATH}/original" "${FILES_ROOT_PATH}/tailored"
mkdir -p "$(dirname "${DATA_STORE_FILE}")"

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
