#!/usr/bin/env bash
# Linux / macOS helper: creates venv (first time) and starts the app.
set -e
cd "$(dirname "$0")/.."
if [ ! -d venv ]; then
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
else
  source venv/bin/activate
fi
[ -f .env ] || cp .env.example .env
echo "IMPORTANT: edit .env and set ADMIN_PASSWORD before logging in."
streamlit run app.py
