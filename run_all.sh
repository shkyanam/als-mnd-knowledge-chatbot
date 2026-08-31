#!/usr/bin/env bash

set -euo pipefail

rebuild=false
serve=false
for argument in "$@"; do
    case "$argument" in
        --rebuild) rebuild=true ;;
        --serve) serve=true ;;
        *)
            echo "Usage: bash run_all.sh [--rebuild] [--serve]" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and add NEBIUS_API_KEY." >&2
    exit 1
fi

echo "==> Syncing dependencies"
uv --cache-dir .uv-cache sync

echo "==> Ensuring the 100-article PMC corpus"
uv --cache-dir .uv-cache run python mnd_data_extract.py

if [[ "$rebuild" == true || ! -f data/chroma_db/chroma.sqlite3 ]]; then
    echo "==> Building the semantic Chroma index"
    uv --cache-dir .uv-cache run python rag_app.py index --strategy semantic
else
    echo "==> Using the existing Chroma index (use --rebuild to rebuild it)"
fi

echo "==> Running the retrieval benchmark"
uv --cache-dir .uv-cache run python evaluate_retrieval.py

if [[ "$serve" == true ]]; then
    echo "==> Starting Streamlit"
    exec uv --cache-dir .uv-cache run streamlit run streamlit_app.py
fi

echo "All validation steps passed. Use --serve to start the dashboard."
