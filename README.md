# RAG Interview Lab

Production-oriented RAG interview scaffold.

## Features

- FastAPI API layer
- Provider abstraction: Mock / OpenAI / Ollama
- Qdrant vector store
- Document loader: TXT/MD direct, LiteParse optional for PDF/Office/Image
- Recursive chunking
- Grounded answer prompt
- Deterministic evaluation
- DeepEval optional hook

## Quick Start

```bash
cp .env.example .env

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d

python -m src.smoke_test_qdrant
