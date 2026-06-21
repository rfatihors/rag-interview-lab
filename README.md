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



## Evaluation Notes



This repo separates evaluation into three layers:



1. Retrieval evaluation:

   - Fast and deterministic.

   - Checks source-hit@k and no-answer threshold behavior.



2. Deterministic answer evaluation:

   - Checks expected answer substrings, forbidden substrings and fallback behavior.



3. DeepEval LLM-as-a-judge:

   - Measures answer relevancy, faithfulness and contextual relevancy.

   - Requires a judge model.

   - Local Ollama judge models may fail structured JSON output; in that case use a stronger local model or OpenAI/Azure judge.



Commands:



```bash

python -m src.evaluation.retrieval_eval

python -m src.evaluation.deterministic_eval

python -m src.evaluation.deepeval_smoke

python -m src.evaluation.deepeval_runner



