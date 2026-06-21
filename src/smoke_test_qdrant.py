from src.config import settings
from src.providers.factory import build_provider
from src.rag.pipeline import build_pipeline
from src.vectorstores.factory import build_vector_store


provider = build_provider()
vector_store = build_vector_store()
pipeline = build_pipeline(provider, vector_store)

print("Ingesting documents...")
ingest_result = pipeline.ingest(recreate=True)
print(ingest_result)

question = "Yapay zeka destekli karar destek sistemi nihai karar verir mi?"
print(f"Question: {question}")

answer = pipeline.ask(question)
print(answer)
