import json
from pathlib import Path
from time import perf_counter

from src.config import settings
from src.providers.factory import build_provider
from src.rag.pipeline import build_pipeline
from src.vectorstores.factory import build_vector_store


EVAL_PATH = Path("eval_sets/rag_eval_set.json")


def source_hit_at_k(sources: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True

    returned_sources = {source.get("source") for source in sources}
    return any(expected in returned_sources for expected in expected_sources)


def top1_source_hit(sources: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True

    if not sources:
        return False

    return sources[0].get("source") in expected_sources


def run_retrieval_eval():
    provider = build_provider()
    vector_store = build_vector_store()
    pipeline = build_pipeline(provider, vector_store)

    print("Indexing before retrieval evaluation...")
    print(pipeline.ingest(recreate=True))

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    results = []

    for case in cases:
        start = perf_counter()

        query_embedding = provider.get_embeddings([case["question"]])[0]
        sources = vector_store.search(
            query_embedding=query_embedding,
            top_k=pipeline.top_k,
        )

        latency = perf_counter() - start
        expected_sources = case["expected_sources"]
        best_score = sources[0]["score"] if sources else 0.0

        if expected_sources:
            checks = {
                "source_hit_at_k": source_hit_at_k(sources, expected_sources),
                "top1_source_hit": top1_source_hit(sources, expected_sources),
            }
            # Final pass için source_hit@k yeterli; top1 analitik metrik olarak raporlanır.
            passed = checks["source_hit_at_k"]
        else:
            checks = {
                "no_answer_score_below_threshold": best_score < settings.min_score,
            }
            passed = checks["no_answer_score_below_threshold"]

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "passed": passed,
                "latency_sec": round(latency, 3),
                "checks": checks,
                "min_score": settings.min_score,
                "best_score": round(best_score, 4),
                "sources": [
                    {
                        "source": source.get("source"),
                        "score": round(source.get("score", 0), 4),
                    }
                    for source in sources
                ],
            }
        )

    passed_count = sum(1 for item in results if item["passed"])
    total = len(results)

    print(f"Retrieval Passed: {passed_count}/{total}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_retrieval_eval()
