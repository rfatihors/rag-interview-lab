import json
from pathlib import Path
from time import perf_counter

from src.providers.factory import build_provider
from src.rag.pipeline import build_pipeline
from src.vectorstores.factory import build_vector_store


EVAL_PATH = Path("eval_sets/rag_eval_set.json")


def contains_any(text: str, expected_items: list[str]) -> bool:
    if not expected_items:
        return True

    lowered = text.lower()
    return any(item.lower() in lowered for item in expected_items)


def contains_none(text: str, forbidden_items: list[str]) -> bool:
    lowered = text.lower()
    return all(item.lower() not in lowered for item in forbidden_items)


def source_hit(sources: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True

    returned_sources = {source.get("source") for source in sources}
    return any(expected in returned_sources for expected in expected_sources)


def run_eval():
    provider = build_provider()
    vector_store = build_vector_store()
    pipeline = build_pipeline(provider, vector_store)

    print("Indexing before evaluation...")
    print(pipeline.ingest(recreate=True))

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    results = []

    for case in cases:
        start = perf_counter()
        error = None

        try:
            output = pipeline.ask(case["question"])
            answer = output["answer"]
            sources = output["sources"]
        except Exception as exc:
            answer = ""
            sources = []
            error = f"{type(exc).__name__}: {exc}"

        latency = perf_counter() - start

        checks = {
            "source_hit": source_hit(sources, case["expected_sources"]),
            "answer_contains": contains_any(answer, case["expected_answer_contains"]),
            "must_not_contain": contains_none(answer, case["must_not_contain"]),
        }

        passed = all(checks.values()) and error is None

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": passed,
                "latency_sec": round(latency, 3),
                "checks": checks,
                "error": error,
                "answer": answer,
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

    print(f"Passed: {passed_count}/{total}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_eval()
