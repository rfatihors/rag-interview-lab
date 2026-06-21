import json
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from src.config import settings
from src.providers.factory import build_provider
from src.rag.pipeline import build_pipeline
from src.vectorstores.factory import build_vector_store


EVAL_PATH = Path("eval_sets/rag_eval_set.json")


def build_deepeval_judge_model():
    if settings.deepeval_judge_provider != "ollama":
        raise ValueError(
            "Bu runner şu an yalnızca Ollama judge provider için hazırlanmıştır."
        )

    return OllamaModel(
        model=settings.deepeval_judge_model,
        base_url=settings.deepeval_judge_base_url,
        temperature=0,
        generation_kwargs={
            "num_predict": settings.deepeval_judge_num_predict,
        },
    )


def build_metrics(judge_model):
    return [
        AnswerRelevancyMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
            async_mode=False,
        ),
        FaithfulnessMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
            async_mode=False,
        ),
        ContextualRelevancyMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
            async_mode=False,
        ),
    ]


def run_deepeval():
    provider = build_provider()
    vector_store = build_vector_store()
    pipeline = build_pipeline(provider, vector_store)

    print("Indexing documents before DeepEval...")
    print(pipeline.ingest(recreate=True))

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    judge_model = build_deepeval_judge_model()
    metrics = build_metrics(judge_model)

    all_results = []

    for case in cases:
        output = pipeline.ask(case["question"])

        answer = output["answer"]
        sources = output["sources"]

        test_case = LLMTestCase(
            input=case["question"],
            actual_output=answer,
            retrieval_context=[
                source.get("text", "")
                for source in sources
                if source.get("text")
            ],
        )

        case_result = {
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
            "metrics": [],
        }

        for metric in metrics:
            metric_name = metric.__class__.__name__

            try:
                metric.measure(test_case)

                case_result["metrics"].append(
                    {
                        "metric": metric_name,
                        "score": metric.score,
                        "passed": metric.is_successful(),
                        "reason": metric.reason,
                        "error": None,
                    }
                )

            except Exception as exc:
                case_result["metrics"].append(
                    {
                        "metric": metric_name,
                        "score": None,
                        "passed": False,
                        "reason": None,
                        "error": (
                            f"{type(exc).__name__}: {exc}. "
                            "Muhtemel neden: local judge model DeepEval'in "
                            "beklediği JSON/schema çıktısını üretemedi."
                        ),
                    }
                )

        all_results.append(case_result)

    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_deepeval()
