from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from src.config import settings
from src.providers.factory import build_provider
from src.rag.pipeline import build_pipeline
from src.vectorstores.factory import build_vector_store


def build_judge_model():
    if settings.deepeval_judge_provider != "ollama":
        raise ValueError(
            "Bu smoke test şu an yalnızca Ollama judge provider için hazırlanmıştır."
        )

    return OllamaModel(
        model=settings.deepeval_judge_model,
        base_url=settings.deepeval_judge_base_url,
        temperature=0,
        generation_kwargs={
            "num_predict": settings.deepeval_judge_num_predict,
        },
    )


def run_smoke():
    judge_model = build_judge_model()

    provider = build_provider()
    vector_store = build_vector_store()
    pipeline = build_pipeline(provider, vector_store)

    print("Indexing documents before DeepEval smoke...")
    print(pipeline.ingest(recreate=True))

    question = "Yapay zeka destekli karar destek sistemi nihai karar verir mi?"
    output = pipeline.ask(question)

    print("RAG answer:")
    print(output["answer"])

    test_case = LLMTestCase(
        input=question,
        actual_output=output["answer"],
        retrieval_context=[
            source["text"]
            for source in output["sources"]
            if source.get("text")
        ],
    )

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=judge_model,
        include_reason=True,
        async_mode=False,
    )

    try:
        metric.measure(test_case)
    except Exception as exc:
        print("DeepEval smoke failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        print(
            "\nNot: Bu genellikle local judge modelin DeepEval'in beklediği "
            "JSON/schema formatını üretememesinden kaynaklanır. "
            "Daha güçlü judge model veya OpenAI/Azure judge kullanılmalıdır."
        )
        return

    print("Score:", metric.score)
    print("Passed:", metric.is_successful())
    print("Reason:", metric.reason)


if __name__ == "__main__":
    run_smoke()
