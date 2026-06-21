"""
Opsiyonel DeepEval örneği.

Kurulum:
pip install -r requirements-optional.txt

Not:
DeepEval bazı metriklerde LLM-as-a-judge kullanır. Bu nedenle environment içinde
uygun API key/model ayarı gerekebilir.
"""


def run_deepeval_sample(question: str, answer: str, contexts: list[str]):
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ]

    return evaluate([test_case], metrics)
