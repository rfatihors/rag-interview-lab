from typing import Any

from src.config import settings
from src.core.interfaces import BaseLLMProvider, BaseVectorStore
from src.loaders.document_loader import load_documents
from src.rag.prompt_builder import DEFAULT_SYSTEM_PROMPT, build_grounded_prompt
from src.splitters.recursive_splitter import RecursiveDocumentSplitter


class RAGPipeline:
    def __init__(
        self,
        provider: BaseLLMProvider,
        vector_store: BaseVectorStore,
        docs_dir: str,
        top_k: int = 5,
        min_score: float = 0.35,
    ):
        self.provider = provider
        self.vector_store = vector_store
        self.docs_dir = docs_dir
        self.top_k = top_k
        self.min_score = min_score
        self.splitter = RecursiveDocumentSplitter()

    def ingest(self, recreate: bool = True) -> dict[str, Any]:
        documents = load_documents(self.docs_dir)
        chunks = self.splitter.split_documents(documents)

        if not chunks:
            return {
                "documents": len(documents),
                "chunks": 0,
                "vector_size": 0,
            }

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.provider.get_embeddings(texts)
        vector_size = len(embeddings[0])

        self.vector_store.create_collection(
            vector_size=vector_size,
            recreate=recreate,
        )
        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
        )

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "vector_size": vector_size,
        }

    def _extractive_fallback(self, contexts: list[dict[str, Any]]) -> str:
        """
        LLM boş cevap dönerse retrieved context'lerden güvenli, kaynaklı fallback üretir.
        Bu yaratıcı cevap değildir; kaynak metinleri kullanıcıya kontrollü şekilde taşır.
        """
        if not contexts:
            return "Dokümanda bu bilgi bulunamadı."

        lines = [
            "LLM yanıtı boş döndüğü için kaynaklardan doğrudan çıkarılan bilgi aşağıdadır:"
        ]

        for item in contexts:
            source = item.get("source", "unknown")
            text = item.get("text", "").strip()

            if not text:
                continue

            lines.append(f"- Kaynak: {source} | Bilgi: {text[:500]}")

        if len(lines) == 1:
            return "Dokümanda bu bilgi bulunamadı."

        return "\n".join(lines)

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_embedding = self.provider.get_embeddings([question])[0]

        contexts = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filters=filters,
        )

        if not contexts:
            return {
                "answer": "Dokümanda bu bilgi bulunamadı.",
                "sources": [],
            }

        best_score = contexts[0]["score"]

        if best_score < self.min_score:
            return {
                "answer": "Dokümanda bu bilgi bulunamadı.",
                "sources": contexts,
            }

        prompt = build_grounded_prompt(question, contexts)
        answer = self.provider.generate_response(
            prompt=prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        if not answer or not answer.strip():
            answer = self._extractive_fallback(contexts)

        return {
            "answer": answer,
            "sources": contexts,
        }


def build_pipeline(
    provider: BaseLLMProvider,
    vector_store: BaseVectorStore,
) -> RAGPipeline:
    return RAGPipeline(
        provider=provider,
        vector_store=vector_store,
        docs_dir=settings.docs_dir,
        top_k=settings.top_k,
        min_score=settings.min_score,
    )
