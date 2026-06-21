from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveDocumentSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: list[dict]) -> list[dict]:
        chunks = []

        for document in documents:
            source = document["source"]
            metadata = document.get("metadata", {}) or {}
            parts = self.splitter.split_text(document["text"])

            for idx, part in enumerate(parts):
                chunks.append(
                    {
                        "id": f"{source}-{idx}",
                        "source": source,
                        "text": part,
                        "metadata": {
                            **metadata,
                            "chunk_id": idx,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    }
                )

        return chunks
