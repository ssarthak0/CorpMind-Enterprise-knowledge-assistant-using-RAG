from langchain_core.embeddings import Embeddings

from corp_mind.client import embed_query, embed_texts


class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)
