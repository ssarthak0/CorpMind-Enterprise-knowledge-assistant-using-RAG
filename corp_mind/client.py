import math

from openai import OpenAI

from corp_mind.config import (
    CHAT_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GEMINI_BASE_URL,
    GOOGLE_API_KEY,
)

_client: OpenAI | None = None


def get_gemini_client() -> OpenAI:
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise ValueError("Set GOOGLE_API_KEY in .env (see .env.example)")
        _client = OpenAI(api_key=GOOGLE_API_KEY, base_url=GEMINI_BASE_URL)
    return _client


def chat_completion(messages: list[dict], *, temperature: float = 0) -> str:
    client = get_gemini_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def structured_completion(messages: list[dict], response_format: type):
    client = get_gemini_client()
    response = client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=messages,
        response_format=response_format,
    )
    return response.choices[0].message.parsed


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


_EMBED_BATCH_SIZE = 100


def _create_embeddings(texts: list[str]) -> list[list[float]]:
    client = get_gemini_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        vectors.extend(item.embedding for item in response.data)

    if EMBEDDING_DIMENSIONS != 3072:
        vectors = [_l2_normalize(vector) for vector in vectors]

    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _create_embeddings(texts)


def embed_query(text: str) -> list[float]:
    return _create_embeddings([text])[0]
