# CorpMind — Enterprise Knowledge Assistant (RAG)

A retrieval-augmented generation (RAG) assistant that answers questions about company knowledge using **Google Gemini** via the **OpenAI Python client**.

All LLM calls (chat, structured output, embeddings) go through Gemini's OpenAI-compatible API — no OpenAI, Groq, or LiteLLM required.

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/)
- A [Google AI Studio API key](https://aistudio.google.com/apikey)

## Setup

1. **Install dependencies**

   ```bash
   cd CorpMind-Enterprise-knowledge-assistant-using-RAG
   uv sync
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   CORPMIND_CHAT_MODEL=gemini-2.5-flash
   CORPMIND_EMBEDDING_MODEL=gemini-embedding-001
   CORPMIND_EMBEDDING_DIMENSIONS=768
   ```

## Run

### 1. Ingest the knowledge base (required first time)

```bash
uv run python -m implementation.ingest
```

This reads markdown files from `knowledge-base/`, embeds them with Gemini, and stores vectors in `vector_db/`.

### 2. Launch the chat UI

```bash
uv run python app.py
```

Opens a Gradio chat in your browser.

### 3. (Optional) Run the evaluation dashboard

```bash
uv run python evaluator.py
```

### 4. (Optional) Advanced RAG pipeline

The `pro_implementation/` track uses LLM-based chunking, query rewriting, and reranking:

```bash
uv run python -m pro_implementation.ingest
uv run python -c "from pro_implementation.answer import answer_question; print(answer_question('When was Insurellm founded?')[0])"
```

## Project structure

```text
.
├── app.py                  # Gradio chat UI
├── evaluator.py            # RAG evaluation dashboard
├── corp_mind/              # Shared Gemini client & config
│   ├── config.py
│   ├── client.py
│   └── embeddings.py
├── implementation/         # Basic RAG (used by app.py)
├── pro_implementation/     # Advanced RAG
├── evaluation/             # Test suite (150 questions)
└── knowledge-base/         # Source documents (markdown)
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | — | Google AI Studio API key |
| `CORPMIND_CHAT_MODEL` | No | `gemini-2.5-flash` | Gemini model for chat & structured output |
| `CORPMIND_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Gemini model for embeddings |
| `CORPMIND_EMBEDDING_DIMENSIONS` | No | `768` | Embedding vector size (768 recommended) |

## Notes

- Re-run `uv run python -m implementation.ingest` whenever you change files in `knowledge-base/` or embedding settings.
- Embeddings use 768 dimensions with L2 normalization (required by Google for `gemini-embedding-001` at non-3072 sizes). Delete `vector_db/` and re-ingest if you change embedding dimensions.
- The knowledge base uses sample data for a fictional company called **Insurellm**.
- `vector_db/` and `preprocessed_db/` are generated locally and are gitignored.
