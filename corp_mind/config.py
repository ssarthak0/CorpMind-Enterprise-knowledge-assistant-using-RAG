import os

from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = os.getenv("CORPMIND_CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("CORPMIND_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("CORPMIND_EMBEDDING_DIMENSIONS", "768"))
