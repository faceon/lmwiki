import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SOURCE_DIR = Path(os.getenv("SOURCE_DIR", "samples/source"))
WIKI_DIR = Path(os.getenv("WIKI_DIR", "samples/wiki"))
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "lm_studio/google/gemma-4-26b-a4b")
EMBED_API_BASE = os.getenv("EMBED_API_BASE", "http://localhost:1234/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "lm_studio/text-embedding-kure-v1")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
INDEX_FILE = WIKI_DIR / "index.md"
LOG_FILE = WIKI_DIR / ".logs" / "state.json"
VECTORDB_DIR = WIKI_DIR / ".vectordb"

TYPE_DIRS = {
    "concept": WIKI_DIR / "concepts",
    "entity": WIKI_DIR / "entities",
    "analysis": WIKI_DIR / "analyses",
}
