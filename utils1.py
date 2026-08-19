from html.parser import HTMLParser
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent

# Module-level singletons — loaded once, reused across all calls.
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

QDRANT_PATH = str(_ROOT / "qdrant_db")
COLLECTION_SCRAPED = "case_pdf"
COLLECTION_USERCASE = "Usercase_pdf"
FOLDER_PATH = str(_ROOT / "case_files")
CHUNK_SIZE = 1000
TOP_K = 10
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

_qdrant_client = QdrantClient(path=QDRANT_PATH)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(p.strip() for p in self._parts if p.strip())


def strip_html(raw: str) -> str:
    """Strip HTML tags and return plain text. Safe to call on plain-text input."""
    extractor = _TextExtractor()
    extractor.feed(raw)
    return extractor.get_text() or raw


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text on sentence boundaries, packing sentences up to `size` chars.
    The last sentence of each chunk is repeated at the start of the next (one-sentence overlap)
    so context is not lost at chunk boundaries.
    Falls back to the full text as a single chunk if no sentence boundaries are found.
    """
    sentences = _SENT_SPLIT.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > size and current:
            chunks.append(" ".join(current))
            current = current[-1:]          # one-sentence overlap
            current_len = len(current[0])
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def loadFile(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Create the named collection if it does not already exist."""
    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info("Created collection: %s", name)


def fetch_from_qdrant(query: str) -> str:
    """Return the top-K most relevant chunks from the scraped case-law collection."""
    _ensure_collection(_qdrant_client, COLLECTION_SCRAPED)

    vector = _embed_model.encode(query).tolist()
    hits = _qdrant_client.query_points(
        collection_name=COLLECTION_SCRAPED,
        query=vector,
        limit=TOP_K,
        with_payload=True,
    ).points
    parts = []
    for hit in hits:
        header = f"[ID: {hit.id}] [Source: {hit.payload.get('source_url', '')}]"
        parts.append(f"{header}\n{hit.payload.get('chunk_text', '')}")
    return "\n\n---\n\n".join(parts)


def fetch_from_qdrant2(query: str) -> str:
    """Return the top-K most relevant chunks from the user's uploaded case files."""
    _ensure_collection(_qdrant_client, COLLECTION_USERCASE)

    # Populate from case_files folder on first run (collection is empty).
    count = _qdrant_client.count(collection_name=COLLECTION_USERCASE, exact=False)
    if count.count == 0:
        folder = Path(FOLDER_PATH)
        if not folder.exists():
            logger.warning("case_files folder not found at %s — skipping population", FOLDER_PATH)
        else:
            for filename in os.listdir(FOLDER_PATH):
                if not (filename.endswith(".txt") or filename.endswith(".html")):
                    continue
                path = folder / filename
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError as e:
                    logger.warning("Skipping '%s': %s", filename, e)
                    continue

                content = strip_html(content)
                chunks = chunk_text(content)
                vectors = _embed_model.encode(chunks, convert_to_numpy=True)

                records = [
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}#{i}")),
                        vector=vectors[i],
                        payload={
                            "chunk_text": chunks[i],
                            "source_file": filename,
                            "chunk_index": i,
                        },
                    )
                    for i in range(len(chunks))
                ]
                _qdrant_client.upsert(collection_name=COLLECTION_USERCASE, points=records)
                logger.info("Inserted %d chunks from '%s'", len(records), filename)

    vector = _embed_model.encode(query).tolist()
    hits = _qdrant_client.query_points(
        collection_name=COLLECTION_USERCASE,
        query=vector,
        limit=TOP_K,
        with_payload=True,
    ).points
    parts = []
    for hit in hits:
        header = f"[ID: {hit.id}] [Source: {hit.payload.get('source_file', '')}]"
        parts.append(f"{header}\n{hit.payload.get('chunk_text', '')}")
    return "\n\n---\n\n".join(parts)