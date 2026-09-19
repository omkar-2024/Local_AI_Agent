from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Keep chunks small enough for retrieval quality while avoiding large LLM context.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
EMBED_BATCH_SIZE = 32
MAX_QUERY_TOP_K = 20
QUERY_RETRIES = 2
COLLECTION_NAME = "company_knowledge"


class VectorStoreService:
    """Persistent local ChromaDB store with local Ollama embeddings.

    ChromaDB stores the document chunks, embeddings and metadata.
    Ollama is used only as the local embedding provider.
    """

    def __init__(self) -> None:
        settings.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(settings.VECTOR_DB_PATH),
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Reuse one HTTP connection instead of creating a new AsyncClient
        # for every chunk/query.
        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_http_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(
                    base_url=self._ollama_base_url(),
                    timeout=httpx.Timeout(60.0, connect=5.0),
                    limits=httpx.Limits(
                        max_connections=8,
                        max_keepalive_connections=4,
                    ),
                )
            return self._http_client

    @staticmethod
    def _ollama_base_url() -> str:
        # EMBEDDING_API_BASE can be configured independently if required.
        base = getattr(
            settings,
            "EMBEDDING_API_BASE",
            "http://localhost:11434",
        )
        return base.rstrip("/").removesuffix("/v1").rstrip("/")

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch in one Ollama request.

        Batching is important during ingestion: the old implementation made
        one HTTP request per chunk, which is unnecessarily slow.
        """
        if not texts:
            return []

        client = await self._get_http_client()
        response = await client.post(
            "/api/embed",
            json={
                "model": settings.EMBEDDING_MODEL,
                "input": texts,
            },
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                "Ollama returned an unexpected embedding response."
            )

        return embeddings

    async def _embed(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed an empty query.")

        last_error: Exception | None = None

        for attempt in range(QUERY_RETRIES):
            try:
                embeddings = await self._embed_many([text])
                return embeddings[0]
            except Exception as exc:
                last_error = exc
                if attempt + 1 < QUERY_RETRIES:
                    await asyncio.sleep(0.15)

        raise RuntimeError(
            f"Failed to create query embedding after {QUERY_RETRIES} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Paragraph-aware character chunking with overlap."""
        text = (text or "").strip()
        if not text:
            return []

        paragraphs = [
            " ".join(p.split())
            for p in text.splitlines()
            if p.strip()
        ]

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) <= CHUNK_SIZE:
                candidate = (
                    f"{current}\n{paragraph}".strip()
                    if current
                    else paragraph
                )

                if len(candidate) <= CHUNK_SIZE:
                    current = candidate
                    continue

            if current:
                chunks.append(current)

                overlap = current[-CHUNK_OVERLAP:]
                current = f"{overlap}\n{paragraph}".strip()

                if len(current) > CHUNK_SIZE:
                    # Hard split an oversized paragraph.
                    start = 0
                    while start < len(current):
                        end = start + CHUNK_SIZE
                        piece = current[start:end].strip()
                        if piece:
                            chunks.append(piece)
                        start = max(
                            end - CHUNK_OVERLAP,
                            start + 1,
                        )
                    current = ""
            else:
                start = 0
                while start < len(paragraph):
                    end = start + CHUNK_SIZE
                    piece = paragraph[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    start = max(
                        end - CHUNK_OVERLAP,
                        start + 1,
                    )

        if current:
            chunks.append(current)

        return chunks or [text[:CHUNK_SIZE]]

    @staticmethod
    def _stable_id(
        source: str,
        chunk_index: int,
        text: str,
    ) -> str:
        digest = hashlib.sha256(
            f"{source}|{chunk_index}|{text}".encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()[:20]
        return f"kb_{digest}"

    async def add_document(
        self,
        text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Replace an existing source and index its chunks."""
        chunks = self._chunk(text)
        if not chunks:
            return 0

        source = str(source).strip() or "unknown"
        metadata = metadata or {}

        ids = [
            self._stable_id(source, i, chunk)
            for i, chunk in enumerate(chunks)
        ]

        metadatas = []
        for i, _ in enumerate(chunks):
            item = {
                "source": source,
                "chunk_index": i,
            }
            item.update(
                {
                    str(k): v
                    for k, v in metadata.items()
                    if v is not None
                    and isinstance(v, (str, int, float, bool))
                }
            )
            metadatas.append(item)

        # Include the trusted source title in the embedding input. The stored
        # document remains unchanged, but retrieval now knows that a chunk came
        # from e.g. "Vision and Mission.pdf".
        embedding_texts = [
            f"Document: {source}\nContent: {chunk}"
            for chunk in chunks
        ]

        all_embeddings: list[list[float]] = []

        # Batch embeddings to dramatically reduce ingestion time.
        for start in range(0, len(embedding_texts), EMBED_BATCH_SIZE):
            batch = embedding_texts[start:start + EMBED_BATCH_SIZE]
            all_embeddings.extend(
                await self._embed_many(batch)
            )

        # Replace only after embedding succeeded, so a failed ingestion does
        # not destroy an already-indexed source.
        await asyncio.to_thread(
            self.collection.delete,
            where={"source": source},
        )

        await asyncio.to_thread(
            self.collection.add,
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=all_embeddings,
        )

        logger.info(
            "Indexed source=%s chunks=%d collection=%s",
            source,
            len(chunks),
            COLLECTION_NAME,
        )
        return len(chunks)

    async def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        source: str,
    ) -> int:
        """Index pre-chunked text with per-chunk metadata such as page."""
        cleaned = [
            c for c in chunks
            if str(c.get("text", "")).strip()
        ]
        if not cleaned:
            return 0

        source = str(source).strip() or "unknown"

        texts = [str(c["text"]).strip() for c in cleaned]
        ids = [
            self._stable_id(source, i, text)
            for i, text in enumerate(texts)
        ]

        metadatas = []
        for i, item in enumerate(cleaned):
            metadata = {
                "source": source,
                "chunk_index": i,
            }
            for key, value in item.items():
                if key == "text" or value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    metadata[str(key)] = value
            metadatas.append(metadata)

        embedding_texts = [
            f"Document: {source}\nContent: {text}"
            for text in texts
        ]

        embeddings: list[list[float]] = []
        for start in range(0, len(embedding_texts), EMBED_BATCH_SIZE):
            embeddings.extend(
                await self._embed_many(
                    embedding_texts[start:start + EMBED_BATCH_SIZE]
                )
            )

        await asyncio.to_thread(
            self.collection.delete,
            where={"source": source},
        )

        await asyncio.to_thread(
            self.collection.add,
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(texts)

    async def search(
        self,
        query: str,
        top_k: int = 4,
        *,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fast semantic retrieval from persistent ChromaDB."""
        query = " ".join((query or "").split())
        if not query:
            return []

        collection_count = await asyncio.to_thread(
            self.collection.count
        )
        if collection_count == 0:
            return []

        query_embedding = await self._embed(query)
        where = {"source": source} if source else None
        requested_k = max(1, min(int(top_k), MAX_QUERY_TOP_K))

        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=min(requested_k, collection_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        output = []
        for i, text in enumerate(documents):
            metadata = (
                metadatas[i]
                if i < len(metadatas)
                else {}
            ) or {}
            if not str(text or "").strip():
                continue

            distance = (
                float(distances[i])
                if i < len(distances)
                else 1.0
            )

            # Chroma's cosine distance is converted to a similarity score.
            # Clamp it to avoid tiny floating-point excursions outside [0, 1].
            score = max(0.0, min(1.0, 1.0 - distance))

            output.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get(
                        "chunk_index",
                        i,
                    ),
                    "text": text,
                    "score": score,
                    "metadata": metadata,
                }
            )

        return output

    async def search_source_candidates(
        self,
        terms: list[str],
        *,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Return chunks whose source filename contains one of the supplied terms.

        This is a lexical safety net for document-specific questions. It avoids
        relying entirely on semantic similarity when the user explicitly names
        a document such as a Vision and Mission or CSR policy file.
        """
        wanted = {
            " ".join(str(term).lower().split())
            for term in terms
            if str(term).strip()
        }
        if not wanted:
            return []

        data = await asyncio.to_thread(
            self.collection.get,
            include=["documents", "metadatas"],
        )

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        candidates: list[dict[str, Any]] = []

        for i, text in enumerate(documents):
            metadata = (
                metadatas[i]
                if i < len(metadatas)
                else {}
            ) or {}
            source_name = str(
                metadata.get("source", "")
            ).lower()

            if not any(term in source_name for term in wanted):
                continue

            if not str(text or "").strip():
                continue

            candidates.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index", i),
                    "text": text,
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "metadata": metadata,
                    "source_match": True,
                }
            )

        return candidates[:max(1, limit)]

    async def search_lexical_candidates(
        self,
        terms: list[str],
        *,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """Find chunks containing query terms as a lexical retrieval fallback."""
        wanted = {
            str(term).strip().lower()
            for term in terms
            if str(term).strip()
        }
        if not wanted:
            return []

        data = await asyncio.to_thread(
            self.collection.get,
            include=["documents", "metadatas"],
        )

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        candidates: list[dict[str, Any]] = []

        for i, text in enumerate(documents):
            clean_text = str(text or "").strip()
            if not clean_text:
                continue

            metadata = (
                metadatas[i] if i < len(metadatas) else {}
            ) or {}
            source_name = str(metadata.get("source", ""))
            haystack = f"{source_name}\n{clean_text}".lower()

            hits = sum(
                1
                for term in wanted
                if re.search(rf"\\b{re.escape(term)}\\b", haystack)
            )
            if hits == 0:
                continue

            candidates.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index", i),
                    "text": clean_text,
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "lexical_score": hits / len(wanted),
                    "metadata": metadata,
                }
            )

        candidates.sort(
            key=lambda item: float(item.get("lexical_score", 0.0)),
            reverse=True,
        )
        return candidates[:max(1, limit)]

    async def expand_neighbors(
        self,
        results: list[dict[str, Any]],
        *,
        max_extra: int = 2,
    ) -> list[dict[str, Any]]:
        """Add adjacent chunks from the same source for local context continuity."""
        if not results or max_extra <= 0:
            return []

        wanted: set[tuple[str, int]] = set()

        for result in results:
            try:
                index = int(result.get("chunk_index", -1))
            except (TypeError, ValueError):
                continue
            if index < 0:
                continue

            source_name = str(result.get("source", ""))
            wanted.add((source_name, index - 1))
            wanted.add((source_name, index + 1))

        wanted = {
            item for item in wanted
            if item[1] >= 0
        }
        if not wanted:
            return []

        data = await asyncio.to_thread(
            self.collection.get,
            include=["documents", "metadatas"],
        )
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        neighbors: list[dict[str, Any]] = []
        for i, text in enumerate(documents):
            metadata = (
                metadatas[i] if i < len(metadatas) else {}
            ) or {}
            source_name = str(metadata.get("source", ""))
            try:
                chunk_index = int(metadata.get("chunk_index", i))
            except (TypeError, ValueError):
                continue

            if (source_name, chunk_index) not in wanted:
                continue

            neighbors.append(
                {
                    "source": source_name or "unknown",
                    "page": metadata.get("page"),
                    "chunk_index": chunk_index,
                    "text": str(text or "").strip(),
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "source_score": 0.0,
                    "metadata": metadata,
                }
            )

        return neighbors[:max_extra]

    async def reset_collection(self) -> None:
        """Delete every indexed knowledge chunk and recreate the collection.

        This is used by the trusted-directory rebuild flow so stale documents
        from an older knowledge base can never survive a full rebuild.
        """
        old_collection = self.collection

        await asyncio.to_thread(
            self.client.delete_collection,
            COLLECTION_NAME,
        )

        self.collection = await asyncio.to_thread(
            self.client.get_or_create_collection,
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Reset knowledge collection=%s",
            COLLECTION_NAME,
        )

    async def count(self) -> int:
        return await asyncio.to_thread(
            self.collection.count
        )

    async def delete_source(self, source: str) -> None:
        await asyncio.to_thread(
            self.collection.delete,
            where={"source": source},
        )

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


vector_store = VectorStoreService()
