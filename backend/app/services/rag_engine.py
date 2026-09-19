from __future__ import annotations

import logging
import re
from typing import Any

from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATES = 20
MAX_CHARS_PER_CHUNK = 1800
MAX_CONTEXT_CHARS = 9000
MIN_SCORE = 0.16
LEXICAL_WEIGHT = 0.40
SOURCE_BOOST = 0.20

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "company", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "our", "the",
    "their", "this", "to", "what", "which", "with", "according", "about",
    "does", "do", "me", "tell", "please", "internal", "document",
    "documents", "mrpl", "mentioned", "described", "say", "says",
}

_SOURCE_HINTS = {
    "vision": ["vision"],
    "mission": ["vision", "mission"],
    "vision and mission": ["vision", "mission"],
    "health": ["health", "safety"],
    "safety": ["health", "safety"],
    "health and safety": ["health", "safety"],
    "work permit": ["health", "safety"],
    "fire": ["health", "safety"],
    "corporate social responsibility": ["csr", "social responsibility"],
    "social responsibility": ["csr", "social responsibility"],
    "csr": ["csr"],
    "sustainability": ["csr", "sustainability"],
}


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def _query_terms(query: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalise(query))
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _source_hints(query: str) -> list[str]:
    text = _normalise(query)
    hints: list[str] = []

    for phrase, terms in _SOURCE_HINTS.items():
        if phrase in text:
            for term in terms:
                if term not in hints:
                    hints.append(term)

    return hints


def _lexical_ratio(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0

    haystack = _normalise(text)
    hits = sum(
        1
        for token in query_terms
        if re.search(rf"\b{re.escape(token)}\b", haystack)
    )
    return hits / len(query_terms)


def _source_match(source: str, hints: list[str]) -> float:
    if not hints:
        return 0.0

    source_text = _normalise(source).replace("_", " ").replace("-", " ")

    matched = sum(
        1
        for hint in hints
        if hint in source_text
    )
    return matched / len(hints)


class RAGEngine:
    """Hybrid local RAG layer between the agent and ChromaDB.

    Retrieval combines semantic similarity with exact lexical and source-name
    signals. This is especially important for short questions that explicitly
    name an internal document.
    """

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        query = " ".join((query or "").split())
        if not query:
            return []

        requested_top_k = max(1, min(int(top_k), 10))
        query_terms = _query_terms(query)
        source_hints = _source_hints(query)

        # Retrieve a wider semantic pool. Short internal questions such as
        # "what is our vision?" are often too small for pure vector search.
        candidates = await vector_store.search(
            query,
            top_k=RETRIEVAL_CANDIDATES,
            source=source,
        )

        # Always supplement semantic retrieval when the question contains a
        # document/source hint. Do not wait for semantic search to fail first:
        # a wrong top semantic result is exactly what we are trying to prevent.
        if source_hints and not source:
            try:
                candidates.extend(
                    await vector_store.search_source_candidates(
                        source_hints,
                        limit=60,
                    )
                )
            except Exception:
                logger.exception("Source-name fallback retrieval failed")

        # For short/keyword-heavy queries, add direct lexical candidates too.
        # This catches exact phrases that a small embedding model can rank poorly.
        if query_terms and not source:
            try:
                candidates.extend(
                    await vector_store.search_lexical_candidates(
                        sorted(query_terms),
                        limit=60,
                    )
                )
            except Exception:
                logger.exception("Lexical fallback retrieval failed")

        rescored: list[dict[str, Any]] = []

        for result in candidates:
            text = str(result.get("text", ""))
            source_name = str(result.get("source", ""))

            semantic_score = float(
                result.get("semantic_score", result.get("score", 0.0))
            )

            searchable_text = f"{source_name}\n{text}"
            lexical = _lexical_ratio(query_terms, searchable_text)
            source_score = _source_match(source_name, source_hints)

            # Exact lexical/source evidence gets more weight for short queries.
            # Semantic similarity remains the main signal for natural-language
            # questions with enough content.
            if len(query_terms) <= 3 or source_hints:
                combined_score = (
                    semantic_score * 0.50
                    + lexical * 0.30
                    + source_score * 0.20
                )
            else:
                combined_score = (
                    semantic_score * 0.60
                    + lexical * 0.25
                    + source_score * 0.15
                )

            item = dict(result)
            item["semantic_score"] = semantic_score
            item["lexical_score"] = lexical
            item["source_score"] = source_score
            item["score"] = min(1.0, combined_score)
            rescored.append(item)

        # Do not throw away a clearly source-matched chunk just because its
        # embedding score is low. This is a deliberate safety net for names,
        # titles and very short questions.
        filtered = [
            result
            for result in rescored
            if (
                float(result.get("score", 0.0)) >= MIN_SCORE
                and (
                    float(result.get("semantic_score", 0.0)) >= MIN_SCORE
                    or float(result.get("lexical_score", 0.0)) > 0
                    or float(result.get("source_score", 0.0)) > 0
                )
            )
        ]

        filtered.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                float(item.get("lexical_score", 0.0)),
                float(item.get("semantic_score", 0.0)),
            ),
            reverse=True,
        )

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()

        for result in filtered:
            try:
                chunk_index = int(result.get("chunk_index", -1))
            except (TypeError, ValueError):
                chunk_index = -1

            key = (
                str(result.get("source", "unknown")),
                chunk_index,
                str(result.get("text", ""))[:80],
            )

            if key in seen:
                continue

            seen.add(key)
            deduped.append(result)

            if len(deduped) >= requested_top_k:
                break

        # Add immediate neighboring chunks from the same source. This helps
        # when the answer starts at the end of one chunk and continues in the
        # next one, without asking the LLM to make another retrieval call.
        if deduped:
            try:
                expanded = await vector_store.expand_neighbors(
                    deduped,
                    max_extra=max(0, requested_top_k // 2),
                )
                if expanded:
                    merged = deduped + expanded
                    final: list[dict[str, Any]] = []
                    seen_final: set[tuple[str, int]] = set()

                    for result in merged:
                        key = (
                            str(result.get("source", "unknown")),
                            int(result.get("chunk_index", -1)),
                        )
                        if key in seen_final:
                            continue
                        seen_final.add(key)
                        final.append(result)
                        if len(final) >= requested_top_k:
                            break
                    return final
            except Exception:
                logger.exception("Neighbor expansion failed")

        return deduped

    def format_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """Format already-retrieved chunks without performing another search."""
        if not results:
            return (
                "No sufficiently relevant information was found "
                "in the organization's local knowledge base."
            )

        blocks: list[str] = []
        total_chars = 0

        for i, result in enumerate(results, start=1):
            source = str(result.get("source", "unknown"))
            page = result.get("page")
            location = (
                f"{source}, page {page}"
                if page is not None
                else source
            )

            text = str(result.get("text", "")).strip()
            if len(text) > MAX_CHARS_PER_CHUNK:
                text = text[:MAX_CHARS_PER_CHUNK].rstrip() + "..."

            block = (
                f"[Source {i}: {location} | "
                f"similarity={float(result.get('score', 0.0)):.2f}]\n{text}"
            )

            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - total_chars
                if remaining > 200:
                    blocks.append(block[:remaining].rstrip() + "...")
                break

            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

    async def answer_context(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        results = await self.retrieve(
            query,
            top_k=top_k,
        )
        return self.format_context(results)


rag_engine = RAGEngine()
