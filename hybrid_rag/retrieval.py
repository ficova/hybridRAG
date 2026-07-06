"""Hybrid retrieval: semantic + lexical branches fused with RRF."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import RetrievalConfig
from .db import InventoryDatabase
from .embeddings import EmbeddingError, OllamaEmbedder, SemanticIndex

logger = logging.getLogger(__name__)

SEMANTIC = "semantic"
LEXICAL = "lexical"
HYBRID = "hybrid"
VALID_MODES: tuple[str, ...] = (SEMANTIC, LEXICAL, HYBRID)

_STOPWORDS: frozenset[str] = frozenset(
    """
    a about all also and any are been but can could did does for from had has
    have how its list many may more most not our out show some than that the
    their them then there these they this those was were what when where which
    who whose why will with would you your files file documents document
    contain contains containing find search give tell
    """.split()
)

_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_PUID_RE = re.compile(r"\b(?:x-)?fmt/\d+\b", re.IGNORECASE)
_SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")
_MIME_RE = re.compile(
    r"\b(?:application|text|image|audio|video|multipart|message|font|model)"
    r"/[\w.+-]+\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?")
_MAX_KEYWORDS = 8


@dataclass
class QueryTerms:
    """Lexical search inputs extracted from a natural-language question."""

    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)

    @property
    def like_terms(self) -> list[str]:
        """Phrases first (more specific), then single keywords."""
        return [*self.phrases, *self.keywords]

    @property
    def is_empty(self) -> bool:
        """True when nothing usable was extracted."""
        return not (self.keywords or self.phrases or self.filters)


@dataclass
class RetrievedDocument:
    """A fused retrieval result enriched with inventory metadata."""

    inventory_id: int
    fused_score: float
    sources: list[str]
    semantic_rank: int | None
    lexical_rank: int | None
    metadata: dict[str, Any]
    snippet: str


def extract_query_terms(question: str) -> QueryTerms:
    """Extract keywords, quoted phrases and exact-match filters from a query.

    Recognized filters: PRONOM PUIDs (formatRegistryKey), SHA-256 digests
    (messageDigest) and MIME types (formatMIMEType).

    Args:
        question: Natural-language question.

    Returns:
        The extracted lexical search inputs.
    """
    terms = QueryTerms()
    remainder = question

    for match in _QUOTED_RE.finditer(question):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            terms.phrases.append(phrase)
    remainder = _QUOTED_RE.sub(" ", remainder)

    sha_match = _SHA256_RE.search(remainder)
    if sha_match:
        terms.filters["messageDigest"] = sha_match.group(0).lower()
        remainder = _SHA256_RE.sub(" ", remainder)

    puid_match = _PUID_RE.search(remainder)
    if puid_match:
        terms.filters["formatRegistryKey"] = puid_match.group(0).lower()
        remainder = _PUID_RE.sub(" ", remainder)

    mime_match = _MIME_RE.search(remainder)
    if mime_match:
        terms.filters["formatMIMEType"] = mime_match.group(0).lower()
        remainder = _MIME_RE.sub(" ", remainder)

    seen: set[str] = set(phrase.lower() for phrase in terms.phrases)
    for token in _TOKEN_RE.findall(remainder):
        word = token.lower()
        if len(word) < 3 or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.keywords.append(word)
        if len(terms.keywords) >= _MAX_KEYWORDS:
            break

    logger.debug(
        "Query terms: keywords=%s phrases=%s filters=%s",
        terms.keywords,
        terms.phrases,
        terms.filters,
    )
    return terms


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked id lists with Reciprocal Rank Fusion.

    Each document scores ``sum(1 / (k + rank))`` over the rankings it appears
    in, with rank starting at 1.

    Args:
        rankings: Ranking name -> ordered list of document ids (best first).
        k: RRF constant dampening the impact of high ranks (default 60).

    Returns:
        List of (document_id, fused_score) sorted by descending score.
    """
    scores: dict[int, float] = {}
    for ranked_ids in rankings.values():
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _build_snippet(text: str | None, terms: Sequence[str], max_chars: int) -> str:
    """Return an excerpt of text centered on the first matching term.

    Args:
        text: Full extracted text (may be None or empty).
        terms: Lexical terms to center the excerpt on, if found.
        max_chars: Maximum excerpt length.

    Returns:
        A whitespace-normalized excerpt, possibly empty.
    """
    if not text:
        return ""
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    start = 0
    for term in terms:
        position = lowered.find(term.lower())
        if position >= 0:
            start = max(0, position - max_chars // 4)
            break
    snippet = normalized[start : start + max_chars]
    if start > 0:
        snippet = "..." + snippet
    if start + max_chars < len(normalized):
        snippet = snippet + "..."
    return snippet


class HybridRetriever:
    """Runs the semantic and lexical branches and fuses them with RRF."""

    def __init__(
        self,
        db: InventoryDatabase,
        index: SemanticIndex | None,
        embedder: OllamaEmbedder | None,
        config: RetrievalConfig,
    ) -> None:
        """Initialize the retriever.

        Args:
            db: Open inventory database.
            index: Loaded semantic index, or None when no embeddings exist.
            embedder: Query embedder, or None when Ollama is unavailable.
            config: Retrieval settings.
        """
        self._db = db
        self._index = index
        self._embedder = embedder
        self._config = config

    @property
    def semantic_available(self) -> bool:
        """True when the semantic branch can run."""
        return self._index is not None and self._embedder is not None

    def _semantic_ranking(self, question: str) -> list[int]:
        """Return semantically ranked ids, or an empty list on failure."""
        if self._index is None or self._embedder is None:
            return []
        try:
            query_vector = self._embedder.embed_query(question)
        except EmbeddingError:
            logger.exception("Query embedding failed; semantic branch skipped.")
            return []
        results = self._index.search(
            query_vector, self._config.semantic_candidates
        )
        return [doc_id for doc_id, _score in results]

    def _lexical_ranking(self, terms: QueryTerms) -> list[int]:
        """Return lexically ranked ids, or an empty list when inapplicable."""
        if terms.is_empty:
            return []
        results = self._db.lexical_search(
            terms.like_terms, terms.filters, self._config.lexical_candidates
        )
        return [doc_id for doc_id, _score in results]

    def retrieve(self, question: str, mode: str) -> list[RetrievedDocument]:
        """Retrieve the top documents for a question.

        Args:
            question: Natural-language question.
            mode: One of ``semantic``, ``lexical`` or ``hybrid``.

        Returns:
            Fused, metadata-enriched documents (may be empty).

        Raises:
            ValueError: If mode is not a valid retrieval mode.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid retrieval mode: {mode!r}")

        if mode != LEXICAL and not self.semantic_available:
            logger.warning(
                "Semantic search unavailable (no embeddings or no Ollama); "
                "falling back to lexical-only retrieval."
            )
            mode = LEXICAL

        terms = extract_query_terms(question)
        rankings: dict[str, list[int]] = {}
        if mode in (SEMANTIC, HYBRID):
            semantic_ids = self._semantic_ranking(question)
            if semantic_ids:
                rankings[SEMANTIC] = semantic_ids
        if mode in (LEXICAL, HYBRID):
            lexical_ids = self._lexical_ranking(terms)
            if lexical_ids:
                rankings[LEXICAL] = lexical_ids

        if not rankings:
            logger.info("No results from any retrieval branch.")
            return []

        fused = reciprocal_rank_fusion(rankings, k=self._config.rrf_k)
        top = fused[: self._config.top_k]
        metadata_by_id = self._db.fetch_documents([doc_id for doc_id, _ in top])

        semantic_ranks = {
            doc_id: rank
            for rank, doc_id in enumerate(rankings.get(SEMANTIC, []), start=1)
        }
        lexical_ranks = {
            doc_id: rank
            for rank, doc_id in enumerate(rankings.get(LEXICAL, []), start=1)
        }

        documents: list[RetrievedDocument] = []
        for doc_id, fused_score in top:
            metadata = metadata_by_id.get(doc_id)
            if metadata is None:
                logger.warning("Inventory row %d vanished during retrieval.", doc_id)
                continue
            extracted_text = metadata.pop("extractedText", None)
            sources = []
            if doc_id in semantic_ranks:
                sources.append(SEMANTIC)
            if doc_id in lexical_ranks:
                sources.append(LEXICAL)
            documents.append(
                RetrievedDocument(
                    inventory_id=doc_id,
                    fused_score=fused_score,
                    sources=sources,
                    semantic_rank=semantic_ranks.get(doc_id),
                    lexical_rank=lexical_ranks.get(doc_id),
                    metadata=metadata,
                    snippet=_build_snippet(
                        extracted_text, terms.like_terms, self._config.snippet_chars
                    ),
                )
            )
        logger.info(
            "Retrieved %d documents (mode=%s, branches=%s).",
            len(documents),
            mode,
            sorted(rankings),
        )
        return documents
