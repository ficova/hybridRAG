"""Embedding generation (Ollama) and in-memory semantic index (numpy)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Sequence

import numpy as np
import numpy.typing as npt
import ollama

from .config import EmbeddingConfig
from .db import InventoryDatabase

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend fails."""


def decode_vector(blob: bytes | None, dim: int) -> FloatArray | None:
    """Decode a float32 BLOB into a numpy vector.

    Args:
        blob: Raw bytes from the database, or None.
        dim: Expected vector dimension.

    Returns:
        A float32 array of shape (dim,), or None if the blob is missing or
        has an unexpected size.
    """
    if blob is None:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.shape[0] != dim:
        logger.warning(
            "Embedding blob has %d floats, expected %d; skipping.",
            vector.shape[0],
            dim,
        )
        return None
    return vector


def _normalize_rows(matrix: FloatArray) -> FloatArray:
    """Return a row-normalized copy of a matrix, guarding zero rows."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class OllamaEmbedder:
    """Generates embeddings with a local Ollama model."""

    def __init__(self, client: ollama.Client, config: EmbeddingConfig) -> None:
        """Initialize the embedder.

        Args:
            client: Configured Ollama client.
            config: Embedding settings.
        """
        self._client = client
        self._config = config

    def _embed(self, texts: Sequence[str]) -> list[FloatArray]:
        """Embed a batch of texts.

        Args:
            texts: Input strings (already instruction-formatted).

        Returns:
            One float32 vector per input.

        Raises:
            EmbeddingError: On backend failure or dimension mismatch.
        """
        try:
            response = self._client.embed(
                model=self._config.model,
                input=list(texts),
                options={"num_ctx": self._config.num_ctx},
            )
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise EmbeddingError(f"Ollama embedding call failed: {exc}") from exc
        raw_vectors = response["embeddings"]
        if len(raw_vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, got {len(raw_vectors)}"
            )
        vectors: list[FloatArray] = []
        for raw in raw_vectors:
            vector = np.asarray(raw, dtype=np.float32)
            if vector.shape[0] != self._config.dim:
                raise EmbeddingError(
                    f"Model returned dim {vector.shape[0]}, expected "
                    f"{self._config.dim}; check embedding.dim in config.yaml"
                )
            vectors.append(vector)
        return vectors

    def embed_query(self, query: str) -> FloatArray:
        """Embed a search query with the query-side instruction.

        Args:
            query: Natural-language query text.

        Returns:
            The query vector.

        Raises:
            EmbeddingError: On backend failure.
        """
        prompt = f"Instruct: {self._config.query_instruction}\nQuery: {query}"
        return self._embed([prompt])[0]

    def embed_document_pair(
        self, title: str, content: str
    ) -> tuple[FloatArray, FloatArray]:
        """Embed a document's title and content with the document instruction.

        Args:
            title: Document title (usually the file name).
            content: Extracted text, already truncated by the caller.

        Returns:
            Tuple of (title_vector, content_vector).

        Raises:
            EmbeddingError: On backend failure.
        """
        instruction = self._config.document_instruction
        prompts = [
            f"Instruct: {instruction}\nQuery: {title}",
            f"Instruct: {instruction}\nQuery: {content}",
        ]
        vectors = self._embed(prompts)
        return vectors[0], vectors[1]


class SemanticIndex:
    """In-memory cosine-similarity index over title and content embeddings."""

    def __init__(
        self,
        ids: FloatArray,
        title_matrix: FloatArray,
        content_matrix: FloatArray,
        title_mask: npt.NDArray[np.bool_],
        content_mask: npt.NDArray[np.bool_],
    ) -> None:
        """Initialize the index; use :meth:`load` instead of calling directly."""
        self._ids = ids
        self._title_matrix = title_matrix
        self._content_matrix = content_matrix
        self._title_mask = title_mask
        self._content_mask = content_mask

    def __len__(self) -> int:
        return int(self._ids.shape[0])

    @classmethod
    def load(cls, db: InventoryDatabase, dim: int) -> "SemanticIndex | None":
        """Load all embeddings from the database into memory.

        Rows where both title and content vectors are missing or malformed
        are skipped.

        Args:
            db: Open inventory database.
            dim: Expected embedding dimension.

        Returns:
            A ready index, or None if no usable embeddings exist.
        """
        ids: list[int] = []
        titles: list[FloatArray] = []
        contents: list[FloatArray] = []
        title_mask: list[bool] = []
        content_mask: list[bool] = []
        zero = np.zeros(dim, dtype=np.float32)

        for row in db.iter_embedding_rows():
            title_vec = decode_vector(row["title_embedding"], dim)
            content_vec = decode_vector(row["content_embedding"], dim)
            if title_vec is None and content_vec is None:
                continue
            ids.append(int(row["inventory_id"]))
            titles.append(title_vec if title_vec is not None else zero)
            contents.append(content_vec if content_vec is not None else zero)
            title_mask.append(title_vec is not None)
            content_mask.append(content_vec is not None)

        if not ids:
            logger.warning("No usable embeddings found in the database.")
            return None

        index = cls(
            ids=np.asarray(ids, dtype=np.int64),
            title_matrix=_normalize_rows(np.vstack(titles)),
            content_matrix=_normalize_rows(np.vstack(contents)),
            title_mask=np.asarray(title_mask, dtype=bool),
            content_mask=np.asarray(content_mask, dtype=bool),
        )
        logger.info("Semantic index loaded: %d documents, dim=%d", len(index), dim)
        return index

    def search(self, query_vector: FloatArray, k: int) -> list[tuple[int, float]]:
        """Return the top-k documents by cosine similarity.

        A document's score is the maximum of its title and content
        similarities (missing vectors are excluded, not treated as zero).

        Args:
            query_vector: Raw (unnormalized) query embedding.
            k: Number of results.

        Returns:
            List of (inventory_id, cosine_similarity) sorted descending.
        """
        norm = float(np.linalg.norm(query_vector))
        if norm == 0.0:
            logger.warning("Query vector has zero norm; semantic search skipped.")
            return []
        query = (query_vector / norm).astype(np.float32)

        title_sims = self._title_matrix @ query
        content_sims = self._content_matrix @ query
        title_sims = np.where(self._title_mask, title_sims, -np.inf)
        content_sims = np.where(self._content_mask, content_sims, -np.inf)
        scores = np.maximum(title_sims, content_sims)

        k = min(k, scores.shape[0])
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self._ids[i]), float(scores[i])) for i in top]


def backfill_embeddings(
    db: InventoryDatabase,
    embedder: OllamaEmbedder,
    config: EmbeddingConfig,
    commit_every: int = 25,
) -> tuple[int, int]:
    """Compute and store embeddings for inventory rows that lack them.

    The title is the file name portion of contentLocation; the content is
    extractedText truncated to ``max_content_chars``. Rows whose embedding
    call fails are skipped and counted, not fatal.

    Args:
        db: Open inventory database.
        embedder: Embedding backend.
        config: Embedding settings.
        commit_every: Commit interval in rows.

    Returns:
        Tuple of (rows_embedded, rows_failed).
    """
    rows = db.rows_missing_embeddings()
    total = len(rows)
    if total == 0:
        logger.info("All inventory rows already have embeddings.")
        return 0, 0
    logger.info("Backfilling embeddings for %d rows...", total)

    done = 0
    failed = 0
    for position, row in enumerate(rows, start=1):
        inventory_id = int(row["id"])
        location = row["contentLocation"] or ""
        title = PurePath(location).name or f"document-{inventory_id}"
        content = (row["extractedText"] or "")[: config.max_content_chars]
        if not content.strip():
            content = title
        try:
            title_vec, content_vec = embedder.embed_document_pair(title, content)
        except EmbeddingError:
            logger.exception("Embedding failed for inventory id %d", inventory_id)
            failed += 1
            continue
        db.upsert_embedding(
            inventory_id=inventory_id,
            title_embedding=title_vec.astype(np.float32).tobytes(),
            content_embedding=content_vec.astype(np.float32).tobytes(),
            model_key=config.model,
            computed_date=datetime.now(timezone.utc).isoformat(),
        )
        done += 1
        if done % commit_every == 0:
            db.commit()
        if position % 50 == 0 or position == total:
            logger.info("Backfill progress: %d/%d", position, total)
    db.commit()
    logger.info("Backfill complete: %d embedded, %d failed.", done, failed)
    return done, failed
