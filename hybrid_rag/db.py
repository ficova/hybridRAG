"""SQLite access layer for the digital-preservation inventory database."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

# Columns returned to the answer-building layer. extractedText is included so
# snippets can be built; callers must truncate before prompting the LLM.
_DOCUMENT_COLUMNS: tuple[str, ...] = (
    "id",
    "contentLocation",
    "size",
    "fileSystemModified",
    "fileSystemCreated",
    "formatMIMEType",
    "formatName",
    "formatVersion",
    "formatRegistryKey",
    "fixityError",
    "processingError",
    "needsOcr",
    "pdfPageCount",
    "extractedTextLength",
    "extractedTextLanguage",
    "extractedText",
)

# Metadata columns on which the lexical branch accepts exact-match filters.
FILTERABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "formatRegistryKey",
        "formatMIMEType",
        "messageDigest",
        "formatName",
        "formatVersion",
        "extractedTextLanguage",
        "needsOcr",
    }
)

_CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    inventory_id INTEGER PRIMARY KEY REFERENCES inventory(id),
    title_embedding BLOB,
    content_embedding BLOB,
    modelKey TEXT NOT NULL,
    computedDate TEXT NOT NULL
)
"""


class InventoryDatabase:
    """Thin wrapper around the SQLite inventory/embeddings database."""

    def __init__(self, path: Path) -> None:
        """Open the database read/write.

        Args:
            path: Filesystem path of the SQLite database.

        Raises:
            FileNotFoundError: If the database file does not exist.
            sqlite3.DatabaseError: If the file is not a valid SQLite database.
        """
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")
        self._path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        # Fail fast on non-SQLite files instead of on the first query.
        self._conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        logger.info("Opened database %s", path)

    def __enter__(self) -> "InventoryDatabase":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def has_table(self, name: str) -> bool:
        """Return True if a table with the given name exists.

        Args:
            name: Table name.
        """
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def count_inventory(self) -> int:
        """Return the number of rows in the inventory table."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM inventory").fetchone()
        return int(row["n"])

    def count_embeddings(self) -> int:
        """Return the number of rows in the embeddings table (0 if absent)."""
        if not self.has_table("embeddings"):
            return 0
        row = self._conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()
        return int(row["n"])

    def fetch_documents(self, ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Fetch document metadata (including extractedText) for the given ids.

        Args:
            ids: Inventory row ids.

        Returns:
            Mapping of id to a column/value dict; ids not found are omitted.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        sql = (
            f"SELECT {', '.join(_DOCUMENT_COLUMNS)} FROM inventory "
            f"WHERE id IN ({placeholders})"
        )
        rows = self._conn.execute(sql, tuple(ids)).fetchall()
        return {int(row["id"]): dict(row) for row in rows}

    def lexical_search(
        self,
        terms: Sequence[str],
        filters: Mapping[str, str],
        limit: int,
    ) -> list[tuple[int, float]]:
        """Rank rows by keyword hits in extractedText/contentLocation plus filters.

        Each term scores 2 points for a substring match in extractedText and
        1 point for a match in contentLocation (``LIKE`` is case-insensitive
        for ASCII in SQLite). Exact-match metadata filters restrict the
        candidate set; with filters only (no terms) matching rows score 1.

        Args:
            terms: Keywords and phrases to match with ``LIKE '%term%'``.
            filters: Column -> exact value; columns outside FILTERABLE_COLUMNS
                are ignored with a warning.
            limit: Maximum number of results.

        Returns:
            List of (inventory_id, score) sorted by descending score.
        """
        safe_filters = {}
        for column, value in filters.items():
            if column in FILTERABLE_COLUMNS:
                safe_filters[column] = value
            else:
                logger.warning("Ignoring non-filterable column %r", column)
        if not terms and not safe_filters:
            return []

        score_parts: list[str] = []
        where_parts: list[str] = []
        score_params: list[str] = []
        where_params: list[str] = []
        for term in terms:
            pattern = f"%{term}%"
            score_parts.append(
                "(CASE WHEN extractedText LIKE ? THEN 2 ELSE 0 END"
                " + CASE WHEN contentLocation LIKE ? THEN 1 ELSE 0 END)"
            )
            score_params.extend([pattern, pattern])
            where_parts.append("(extractedText LIKE ? OR contentLocation LIKE ?)")
            where_params.extend([pattern, pattern])

        filter_clauses: list[str] = []
        filter_params: list[str] = []
        for column, value in safe_filters.items():
            filter_clauses.append(f"{column} = ?")
            filter_params.append(value)

        score_expr = " + ".join(score_parts) if score_parts else "1"
        conditions: list[str] = []
        if where_parts:
            conditions.append("(" + " OR ".join(where_parts) + ")")
        if filter_clauses:
            conditions.append(" AND ".join(filter_clauses))
        where_expr = " AND ".join(conditions)

        sql = (
            f"SELECT id, ({score_expr}) AS lex_score FROM inventory "
            f"WHERE {where_expr} ORDER BY lex_score DESC, id ASC LIMIT ?"
        )
        params = [*score_params, *where_params, *filter_params, limit]
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            logger.exception("Lexical search query failed")
            return []
        results = [(int(row["id"]), float(row["lex_score"])) for row in rows]
        logger.debug(
            "Lexical search terms=%s filters=%s -> %d rows",
            list(terms),
            safe_filters,
            len(results),
        )
        return results

    def iter_embedding_rows(self) -> Iterator[sqlite3.Row]:
        """Yield all rows from the embeddings table.

        Yields:
            Rows with inventory_id, title_embedding, content_embedding.
        """
        if not self.has_table("embeddings"):
            return
        cursor = self._conn.execute(
            "SELECT inventory_id, title_embedding, content_embedding "
            "FROM embeddings"
        )
        yield from cursor

    def ensure_embeddings_table(self) -> None:
        """Create the embeddings table if it does not exist."""
        self._conn.execute(_CREATE_EMBEDDINGS_TABLE)
        self._conn.commit()

    def rows_missing_embeddings(self) -> list[sqlite3.Row]:
        """Return inventory rows that have no embeddings row yet.

        Returns:
            Rows with id, contentLocation and extractedText.
        """
        self.ensure_embeddings_table()
        cursor = self._conn.execute(
            "SELECT i.id, i.contentLocation, i.extractedText "
            "FROM inventory AS i "
            "LEFT JOIN embeddings AS e ON e.inventory_id = i.id "
            "WHERE e.inventory_id IS NULL "
            "ORDER BY i.id"
        )
        return cursor.fetchall()

    def upsert_embedding(
        self,
        inventory_id: int,
        title_embedding: bytes | None,
        content_embedding: bytes | None,
        model_key: str,
        computed_date: str,
    ) -> None:
        """Insert or replace an embeddings row (caller must commit).

        Args:
            inventory_id: Inventory row id.
            title_embedding: float32 vector bytes for the title, or None.
            content_embedding: float32 vector bytes for the content, or None.
            model_key: Identifier of the embedding model used.
            computed_date: ISO-8601 timestamp of computation.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(inventory_id, title_embedding, content_embedding, modelKey, "
            "computedDate) VALUES (?, ?, ?, ?, ?)",
            (
                inventory_id,
                title_embedding,
                content_embedding,
                model_key,
                computed_date,
            ),
        )

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()
