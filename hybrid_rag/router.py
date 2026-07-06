"""Query routing: decide whether to emphasize semantic, lexical or hybrid."""

from __future__ import annotations

import logging
import re

from .llm import ChatError, OllamaChat
from .retrieval import HYBRID, LEXICAL, SEMANTIC, VALID_MODES, extract_query_terms

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You route search queries for a digital-preservation archive to the best
retrieval strategy. Reply with exactly one word.

- "lexical": the query targets exact identifiers or literal strings - file
  names or paths, checksums, PRONOM PUIDs (e.g. fmt/19), MIME types, format
  names/versions, error messages, or quoted phrases that must match verbatim.
- "semantic": the query is conceptual or topical - it asks about meaning,
  subjects, or content of documents, paraphrased in natural language.
- "hybrid": the query mixes both, or you are unsure.

Reply with one word: semantic, lexical, or hybrid.
"""

_MODE_RE = re.compile(r"\b(semantic|lexical|hybrid)\b", re.IGNORECASE)
_CONCEPTUAL_OPENERS = ("what", "which", "who", "why", "how", "summarize", "describe")


def heuristic_route(question: str) -> str:
    """Classify a query with pattern-based rules (no LLM required).

    Args:
        question: Natural-language question.

    Returns:
        One of ``semantic``, ``lexical`` or ``hybrid``.
    """
    terms = extract_query_terms(question)
    if terms.filters or terms.phrases:
        return LEXICAL
    first_word = question.strip().split(" ", 1)[0].lower() if question.strip() else ""
    if first_word in _CONCEPTUAL_OPENERS and not terms.filters:
        return SEMANTIC if len(terms.keywords) <= 3 else HYBRID
    return HYBRID


class QueryRouter:
    """Chooses a retrieval mode, preferring LLM classification when available."""

    def __init__(self, chat: OllamaChat | None) -> None:
        """Initialize the router.

        Args:
            chat: Chat backend for LLM classification, or None to always use
                the heuristic.
        """
        self._chat = chat

    def route(self, question: str) -> str:
        """Pick the retrieval mode for a question.

        Asks the LLM to classify the query; on failure or an unparseable
        reply, falls back to :func:`heuristic_route`.

        Args:
            question: Natural-language question.

        Returns:
            One of ``semantic``, ``lexical`` or ``hybrid``.
        """
        if self._chat is not None:
            try:
                reply = self._chat.complete(
                    ROUTER_SYSTEM_PROMPT, f"Query: {question}"
                )
            except ChatError:
                logger.exception("LLM routing failed; using heuristic.")
            else:
                match = _MODE_RE.search(reply)
                if match:
                    mode = match.group(1).lower()
                    if mode in VALID_MODES:
                        logger.info("LLM routed query to %r.", mode)
                        return mode
                logger.warning(
                    "Unparseable router reply %r; using heuristic.", reply[:120]
                )
        mode = heuristic_route(question)
        logger.info("Heuristic routed query to %r.", mode)
        return mode
