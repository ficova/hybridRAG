"""Terminal interface for the hybrid RAG system."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import ollama

from .config import AppConfig, load_config, setup_logging
from .db import InventoryDatabase
from .embeddings import OllamaEmbedder, SemanticIndex, backfill_embeddings
from .llm import ChatError, OllamaChat, build_context_block
from .retrieval import (
    HYBRID,
    LEXICAL,
    SEMANTIC,
    HybridRetriever,
    RetrievedDocument,
)
from .router import QueryRouter

logger = logging.getLogger(__name__)

AUTO = "auto"
_REPL_HELP = """\
Commands:
  /mode auto|semantic|lexical|hybrid   set the retrieval mode
  /context                             toggle printing the retrieved context
  /help                                show this help
  /exit                                quit
Anything else is treated as a question about the archive.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="hybrid-rag",
        description=(
            "Ask natural-language questions about a digital-preservation "
            "SQLite inventory using local hybrid retrieval (semantic + "
            "lexical, fused with RRF) and a local Ollama LLM."
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to answer; omit to start an interactive session.",
    )
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path.")
    parser.add_argument(
        "--config", type=Path, default=None, help="Config YAML path."
    )
    parser.add_argument(
        "--mode",
        choices=[AUTO, SEMANTIC, LEXICAL, HYBRID],
        default=None,
        help="Retrieval mode; 'auto' lets the LLM decide (default from config).",
    )
    parser.add_argument(
        "--model", default=None, help="Ollama chat model (overrides config)."
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Documents passed to the LLM."
    )
    parser.add_argument(
        "--backfill-embeddings",
        action="store_true",
        help="Compute missing embeddings for the inventory, then exit.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Retrieval only: print ranked documents without generating an answer.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the exact context block sent to the LLM.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="DEBUG logging on stdout."
    )
    return parser


def _format_documents(documents: Sequence[RetrievedDocument]) -> str:
    """Render retrieved documents as a compact terminal listing."""
    lines: list[str] = []
    for position, document in enumerate(documents, start=1):
        location = document.metadata.get("contentLocation") or "(no path)"
        format_name = document.metadata.get("formatName") or "unknown format"
        ranks: list[str] = []
        if document.semantic_rank is not None:
            ranks.append(f"sem#{document.semantic_rank}")
        if document.lexical_rank is not None:
            ranks.append(f"lex#{document.lexical_rank}")
        lines.append(
            f"  {position:2d}. [id {document.inventory_id}] {location}\n"
            f"      {format_name} | rrf={document.fused_score:.4f} "
            f"| {' '.join(ranks)}"
        )
    return "\n".join(lines)


def _resolve_mode(
    requested: str, question: str, router: QueryRouter
) -> str:
    """Resolve 'auto' to a concrete retrieval mode via the router."""
    if requested != AUTO:
        return requested
    return router.route(question)


def answer_question(
    question: str,
    requested_mode: str,
    retriever: HybridRetriever,
    router: QueryRouter,
    chat: OllamaChat | None,
    show_context: bool,
) -> int:
    """Retrieve documents for a question and (optionally) generate an answer.

    Args:
        question: The user's question.
        requested_mode: ``auto`` or a concrete retrieval mode.
        retriever: Hybrid retriever.
        router: Query router used when the mode is ``auto``.
        chat: Chat backend, or None for retrieval-only output.
        show_context: Whether to print the LLM context block.

    Returns:
        Process exit code (0 on success, 1 when nothing could be produced).
    """
    mode = _resolve_mode(requested_mode, question, router)
    print(f"\n[retrieval mode: {mode}]")

    documents = retriever.retrieve(question, mode)
    if not documents:
        print(
            "No matching documents found. Try different keywords, another "
            "retrieval mode (--mode), or check that the database has "
            "extracted text and embeddings."
        )
        return 1

    print(f"\nRetrieved {len(documents)} documents:")
    print(_format_documents(documents))

    if show_context:
        print("\n--- context sent to LLM ---")
        print(build_context_block(documents))
        print("--- end context ---")

    if chat is None:
        return 0

    print(f"\n[answering with {chat.model}]\n")
    try:
        chat.answer(question, documents, on_token=lambda piece: print(piece, end="", flush=True))
    except ChatError as exc:
        print(f"\nLLM error: {exc}")
        print("The retrieved documents above are still valid results.")
        return 1
    print()
    return 0


def run_repl(
    requested_mode: str,
    retriever: HybridRetriever,
    router: QueryRouter,
    chat: OllamaChat | None,
    show_context: bool,
) -> int:
    """Run the interactive question loop.

    Args:
        requested_mode: Initial retrieval mode (``auto`` or concrete).
        retriever: Hybrid retriever.
        router: Query router.
        chat: Chat backend, or None for retrieval-only output.
        show_context: Initial context-printing toggle.

    Returns:
        Process exit code.
    """
    mode = requested_mode
    print("Hybrid RAG - ask questions about the archive. /help for commands.")
    while True:
        try:
            line = input(f"\n[{mode}] rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split()
            command = parts[0].lower()
            if command in ("/exit", "/quit"):
                print("Bye.")
                return 0
            if command == "/help":
                print(_REPL_HELP)
            elif command == "/context":
                show_context = not show_context
                print(f"Context printing is now {'on' if show_context else 'off'}.")
            elif command == "/mode" and len(parts) == 2 and parts[1] in (
                AUTO,
                SEMANTIC,
                LEXICAL,
                HYBRID,
            ):
                mode = parts[1]
                print(f"Retrieval mode set to {mode}.")
            else:
                print(f"Unknown command: {line}\n{_REPL_HELP}")
            continue
        answer_question(line, mode, retriever, router, chat, show_context)


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    if hasattr(sys.stdout, "reconfigure"):
        # Windows consoles often default to cp1252; model output is UTF-8.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_arg_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    setup_logging(config.log_file, "DEBUG" if args.verbose else config.log_level)

    if args.top_k is not None:
        config.retrieval.top_k = max(1, args.top_k)
    requested_mode = args.mode or (
        config.router_mode
        if config.router_mode in (AUTO, SEMANTIC, LEXICAL, HYBRID)
        else AUTO
    )
    database_path = args.db or Path(config.database_path)

    try:
        db = InventoryDatabase(database_path)
    except (FileNotFoundError, sqlite3.DatabaseError) as exc:
        print(f"Error opening database {database_path}: {exc}", file=sys.stderr)
        return 2

    with db:
        client = ollama.Client(
            host=config.ollama_host, timeout=config.chat.timeout_seconds
        )
        embedder = OllamaEmbedder(client, config.embedding)

        if args.backfill_embeddings:
            done, failed = backfill_embeddings(db, embedder, config.embedding)
            print(f"Backfill finished: {done} embedded, {failed} failed.")
            return 0 if failed == 0 else 1

        total = db.count_inventory()
        with_embeddings = db.count_embeddings()
        logger.info(
            "Inventory: %d rows, %d with embeddings.", total, with_embeddings
        )
        index = None
        if with_embeddings > 0:
            index = SemanticIndex.load(db, config.embedding.dim)
        else:
            logger.warning(
                "No embeddings found; semantic search is disabled. "
                "Run with --backfill-embeddings to create them."
            )
        retriever = HybridRetriever(db, index, embedder, config.retrieval)

        chat: OllamaChat | None = None
        if not args.no_llm:
            try:
                chat = OllamaChat(client, config.chat, model=args.model)
            except ChatError as exc:
                print(
                    f"Warning: LLM unavailable ({exc}). "
                    "Continuing in retrieval-only mode.",
                    file=sys.stderr,
                )
        router = QueryRouter(chat)

        if args.question:
            return answer_question(
                args.question,
                requested_mode,
                retriever,
                router,
                chat,
                args.show_context,
            )
        return run_repl(
            requested_mode, retriever, router, chat, args.show_context
        )
