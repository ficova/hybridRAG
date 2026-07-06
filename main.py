"""Entry point for the hybrid RAG CLI.

Usage:
    python main.py "your question" --db inventory.db
    python main.py            # interactive session
"""

from __future__ import annotations

from hybrid_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
