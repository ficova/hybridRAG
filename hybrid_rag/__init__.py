"""Local hybrid RAG over a digital-preservation SQLite inventory.

Combines semantic search (numpy cosine similarity over Ollama embeddings)
with lexical SQL search, fused via Reciprocal Rank Fusion, and answers
questions with a local Ollama chat model.
"""

__version__ = "1.0.0"
