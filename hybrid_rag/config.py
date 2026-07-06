"""Configuration loading and logging setup for the hybrid RAG system."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "config.yaml"

_DEFAULT_PREFERRED_MODELS: tuple[str, ...] = (
    "qwen3.5:9b-q4_K_M",
    "qwen3:8b",
    "phi4:14b",
    "gemma3:12b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "mistral:7b",
    "gemma3:4b",
    "llama3.2:3b",
)


@dataclass
class EmbeddingConfig:
    """Settings for the Ollama embedding model."""

    model: str = "qwen3-embedding:0.6b"
    dim: int = 1024
    document_instruction: str = (
        "Represent this archival document for semantic classification"
    )
    query_instruction: str = (
        "Given a search query, retrieve archival documents relevant to the query"
    )
    num_ctx: int = 11008
    max_content_chars: int = 32000


@dataclass
class ChatConfig:
    """Settings for the Ollama chat model."""

    preferred_models: list[str] = field(
        default_factory=lambda: list(_DEFAULT_PREFERRED_MODELS)
    )
    num_ctx: int = 16384
    temperature: float = 0.2
    timeout_seconds: float = 300.0


@dataclass
class RetrievalConfig:
    """Settings for the hybrid retrieval pipeline."""

    top_k: int = 8
    semantic_candidates: int = 30
    lexical_candidates: int = 30
    rrf_k: int = 60
    snippet_chars: int = 1500


@dataclass
class AppConfig:
    """Top-level application configuration."""

    database_path: str = "inventory.db"
    ollama_host: str = "http://localhost:11434"
    log_file: str = "hybrid_rag.log"
    log_level: str = "INFO"
    router_mode: str = "auto"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a mapping section from raw YAML data, tolerating absence.

    Args:
        raw: Parsed YAML document.
        key: Section name.

    Returns:
        The section as a dict, or an empty dict if missing or malformed.
    """
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    if value is not None:
        logger.warning("Config section %r is not a mapping; ignoring it.", key)
    return {}


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        path: Explicit config file path. When None, ``config.yaml`` next to
            the current working directory is used if it exists.

    Returns:
        A fully populated AppConfig.

    Raises:
        FileNotFoundError: If an explicit path was given but does not exist.
    """
    config = AppConfig()
    if path is None:
        candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if not candidate.exists():
            logger.info("No config.yaml found; using built-in defaults.")
            return config
        path = candidate
    elif not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    database = _section(raw, "database")
    config.database_path = str(database.get("path", config.database_path))

    ollama_cfg = _section(raw, "ollama")
    config.ollama_host = str(ollama_cfg.get("host", config.ollama_host))

    logging_cfg = _section(raw, "logging")
    config.log_file = str(logging_cfg.get("file", config.log_file))
    config.log_level = str(logging_cfg.get("level", config.log_level))

    router_cfg = _section(raw, "router")
    config.router_mode = str(router_cfg.get("mode", config.router_mode)).lower()

    emb = _section(raw, "embedding")
    e = config.embedding
    e.model = str(emb.get("model", e.model))
    e.dim = int(emb.get("dim", e.dim))
    e.document_instruction = str(
        emb.get("document_instruction", e.document_instruction)
    )
    e.query_instruction = str(emb.get("query_instruction", e.query_instruction))
    e.num_ctx = int(emb.get("num_ctx", e.num_ctx))
    e.max_content_chars = int(emb.get("max_content_chars", e.max_content_chars))

    chat = _section(raw, "chat")
    c = config.chat
    preferred = chat.get("preferred_models")
    if isinstance(preferred, list) and preferred:
        c.preferred_models = [str(name) for name in preferred]
    c.num_ctx = int(chat.get("num_ctx", c.num_ctx))
    c.temperature = float(chat.get("temperature", c.temperature))
    c.timeout_seconds = float(chat.get("timeout_seconds", c.timeout_seconds))

    retrieval = _section(raw, "retrieval")
    r = config.retrieval
    r.top_k = int(retrieval.get("top_k", r.top_k))
    r.semantic_candidates = int(
        retrieval.get("semantic_candidates", r.semantic_candidates)
    )
    r.lexical_candidates = int(
        retrieval.get("lexical_candidates", r.lexical_candidates)
    )
    r.rrf_k = int(retrieval.get("rrf_k", r.rrf_k))
    r.snippet_chars = int(retrieval.get("snippet_chars", r.snippet_chars))

    logger.info("Loaded configuration from %s", path)
    return config


def setup_logging(log_file: str, level: str = "INFO") -> None:
    """Configure root logging to both a file and stdout.

    Args:
        log_file: Path of the log file (created/appended).
        level: Logging level name for the stdout handler; the file handler
            always records DEBUG and above.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid duplicate handlers when called twice (e.g. tests, REPL restarts).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(file_handler)

    # Third-party HTTP chatter is noisy at DEBUG; keep it at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
