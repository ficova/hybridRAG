"""Ollama chat integration: model selection, prompting and answering."""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import httpx
import ollama

from .config import ChatConfig
from .retrieval import RetrievedDocument

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """\
You are an assistant for a digital-preservation archive. You answer questions
about files in the archive using ONLY the documents provided in the context.

Rules:
- Base every claim on the provided documents; never invent files or metadata.
- Cite supporting documents inline with their label, e.g. [Doc 2], and mention
  the file path when it helps the reader locate the file.
- If the context does not contain enough information to answer, say so
  explicitly and suggest what to search for instead.
- Be concise and factual. Preserve exact identifiers (paths, checksums,
  PRONOM PUIDs, MIME types) verbatim.
"""

_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("contentLocation", "Path"),
    ("formatName", "Format"),
    ("formatVersion", "Format version"),
    ("formatMIMEType", "MIME type"),
    ("formatRegistryKey", "PRONOM PUID"),
    ("size", "Size (bytes)"),
    ("fileSystemModified", "Modified"),
    ("fileSystemCreated", "Created"),
    ("pdfPageCount", "PDF pages"),
    ("extractedTextLength", "Extracted text length"),
    ("extractedTextLanguage", "Language"),
    ("needsOcr", "Needs OCR"),
    ("fixityError", "Fixity error"),
    ("processingError", "Processing error"),
)


class ChatError(RuntimeError):
    """Raised when the chat backend fails or no usable model is installed."""


def list_installed_models(client: ollama.Client) -> list[str]:
    """Return the names of all locally installed Ollama models.

    Args:
        client: Configured Ollama client.

    Returns:
        Model names (possibly empty on API shape mismatch).

    Raises:
        ChatError: If the Ollama server cannot be reached.
    """
    try:
        response = client.list()
    except (ollama.ResponseError, ConnectionError, OSError, httpx.HTTPError) as exc:
        raise ChatError(f"Cannot reach Ollama server: {exc}") from exc
    models: Any = (
        response.get("models", [])
        if isinstance(response, dict)
        else getattr(response, "models", [])
    )
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(str(name))
    return names


def select_chat_model(client: ollama.Client, config: ChatConfig) -> str:
    """Pick the best installed chat model.

    Walks ``preferred_models`` in order and returns the first installed one.
    Cloud-hosted models (``-cloud`` tags) and embedding models are never
    selected, keeping all inference strictly local.

    Args:
        client: Configured Ollama client.
        config: Chat settings with the preference list.

    Returns:
        The selected model name.

    Raises:
        ChatError: If no suitable local model is installed.
    """

    def is_local_chat_model(name: str) -> bool:
        lowered = name.lower()
        return "cloud" not in lowered and "embed" not in lowered

    installed = [name for name in list_installed_models(client) if name]
    installed_set = set(installed)
    for candidate in config.preferred_models:
        if candidate in installed_set and is_local_chat_model(candidate):
            logger.info("Selected chat model: %s", candidate)
            return candidate
    for name in installed:
        if is_local_chat_model(name):
            logger.warning(
                "No preferred model installed; falling back to %s. "
                "Adjust chat.preferred_models in config.yaml.",
                name,
            )
            return name
    raise ChatError(
        "No local chat model installed. Pull one, e.g.: "
        "ollama pull qwen3.5:9b-q4_K_M"
    )


def build_context_block(documents: Sequence[RetrievedDocument]) -> str:
    """Format retrieved documents (metadata + text excerpt) for the prompt.

    Args:
        documents: Fused retrieval results.

    Returns:
        A plain-text context block with one labeled section per document.
    """
    sections: list[str] = []
    for position, document in enumerate(documents, start=1):
        lines = [f"[Doc {position}] (inventory id {document.inventory_id})"]
        for column, label in _METADATA_FIELDS:
            value = document.metadata.get(column)
            if value is None or str(value).strip() == "":
                continue
            lines.append(f"{label}: {value}")
        lines.append(f"Retrieved via: {', '.join(document.sources)}")
        if document.snippet:
            lines.append(f"Text excerpt: {document.snippet}")
        else:
            lines.append("Text excerpt: (no extracted text available)")
        sections.append("\n".join(lines))
    return "\n\n---\n\n".join(sections)


def build_answer_prompt(question: str, documents: Sequence[RetrievedDocument]) -> str:
    """Build the user prompt combining the context block and the question.

    Args:
        question: The user's natural-language question.
        documents: Retrieved documents to ground the answer.

    Returns:
        The complete user-role prompt.
    """
    context = build_context_block(documents)
    return (
        f"Context documents from the archive:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context documents above."
    )


class OllamaChat:
    """Wrapper around Ollama chat completions with a fixed model."""

    def __init__(
        self,
        client: ollama.Client,
        config: ChatConfig,
        model: str | None = None,
    ) -> None:
        """Initialize the chat backend.

        Args:
            client: Configured Ollama client (its timeout applies here).
            config: Chat settings.
            model: Explicit model name; when None the best installed
                preferred model is selected automatically.

        Raises:
            ChatError: If no usable model can be selected.
        """
        self._client = client
        self._config = config
        self.model = model or select_chat_model(client, config)

    def complete(
        self,
        system: str,
        user: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Run a chat completion.

        Args:
            system: System prompt.
            user: User prompt.
            on_token: Optional callback invoked with each streamed text
                chunk; when None the call is non-streaming.

        Returns:
            The full response text.

        Raises:
            ChatError: On timeout, connection failure or backend error.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        options = {
            "temperature": self._config.temperature,
            "num_ctx": self._config.num_ctx,
        }
        try:
            if on_token is None:
                response = self._client.chat(
                    model=self.model, messages=messages, options=options
                )
                return str(response["message"]["content"])
            chunks: list[str] = []
            for part in self._client.chat(
                model=self.model, messages=messages, options=options, stream=True
            ):
                piece = str(part["message"]["content"])
                if piece:
                    chunks.append(piece)
                    on_token(piece)
            return "".join(chunks)
        except httpx.TimeoutException as exc:
            raise ChatError(
                f"LLM call timed out after {self._config.timeout_seconds:.0f}s "
                f"(model {self.model}). Try a smaller model or raise "
                "chat.timeout_seconds."
            ) from exc
        except (ollama.ResponseError, ConnectionError, OSError, httpx.HTTPError) as exc:
            raise ChatError(f"LLM call failed: {exc}") from exc

    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Answer a question grounded in the retrieved documents.

        Args:
            question: The user's question.
            documents: Retrieved context documents.
            on_token: Optional streaming callback.

        Returns:
            The model's answer text.

        Raises:
            ChatError: On backend failure.
        """
        prompt = build_answer_prompt(question, documents)
        logger.debug("Answer prompt is %d characters.", len(prompt))
        return self.complete(ANSWER_SYSTEM_PROMPT, prompt, on_token=on_token)
