# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local hybrid RAG over a digital-preservation SQLite inventory (one row per
archived file). Retrieval is pure `sqlite3` + `numpy` — deliberately **no
vector database** — and all LLM work (embeddings + chat) goes through a local
Ollama server. No external APIs; models with a `-cloud` tag are explicitly
filtered out in `llm.select_chat_model`.

## Commands

```powershell
# Environment (Python 3.11 pinned for CUDA compatibility; venv, not conda)
py -3.11 -m venv .venv            # macOS: python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS: source .venv/bin/activate
pip install -r requirements.txt

# Run (one-shot / REPL / retrieval-only debug)
python main.py "question" --db inventory.db [--mode semantic|lexical|hybrid|auto]
python main.py                                    # interactive REPL, /help
python main.py "q" --db x.db --no-llm --verbose   # skip generation, DEBUG logs
python main.py "q" --db x.db --show-context       # print exact LLM prompt context

# Compute embeddings for a DB that lacks them (creates the table if absent)
python main.py --db x.db --backfill-embeddings
```

There is no committed test suite, linter config, or build step. To verify
changes without a real archive: build a small SQLite DB with the two tables
(schema in README.md), insert rows with known `extractedText`, and drive
retrieval with `--no-llm --verbose` — the lexical branch and RRF fusion need
no Ollama at all (`--mode lexical`). CLI flags override `config.yaml`, which
overrides dataclass defaults in `hybrid_rag/config.py`.

## Architecture

Question flow (one direction, no cycles — keep it that way):

```
cli.py ── mode "auto"? ──> router.py (LLM classifies semantic|lexical|hybrid,
   |                                  heuristic_route fallback on any failure)
   v
retrieval.HybridRetriever.retrieve(question, mode)
   ├─ semantic: embeddings.OllamaEmbedder.embed_query -> SemanticIndex.search
   │            (numpy cosine over row-normalized matrices loaded once at startup;
   │             doc score = max(title_sim, content_sim))
   ├─ lexical:  retrieval.extract_query_terms -> db.lexical_search
   │            (LIKE '%term%' on extractedText/contentLocation, scored 2/1;
   │             quoted phrases; PUID/SHA-256/MIME regexes become exact-match
   │             filters, whitelisted in db.FILTERABLE_COLUMNS)
   └─ fusion:   reciprocal_rank_fusion — score = Σ 1/(k + rank), rank from 1, k=60
   v
llm.OllamaChat.answer — context block = per-doc metadata + text excerpt,
                        system prompt demands [Doc N] citations, grounded-only
```

Module dependency order (import downward only):
`cli` → `router` → `llm` → `retrieval` → `embeddings`/`db` → `config`.

### Graceful-degradation contract

Every layer must keep working when the layer above it can't:
no `embeddings` table or unreachable Ollama → lexical-only retrieval (warned,
not fatal); chat model init fails → retrieval-only output; LLM router fails →
`heuristic_route`; a branch returning nothing → RRF fuses whatever remains;
malformed embedding BLOBs → skipped row, not a crash. Preserve this when
extending — errors surface as logs + fallbacks, not exceptions to the user.

### Embedding-space invariants (breaking these silently ruins retrieval)

- Vectors are 1024-dim float32 little-endian BLOBs from `qwen3-embedding:0.6b`.
  `embedding.dim` in config.yaml must match the model's output; `decode_vector`
  drops mismatched rows.
- Both queries and backfilled documents use the `Instruct: {instruction}\nQuery:
  {text}` template (`document_instruction` vs `query_instruction` in config).
  New vectors must stay in the same space as the pre-existing corpus vectors.
- `embedding.num_ctx` is fixed at 11008: changing it between calls makes Ollama
  reload the model every batch (throughput collapse). Content is capped at
  `max_content_chars` = 32000 before embedding.

### Other conventions

- Target machines: MacBook Air M1 16GB (Metal) and Windows 11 PC with 6GB VRAM
  / 64GB RAM (CUDA). Ollama owns the GPU; Python deps stay GPU-free (numpy,
  ollama, PyYAML only — don't add heavy deps). Default chat model
  `qwen3.5:9b-q4_K_M` via `chat.preferred_models` (first installed wins).
- All SQL uses parameterized queries; new metadata filters must be added to
  `db.FILTERABLE_COLUMNS`, never interpolated column names from user input.
- Ollama python API responses are accessed tolerantly (dict or pydantic,
  `model` or `name` keys) — keep that when touching `llm.list_installed_models`.
- Logging goes to stdout **and** `hybrid_rag.log` (configured once in
  `config.setup_logging`); `print()` is reserved for user-facing CLI output in
  `cli.py`. Style: Black 88 cols, Google docstrings, full type hints.
