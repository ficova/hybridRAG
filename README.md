# hybridRAG

Local hybrid RAG over a digital-preservation SQLite inventory. Everything runs
on your machine: retrieval is pure `sqlite3` + `numpy`, and both embeddings and
chat completions go through a local [Ollama](https://ollama.com) server. No
vector database, no external APIs.

## How it works

```
question
   |
   v
QueryRouter (LLM classifies: semantic | lexical | hybrid; heuristic fallback)
   |
   +--> semantic branch: Ollama query embedding -> numpy cosine similarity
   |                     over title/content vectors from the embeddings table
   |
   +--> lexical branch:  SQL LIKE '%kw%' on extractedText/contentLocation
   |                     + exact-match filters (PUID, SHA-256, MIME type)
   |
   v
Reciprocal Rank Fusion   score(d) = sum over branches of 1 / (k + rank_d), k=60
   |
   v
top-k documents (metadata + text excerpt) -> local Ollama chat model -> answer
```

Graceful degradation: a database without an `embeddings` table (or an
unreachable Ollama server) automatically falls back to lexical-only retrieval;
a failing chat model still prints the ranked documents.

## Setup

Python **3.11** is pinned (CUDA toolchain compatibility on the PC). The Python
side has no GPU dependencies — Ollama handles Metal (macOS) / CUDA (Windows)
itself — so a plain `venv` is preferred over conda on both machines.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS (M1)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama models

```bash
ollama pull qwen3-embedding:0.6b     # embeddings (1024-dim, matches the DB)
ollama pull qwen3.5:9b-q4_K_M        # default chat model
```

### Choosing the chat model

`chat.preferred_models` in `config.yaml` is an ordered preference list; the
first *installed* model wins, and `--model` overrides everything. Guidance for
the two target machines:

- **MacBook Air M1, 16 GB**: `qwen3.5:9b-q4_K_M` (6.6 GB) is the sweet spot.
  `qwen3:8b` and `gemma3:4b` are lighter fallbacks.
- **Windows PC, 6 GB VRAM / 64 GB RAM**: `qwen3.5:9b-q4_K_M` with partial CPU
  offload works well; `phi4:14b` or `qwen2.5:14b` are usable but slower.
  `qwen3.5:27b`/`35b` only make sense here with heavy CPU offload — put them
  first in the list on a bigger GPU.
- Models tagged `-cloud` execute on ollama.com and are **never** selected —
  all inference stays local.

## Usage

```powershell
# (a) semantic query - conceptual, no exact terms
python main.py "Which documents discuss land ownership disputes in the 19th century?" --db inventory.db --mode semantic

# (b) lexical / exact-term query - identifiers force the lexical branch
python main.py 'Show files with PRONOM PUID fmt/19 that mention "notarial deed"' --db inventory.db --mode lexical

# (c) hybrid query - concept + exact term, both branches fused via RRF
python main.py "Correspondence about water rights mentioning the name Ybarra" --db inventory.db --mode hybrid
```

Leave `--mode` off (or set `auto`) and the LLM routes the query itself.

Other useful invocations:

```powershell
python main.py                                  # interactive REPL (/help)
python main.py --db other.db --backfill-embeddings   # compute missing embeddings
python main.py "..." --no-llm                   # retrieval only, no generation
python main.py "..." --show-context --verbose   # inspect prompt + debug logs
```

## Database schema assumption

The SQLite database must contain an `inventory` table (one row per file):

| column | type | notes |
|---|---|---|
| id | INTEGER PK | AUTOINCREMENT |
| contentLocation | TEXT | file path, indexed |
| size | TEXT | |
| fileSystemAccessed / Modified / Created | TEXT | timestamps |
| messageDigestAlgorithm / messageDigest | TEXT | SHA-256 |
| fixityError / fixityVerificationDate | TEXT | |
| formatMIMEType / formatName / formatVersion | TEXT | |
| formatRegistryKey | TEXT | PRONOM PUID |
| siegfriedError / processingError | TEXT | |
| extractedText | TEXT | full text used for lexical search |
| needsOcr / extractionError / pdfPageCount | TEXT | |
| extractedTextLength | INTEGER | |
| extractedTextLanguage | TEXT | |

And optionally an `embeddings` table (created by `--backfill-embeddings` when
absent):

| column | type | notes |
|---|---|---|
| inventory_id | INTEGER PK, FK | references inventory(id) |
| title_embedding | BLOB | 1024 x float32, little-endian |
| content_embedding | BLOB | 1024 x float32, little-endian |
| modelKey | TEXT NOT NULL | e.g. `qwen3-embedding:0.6b` |
| computedDate | TEXT NOT NULL | ISO-8601 |

Embeddings are `qwen3-embedding:0.6b` vectors (1024-dim) computed with the
instruction *"Represent this archival document for semantic classification"*
and a fixed `num_ctx` of 11008 (keep it constant or Ollama reloads the model
every batch). Backfill reuses the same instruction format so new vectors stay
in the same embedding space; if your original pipeline used a different prompt
template, adjust `embedding.document_instruction` in `config.yaml` to match.

## Logging

All modules log through the standard `logging` module to both stdout and
`hybrid_rag.log` (paths and level configurable in `config.yaml`). Use
`--verbose` for DEBUG output on the console.
