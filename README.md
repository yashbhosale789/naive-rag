# Naive RAG

The simplest retrieval-augmented generation pipeline: embed the question, take the nearest chunks, answer from them. One of six RAG variants in a portfolio (Naive, Advanced, Production, Agentic, Graph, Multimodal) measured on the same corpus and dataset; the production system is [Personal-Knowledge-RAG](https://github.com/yashbhosale789/Personal-Knowledge-RAG).

## Pipeline

| Stage | Choice |
|---|---|
| Loading | LangChain `PyPDFLoader` (one document per page); `.txt`, `.md`, `.docx` also read |
| Chunking | `RecursiveCharacterTextSplitter`, 512 characters, 64 overlap |
| Embedding | `BAAI/bge-small-en-v1.5` (384 dims, local), query prefix `Represent this sentence: ` |
| Store | Chroma, persisted in `storage/chroma/` |
| Retrieval | dense cosine top 5, nothing else (no keyword search, fusion, reranking or query rewrite) |
| Generation | OpenRouter `nvidia/nemotron-3-nano-30b-a3b:free`, answer only from the context |

## Run

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
echo OPENROUTER_API_KEY=your_key > .env          # only needed for answers, not for retrieval

# Index a corpus folder in place (default: data/documents). Delete storage/ before re-indexing:
# chunks get random ids, so a second ingest duplicates them.
RAG_DATA_DIR=/path/to/corpus RAG_DATA_GLOB="*.pdf" python scripts/ingest.py
python scripts/query.py
```

## Retrieval results (provisional)

Measured 2026-11-07 on the 71 answerable questions of the Personal-Knowledge-RAG golden dataset (`eval/data/golden_dataset.json`, id `8f89d42be9`), retrieval only ($0, no LLM calls), with `python scripts/retrieval_eval.py <golden_dataset.json>`; per-question results are in `eval/retrieval_results.json`. The metric definitions copy that project's eval harness. A hit means the labelled evidence span (source file + text, chunk-independent) is inside a retrieved chunk. **Provisional:** every variant is re-measured through one shared eval kit before the final cross-variant table.

Corpus: the 236 PDFs below, 795 pages → 2,222 chunks (id `2222:9e32e2edff`).

| Variant | Recall@1 | Recall@3 | Recall@5 | Success@3 | MRR |
|---|---|---|---|---|---|
| **Naive RAG** (this repo; top 5 chunks) | 0.41 | 0.65 | 0.73 | 0.59 | 0.59 |
| Advanced RAG (raw question) | 0.58 | 0.74 | 0.79 | 0.68 | 0.73 |
| Production RAG (with query rewrite; top 3 chunks) | 0.68 | 0.88 | 0.88 | 0.85 | 0.83 |

Paired comparison (bootstrap CI + sign test, Bonferroni over 6 comparisons, α 0.0083): Production RAG is higher than Naive RAG on recall@1, @3, @5 and MRR, all distinguishable (recall@3 −0.22, CI [−0.35, −0.11]). Naive vs Advanced is not distinguishable at n = 71 (MRR +0.14 for Advanced, sign test p 0.013).

## Corpus and licence

The measurement corpus is [RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus) (commit `39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3`): every PDF under `datasets/<Company>/pdf/`, 236 files, synthetic documents of five fictional companies. It is MIT-licensed, Copyright (c) 2025 UDAY ALLU. This repository ships one of its files as a sample (`data/documents/`, with the licence in `data/documents/LICENSE`); download the rest from the upstream repository.
