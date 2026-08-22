# Bulbar MND Literature RAG — Design Document

## 1. Purpose

The application helps MND learners and research users ask literature-based questions about motor neuron disease (MND), amyotrophic lateral sclerosis (ALS), and bulbar-onset disease. It is delivered through a Streamlit dashboard chatbot and a command-line interface; it is research support only, not clinical decision support.

## 2. Scope

The chatbot answers only from the indexed research corpus and provides clickable PubMed Central (PMC) links for the articles it used. A separate Streamlit dashboard presents IHME MND burden estimates; those estimates are not part of the literature RAG corpus.

## 3. Corpus

| Item | Design |
| --- | --- |
| Source of truth | Open-access PubMed Central articles |
| Current size | 100 articles |
| File format | PMC XML |
| Language | Primarily English |
| Local source folder | `data/pmc_xml` |
| Vector store | Chroma, persisted in `data/chroma_db` |

## 4. Frameworks and Libraries

| Component | Framework / library | How it is used |
| --- | --- | --- |
| Application UI | Streamlit | Dashboard filters, burden visualizations, and the literature chatbot. |
| RAG building blocks | LangChain | `Document` objects, embedding interface, text splitter baseline, Chroma integration, and chat-model wrapper. |
| Workflow orchestration | LangGraph | The retrieve → generate state-machine flow. |
| Vector database | Chroma | Local persistent storage and vector similarity retrieval. |
| AI provider integration | OpenAI Python SDK with Nebius endpoint | Calls Nebius's OpenAI-compatible embedding and chat APIs. |
| Project environment | uv | Dependency management and repeatable commands. |
| Observability | LangSmith (optional) | Traces the LangGraph retrieve → generate workflow when the LangSmith environment variables are enabled. |

The application does **not** currently use LCEL chains or a dedicated BM25 search framework. Hybrid retrieval is implemented directly in `rag_app.py` with dense Chroma retrieval plus local keyword scoring. LangSmith tracing is opt-in; formal evaluation datasets have not yet been created.

## 5. Architecture

```text
PMC XML files
    → XML body-text extraction and cleaning
    → semantic chunking
    → Nebius embeddings
    → local Chroma vector store
    → hybrid retrieval (dense + sparse)
    → LangGraph retrieve/generate flow
    → Nebius chat model
    → Streamlit chatbot with PMC links
```

## 6. Ingestion and Cleaning

`mnd_data_extract.py` queries PMC and downloads eligible open-access XML articles into `data/pmc_xml`. Ingestion is currently manual: after downloading or changing documents, the index is rebuilt.

`load_pmc_xml_documents()` in `rag_app.py` reads each XML file, extracts body text, removes XML markup through parsing, and normalizes repeated whitespace. The original XML file name and corresponding PMC article URL are retained as metadata for source links.

## 7. Chunking and Embeddings

### Baseline tested: recursive chunking

The initial baseline used `RecursiveCharacterTextSplitter` with `chunk_size=700` and `chunk_overlap=100`. Across the current 100 documents, this produced **6,389 chunks**.

This approach is fast and predictable, but can split a clinical finding or numerical result across an arbitrary character boundary.

### Current index: semantic chunking

The current index uses the in-project semantic splitter in `rag_app.py`. It embeds individual sentences, calculates cosine distance between adjacent sentence embeddings, and creates a boundary at unusually large meaning changes (the 95th-percentile distance within an article).

Semantic chunks have no artificial overlap and use a 3,500-character safety cap to prevent excessively large vector-store records. This produced **1,907 chunks** from the same 100 documents, generally preserving a complete topic or finding in each chunk.

### Embedding model

Nebius `Qwen/Qwen3-Embedding-8B` creates dense embeddings for document chunks and questions. The project uses a custom `NebiusEmbeddings` class because Nebius expects text strings rather than tokenized-input arrays.

## 8. Retrieval Design

### Vector store

Chroma stores each chunk's text, embedding, and metadata locally in `data/chroma_db`. Rebuilding the index replaces the current Chroma collection.

### Hybrid retrieval

The retriever returns five chunks to the answer-generation step. It combines:

1. **Dense retrieval**: Nebius embedding similarity identifies passages with similar meaning.
2. **Sparse retrieval**: local keyword scoring identifies exact terms, abbreviations, and technical phrases.
3. **Rank fusion**: the dense and sparse rankings are combined to select the strongest evidence.
4. **Keyword coverage safeguard**: the strongest sparse match is retained in the five excerpts if broad dense matches would otherwise displace it.

The system evaluates up to 100 dense candidates and 100 sparse candidates, then supplies the final **top-k = 5** excerpts to the chat model.

### Tested retrieval case

For the question about the spectral signal-to-noise ratio (SNR) in an ALS speech BCI study, dense retrieval located the correct article (`PMC13042181.xml`) but ranked the exact numeric passage 79th. The hybrid retriever recognized `signal + noise` as `SNR`, included the exact passage, and the chatbot correctly returned: approximately **1 dB to 6 dB**, declining to approximately **3 dB** by day 763.

## 9. Answer Generation

LangGraph controls a simple two-node flow:

```text
START → retrieve → generate → END
```

The generator receives only the five retrieved excerpts and is instructed to summarize them, avoid unsupported inferences, and state when the excerpts do not answer the question. The application appends deduplicated PMC web links after the answer instead of inline XML-file citations.

## 10. Interfaces and Operations

| Task | Command |
| --- | --- |
| Download articles | `python mnd_data_extract.py` |
| Preview recursive chunks | `uv run python rag_app.py preview-chunks --limit 5` |
| Preview semantic chunks | `uv run python rag_app.py preview-chunks --strategy semantic --limit 5` |
| Build semantic index | `uv run python rag_app.py index --strategy semantic` |
| Ask a question | `uv run python rag_app.py ask "What are early bulbar-onset symptoms?"` |
| Run dashboard | `uv run streamlit run streamlit_app.py` |

## 11. Evaluation Plan

There is no measured faithfulness or relevance percentage yet. Create a small evaluation set of 20–30 representative questions, including factual, numeric, terminology, negative/unanswerable, and source-specific questions.

When `LANGSMITH_TRACING=true` and a LangSmith API key are configured, each LangGraph request is traceable in the `als-mnd-knowledge-chatbot` LangSmith project. Use those traces to inspect retrieved chunks, prompt inputs, model output, and latency while evaluating the question set.

For each question, record the expected source and answer, then measure:

| Metric | Definition |
| --- | --- |
| Retrieval relevance | Whether at least one top-5 chunk contains the evidence needed for the answer |
| Faithfulness | Whether every material answer claim is supported by the retrieved excerpts |
| Answer correctness | Whether the response matches the reference answer |
| Source accuracy | Whether supplied PMC links support the answer |
| Latency | Time for retrieval and generation |

Compare recursive and semantic indexes using the same question set. The chosen strategy should be based on measured retrieval relevance and faithfulness, rather than chunk count alone.

## 12. Limitations and Next Steps

- The corpus is limited to 100 PMC articles and is manually refreshed.
- Hybrid sparse scoring is local and lightweight; BM25 or a dedicated search service could be evaluated for larger corpora.
- The application currently has no automated test set, scheduled ingestion, reranker, or user feedback loop. LangSmith tracing is available but requires a user-configured API key.
- Sources may discuss ALS specifically; the answer prompt should distinguish ALS-specific findings from evidence about all MND subtypes.
- Medical answers should remain literature summaries and direct users to qualified clinicians for personal medical advice.
