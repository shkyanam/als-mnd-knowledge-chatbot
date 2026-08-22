# ALS/MND Knowledge Chatbot — Project Workflow Report

## 1. Project Overview

This project is a research-oriented retrieval-augmented generation (RAG) application for questions about motor neuron disease (MND), amyotrophic lateral sclerosis (ALS), and bulbar-onset disease. It provides a Streamlit dashboard with a literature chatbot, plus command-line tools for downloading articles, inspecting chunks, rebuilding the index, and asking questions.

The goal is to make a small, transparent research assistant that grounds answers in retrieved open-access literature and links readers back to the original PubMed Central (PMC) articles. It is not intended for diagnosis, treatment recommendations, or personal medical advice.

## 2. What Was Built

| Area | Implementation |
| --- | --- |
| User interface | Streamlit dashboard with an MND burden tab and a literature chatbot tab. |
| Literature corpus | 100 downloaded open-access PMC articles related to MND, ALS, and bulbar disease. |
| Ingestion | `mnd_data_extract.py` searches PMC and stores XML articles in `data/pmc_xml`. |
| Document preparation | `rag_app.py` parses XML, extracts body text, normalizes whitespace, and stores PMC file/URL metadata. |
| Chunking | Semantic sentence-boundary chunking is used in the current index. |
| Embeddings | Nebius `Qwen/Qwen3-Embedding-8B`, accessed through Nebius's OpenAI-compatible API. |
| Vector store | Local Chroma database in `data/chroma_db`. |
| Retrieval | Hybrid retrieval: dense-vector similarity plus local sparse keyword scoring. |
| Orchestration | LangGraph retrieve → generate workflow. |
| Generation | Nebius-hosted `meta-llama/Llama-3.3-70B-Instruct` chat model. |
| Public-health dashboard data | A prepared IHME GBD Motor Neuron Disease export in `data/mnd_burden.csv`, used separately from the literature corpus. |

## 3. Datasets Used

### 3.1 PMC literature dataset

The RAG corpus contains 100 open-access PMC XML articles, primarily in English. The local XML files are the working copy; PMC is the source of truth. Each indexed chunk keeps its file name and a URL such as `https://pmc.ncbi.nlm.nih.gov/articles/PMCxxxxxxx/` so the final chatbot answer can link to the source article.

The corpus was selected for relevance to MND/ALS and bulbar-onset concepts. Older or irrelevant downloads were moved to `data/review_excluded` and are not indexed.

### 3.2 IHME GBD burden dataset

The Streamlit dashboard separately uses an IHME Global Burden of Disease (GBD) export filtered for Motor Neuron Disease. `prepare_gbd_data.py` converts the downloaded export into a compact local file with region, age group, sex, year, measure, and value fields.

This dataset supports aggregate burden visualizations such as deaths and DALYs. It does not provide patient-level information, and it is not used by the literature chatbot unless a future data-query tool is added.

## 4. Prompt and Agent Instructions

### 4.1 Retrieval workflow

The LangGraph workflow is deliberately small:

```text
START → retrieve → generate → END
```

The `retrieve` step selects five evidence chunks. It combines semantic vector similarity with local exact-term scoring before the language model receives any context.

### 4.2 Answer-generation system instruction

The generator is instructed to follow this behavior:

> Answer only from the supplied PMC article excerpts. Give a clear, concise summary of what the excerpts say. Do not include inline citations, reference lists, or source filenames; source links are added separately. If the excerpts do not answer the question, say so. Do not make inferences or claim a cause, relationship, or conclusion that is not explicitly stated in the excerpts. This is research support, not medical advice.

After generation, the application appends deduplicated clickable PMC article links. This approach keeps the main answer readable while allowing users to inspect the original research.

### 4.3 Scope instructions used in the design

- The chatbot should distinguish ALS-specific findings from claims applicable to all MND subtypes.
- The chatbot should not invent an answer when retrieved excerpts lack evidence.
- The application should summarize evidence rather than provide clinical advice.
- Retrieved sources should be shown as PMC web links rather than internal XML references alone.

## 5. Iterations Tried

### 5.1 Initial ingestion and corpus cleanup

The project first downloaded PMC XML articles using an MND/bulbar-focused query. Some downloaded files were found to be irrelevant, so they were moved out of the active corpus rather than being indexed. This reinforced the importance of verifying source relevance before measuring RAG quality.

### 5.2 Recursive chunking baseline

The first chunking strategy was `RecursiveCharacterTextSplitter` with:

```python
chunk_size = 700
chunk_overlap = 100
```

This created **6,389 chunks** from the 100 articles. It is fast, easy to reason about, and offers predictable chunk lengths, but a finding can be separated from its qualifying context at a character boundary.

### 5.3 LangChain experimental semantic chunking

An initial semantic-chunking attempt used LangChain's `SemanticChunker`. The package emitted a deprecation warning because `langchain-experimental` was archived/sunset. The approach was therefore replaced rather than keeping an unmaintained dependency in the project.

### 5.4 In-project semantic chunking

The current semantic splitter embeds sentences, computes cosine distance between adjacent sentence embeddings, and creates a new chunk at large semantic shifts (the 95th-percentile distance within each document). A 3,500-character safety limit prevents unusually large chunks.

The semantic index produced **1,907 chunks** from the same 100 articles. It reduced fragmentation and generally kept a topic or result together, although the chunks are less uniform in length than recursive chunks.

### 5.5 Dense retrieval only

The first retriever used only Chroma dense-vector similarity with `top-k = 5`. Dense retrieval was useful for meaning and synonyms—for example, it could relate MND and ALS terminology—but it could miss a precise numeric or technical result.

### 5.6 Hybrid retrieval

An SNR question exposed this limitation. The correct paper, `PMC13042181`, was retrieved by dense search, but the exact passage containing the numerical answer ranked 79th and was not supplied to the chat model.

Hybrid retrieval was then added directly in `rag_app.py`:

1. Retrieve up to 100 dense Chroma candidates.
2. Score all local chunks using important query terms and inverse-document-frequency weighting.
3. Expand `signal + noise` into the abbreviation `SNR`.
4. Fuse dense and sparse ranks.
5. Preserve the strongest exact-keyword chunk in the final five excerpts.

After this change, the chatbot correctly answered that active-period SNR rose from approximately **1 dB to 6 dB**, then declined to approximately **3 dB** by day 763.

## 6. Learnings and Observations

1. **Chunk count is not a quality metric by itself.** Semantic chunking reduced the index from 6,389 to 1,907 chunks, but evaluation should compare evidence retrieval and answer quality, not simply choose the smaller number.

2. **Semantic chunking and semantic retrieval are different.** Chunking decides how source text is divided; retrieval decides which chunks are selected for a question. Improving one does not automatically fix the other.

3. **Dense retrieval is strong for meaning but weaker for exact technical facts.** The SNR test showed that dense embeddings can retrieve the correct article yet miss the specific result passage needed to answer a numeric question.

4. **Hybrid retrieval improves precision.** Sparse keyword matching complements dense similarity for acronyms, named measures, numbers, and exact domain language.

5. **Source-level links improve transparency.** Returning PMC URLs lets a user verify the summarized research without cluttering the answer with raw XML file names.

6. **Corpus quality matters before model tuning.** Removing irrelevant downloads improves retrieval quality more reliably than only changing prompts or embedding models.

7. **MND and ALS terminology needs careful wording.** ALS is the most common adult form of MND, but they are not technically identical terms. The assistant should say when evidence is specifically ALS-focused.

8. **Public-health data and literature evidence should remain separate.** The IHME burden dashboard answers aggregate burden questions; the PMC RAG corpus answers literature questions. Joining them requires an explicit data-query layer rather than treating papers as structured epidemiology data.

## 7. Current Limitations

- The corpus is manually refreshed and limited to 100 articles.
- No formal faithfulness, retrieval-relevance, or answer-correctness benchmark has been run yet.
- Hybrid sparse retrieval is a lightweight in-project implementation, not a full BM25 or production search service.
- There is no reranker, feedback capture, scheduled ingestion, LangSmith tracing, or automated regression suite.
- The chatbot must remain a research aid and should not be used for clinical decisions.

## 8. Recommended Next Steps

1. Create 20–30 evaluation questions with expected evidence sources and answers.
2. Compare recursive and semantic indexes using retrieval relevance, answer correctness, faithfulness, and latency.
3. Add tests for known questions such as the SNR example so future retrieval changes do not reintroduce the issue.
4. Add a user-visible note when retrieved evidence is ALS-specific rather than applicable to all MND categories.
5. Decide whether to adopt a standard BM25 retriever and/or reranker if the corpus grows beyond the current scale.
