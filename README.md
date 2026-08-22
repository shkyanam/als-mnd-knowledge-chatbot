# Bulbar MND research dashboard and RAG

This project downloads open-access PMC XML articles, creates a local Chroma vector index using Nebius embeddings, and provides a Streamlit dashboard and cited literature chatbot.

## Setup

1. Create the uv-managed project environment:

   ```bash
   uv sync
   ```

2. Create `.env` from `.env.example`, then add your Nebius API key. Do not put the key in Python files or commit `.env`. The default endpoint and models use Nebius Token Factory's OpenAI-compatible API; adjust the model names if your Nebius account provides different ones.

3. Download the articles:

   ```bash
   python mnd_data_extract.py
   ```

   XML files are saved in `data/pmc_xml`.

4. Build the local vector index:

   ```bash
   uv run python rag_app.py index
   ```

   To inspect the recursive chunks first, without calling Nebius:

   ```bash
   uv run python rag_app.py preview-chunks --limit 5
   ```

   To compare semantic chunks, which uses Nebius embeddings to find likely topic
   boundaries, run:

   ```bash
   uv run python rag_app.py preview-chunks --strategy semantic --limit 5
   ```

   Semantic chunks have a 3,500-character safety limit, but do not use the
   recursive splitter's fixed size or overlap settings.

   If the semantic results are better for your questions, rebuild the index with
   that same strategy:

   ```bash
   uv run python rag_app.py index --strategy semantic
   ```

5. Ask a question:

   ```bash
   uv run python rag_app.py ask "What clinical features are reported for bulbar-onset MND?"
   ```

The answer is restricted to retrieved text, summarized without inline citations, and ends with clickable PubMed Central article links. It is for literature research only, not clinical decision-making.

Retrieval is hybrid: Nebius dense-vector similarity is combined with a local keyword score. This helps precise terms, abbreviations, and numeric technical questions while retaining semantic matching.


## Streamlit dashboard

Start the application after indexing:

```bash
uv run streamlit run streamlit_app.py
```

The **Patient impact** tab presents MND deaths and disability-burden estimates from a prepared IHME GBD export, with filters for measure, region, age group, sex, year, and COVID era.

The prepared data uses these columns:

```text
region,age_group,sex,year,measure,value,source
```

Save the prepared export as `data/mnd_burden.csv`. The app does not estimate burden from the PMC papers. The **Literature chatbot** tab answers questions using the indexed corpus and cites its XML sources.

### Loading public MND estimates from IHME GBD

The dashboard uses an export from the [IHME GBD Results tool](https://vizhub.healthdata.org/gbd-results/). Select **Motor neuron disease**, **Number**, the desired measures (for example Deaths and DALYs), locations, age groups, sexes, and years. Convert it once for direct in-app use:

```bash
uv run python prepare_gbd_data.py IHME-GBD_2021_DATA.csv
```

This preserves IHME's estimated prevalence counts and records the source in the dashboard. Use “COVID era and later” for 2020 onward; this is not necessarily the same as the post-pandemic period.
