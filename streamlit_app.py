"""Streamlit dashboard and cited RAG chat for the bulbar MND corpus."""

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from prepare_gbd_data import normalize_mnd_burden
from rag_app import build_graph


DATASET_PATH = Path("data/mnd_burden.csv")
REQUIRED_COLUMNS = {"region", "age_group", "sex", "year", "measure", "value", "source"}


st.set_page_config(page_title="Bulbar MND Research", page_icon="🧠", layout="wide")
load_dotenv()


@st.cache_data(show_spinner="Loading MND burden estimates...")
def read_mnd_burden() -> pd.DataFrame | None:
    """Load the prepared IHME MND export stored with this project."""
    try:
        if not DATASET_PATH.exists():
            return None
        frame = pd.read_csv(DATASET_PATH)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not read MND burden data: {error}")
        return None

    if not REQUIRED_COLUMNS.issubset(frame.columns):
        try:
            frame = normalize_mnd_burden(frame)
        except ValueError:
            pass
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        st.error(f"Patient-impact data is missing columns: {', '.join(sorted(missing))}")
        return None
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["region", "age_group", "sex", "year", "measure", "value"])


def filter_mnd_burden(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        st.header("MND burden filters")
        measures = sorted(frame["measure"].unique())
        measure = st.selectbox("Measure", measures, index=measures.index("Deaths") if "Deaths" in measures else 0)
        regions = st.multiselect("Region", sorted(frame["region"].unique()), default=sorted(frame["region"].unique()))
        ages = sorted(frame["age_group"].unique())
        default_ages = ["All ages"] if "All ages" in ages else ages
        age_groups = st.multiselect("Age group", ages, default=default_ages, help="Use All ages alone for totals; do not combine it with overlapping age bands.")
        sexes = st.multiselect("Sex", sorted(frame["sex"].unique()), default=sorted(frame["sex"].unique()))
        minimum_year, maximum_year = int(frame["year"].min()), int(frame["year"].max())
        years = st.slider("Year range", minimum_year, maximum_year, (minimum_year, maximum_year))
        period = st.radio("COVID comparison", ["All available years", "Before COVID (≤ 2019)", "COVID era and later (≥ 2020)"])

    filtered = frame[
        (frame["measure"] == measure)
        & frame["region"].isin(regions)
        & frame["age_group"].isin(age_groups)
        & frame["sex"].isin(sexes)
        & frame["year"].between(*years)
    ]
    if period.startswith("Before"):
        filtered = filtered[filtered["year"] <= 2019]
    elif period.startswith("COVID"):
        filtered = filtered[filtered["year"] >= 2020]
    return filtered, measure


@st.cache_resource(show_spinner=False)
def rag_graph():
    return build_graph()


def render_dashboard() -> None:
    st.title("MND deaths and disability burden")
    st.caption("IHME GBD 2023 estimates for Motor neuron disease. Counts are not inferred from the PMC literature.")
    frame = read_mnd_burden()
    if frame is None:
        st.info("The prepared IHME MND burden dataset was not found. Run prepare_gbd_data.py first.")
        return

    if frame["year"].min() >= 2020:
        st.info("This export begins in 2020, so it does not contain before-COVID data. Add a 2018–2019 export for a true before/after comparison.")
    filtered, measure = filter_mnd_burden(frame)
    total, records, regions = st.columns(3)
    total.metric(measure, f"{filtered['value'].sum():,.0f}")
    records.metric("Data records", f"{len(filtered):,}")
    regions.metric("Regions", filtered["region"].nunique())

    if filtered.empty:
        st.warning("No records match the current filters.")
        return

    yearly = filtered.groupby("year", as_index=False)["value"].sum().sort_values("year")
    by_region = filtered.groupby("region", as_index=False)["value"].sum().sort_values("value", ascending=False)
    first, second = st.columns(2)
    first.subheader(f"{measure} by year")
    first.line_chart(yearly, x="year", y="value")
    second.subheader(f"{measure} by region")
    second.bar_chart(by_region, x="region", y="value")
    st.subheader("Filtered records")
    st.dataframe(filtered.sort_values(["year", "region"]), use_container_width=True, hide_index=True)


def render_chat() -> None:
    st.title("Ask the bulbar MND literature")
    st.caption("Answers summarize retrieved PMC articles and link to the full source papers. Not medical advice.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a research question about bulbar MND"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving article evidence..."):
                try:
                    result = rag_graph().invoke({"question": question})
                    answer = result["answer"]
                except Exception as error:
                    answer = f"Unable to answer: {error}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


dashboard_tab, chatbot_tab = st.tabs(["Patient impact", "Literature chatbot"])
with dashboard_tab:
    render_dashboard()
with chatbot_tab:
    render_chat()
