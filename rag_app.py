"""Index PMC XML articles and ask cited questions over them.

Usage:
    python rag_app.py index
    python rag_app.py preview-chunks --limit 5
    python rag_app.py preview-chunks --strategy semantic --limit 5
    python rag_app.py index --strategy semantic
    python rag_app.py ask "What are common bulbar-onset symptoms?"
"""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import re
from typing import TypedDict
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

load_dotenv()


DATA_DIR = Path("data/pmc_xml")
VECTOR_DIR = "data/chroma_db"
COLLECTION_NAME = "mnd_pmc_articles"
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
EMBEDDING_MODEL = os.getenv("NEBIUS_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
CHAT_MODEL = os.getenv("NEBIUS_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
CHROMA_BATCH_SIZE = 1000
RECURSIVE_CHUNK_SIZE = 700
RECURSIVE_CHUNK_OVERLAP = 100
SEMANTIC_BREAKPOINT_PERCENTILE = 95
SEMANTIC_MAX_CHUNK_SIZE = 3500
DENSE_CANDIDATE_COUNT = 100
SPARSE_CANDIDATE_COUNT = 100
RETRIEVAL_TOP_K = 5
SPARSE_RRF_WEIGHT = 2
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "what", "when",
    "where", "which", "with",
}


def nebius_api_key() -> str:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NEBIUS_API_KEY. Add it to .env before indexing or asking a question."
        )
    return api_key


class NebiusEmbeddings(Embeddings):
    """Embed text with Nebius without sending token-ID arrays to its API."""

    def __init__(self, batch_size: int = 64) -> None:
        self.client = OpenAI(api_key=nebius_api_key(), base_url=NEBIUS_BASE_URL)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_pmc_xml_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Extract readable body text from each downloaded PMC XML article."""
    documents = []
    for xml_path in sorted(data_dir.glob("PMC*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
            body = root.find(".//body")
            text = " ".join((body if body is not None else root).itertext())
            text = " ".join(text.split())
            if text:
                documents.append(Document(text, metadata={
                    "source": xml_path.name,
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{xml_path.stem}/",
                }))
        except ET.ParseError as exc:
            print(f"Skipping invalid XML {xml_path.name}: {exc}")
    return documents


def vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DIR,
        embedding_function=NebiusEmbeddings(),
    )


def percentile(values: list[float], percentage: int) -> float:
    """Return a percentile without adding a numerical-computing dependency."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cosine_distance(first: list[float], second: list[float]) -> float:
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return 1 - dot_product / (first_norm * second_norm)


def semantic_split_document(document: Document, embeddings: NebiusEmbeddings) -> list[Document]:
    """Split at sentence boundaries where adjacent sentence meaning changes."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", document.page_content)
        if sentence.strip()
    ]
    if len(sentences) < 3:
        return [document]

    vectors = embeddings.embed_documents(sentences)
    distances = [
        cosine_distance(vectors[index], vectors[index + 1])
        for index in range(len(vectors) - 1)
    ]
    threshold = percentile(distances, SEMANTIC_BREAKPOINT_PERCENTILE)

    chunks: list[Document] = []
    current_sentences: list[str] = []
    for index, sentence in enumerate(sentences):
        current_sentences.append(sentence)
        reaches_semantic_boundary = (
            index < len(distances) and distances[index] >= threshold
        )
        reaches_safety_limit = len(" ".join(current_sentences)) >= SEMANTIC_MAX_CHUNK_SIZE
        if reaches_semantic_boundary or reaches_safety_limit:
            chunks.append(Document(" ".join(current_sentences), metadata=document.metadata.copy()))
            current_sentences = []

    if current_sentences:
        chunks.append(Document(" ".join(current_sentences), metadata=document.metadata.copy()))
    return chunks


def chunk_documents(documents: list[Document], strategy: str) -> list[Document]:
    """Split documents using a repeatable baseline or semantic boundaries."""
    if strategy == "semantic":
        embeddings = NebiusEmbeddings()
        chunks: list[Document] = []
        for index, document in enumerate(documents, start=1):
            chunks.extend(semantic_split_document(document, embeddings))
            if index % 5 == 0 or index == len(documents):
                print(f"Prepared semantic chunks for {index}/{len(documents)} articles")
        return chunks
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RECURSIVE_CHUNK_SIZE,
            chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        )
        return splitter.split_documents(documents)


def index_documents(strategy: str) -> None:
    documents = load_pmc_xml_documents()
    if not documents:
        raise SystemExit(f"No PMC XML files found in {DATA_DIR.resolve()}")

    chunks = chunk_documents(documents, strategy)
    store = vector_store()
    try:
        store.delete_collection()
        store = vector_store()
    except ValueError:
        pass
    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        batch = chunks[start : start + CHROMA_BATCH_SIZE]
        store.add_documents(batch)
        completed = min(start + len(batch), len(chunks))
        print(f"Indexed {completed}/{len(chunks)} chunks")
    print(
        f"Indexed {len(chunks)} {strategy} chunks from {len(documents)} PMC articles "
        f"into {VECTOR_DIR}."
    )


def preview_chunks(limit: int, strategy: str) -> None:
    """Print sample chunks. Semantic previews call Nebius to find boundaries."""
    documents = load_pmc_xml_documents()
    if not documents:
        raise SystemExit(f"No PMC XML files found in {DATA_DIR.resolve()}")
    chunks = chunk_documents(documents, strategy)

    print(f"Documents: {len(documents)}")
    print(f"Strategy: {strategy}")
    print(f"Chunks: {len(chunks)}\n")
    for index, chunk in enumerate(chunks[:limit], start=1):
        print(f"--- Chunk {index} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Characters: {len(chunk.page_content)}")
        print(chunk.page_content)
        print()


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str


def retrieval_terms(question: str) -> set[str]:
    """Return meaningful query terms, including common scientific abbreviations."""
    terms = {
        token for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in STOP_WORDS and len(token) > 1
    }
    if {"signal", "noise"}.issubset(terms):
        terms.add("snr")
    if "snr" in terms:
        terms.update({"signal", "noise", "ratio"})
    return terms


def sparse_score(
    document: Document, terms: set[str], inverse_document_frequency: dict[str, float]
) -> float:
    """A small local keyword score used alongside vector similarity."""
    tokens = re.findall(r"[a-z0-9]+", document.page_content.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    matches = sum(counts[term] * inverse_document_frequency[term] for term in terms)
    return matches / math.sqrt(len(tokens))


def document_key(document: Document) -> str:
    return f"{document.metadata.get('source', '')}\0{document.page_content}"


def hybrid_search(question: str, k: int = RETRIEVAL_TOP_K) -> list[Document]:
    """Fuse meaning-based retrieval with exact-term retrieval using rank fusion."""
    store = vector_store()
    dense = store.similarity_search_with_relevance_scores(
        question, k=DENSE_CANDIDATE_COUNT
    )
    stored = store.get(include=["documents", "metadatas"])
    all_documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored["documents"], stored["metadatas"])
        if text
    ]
    terms = retrieval_terms(question)
    document_frequency = Counter()
    for document in all_documents:
        present_terms = set(re.findall(r"[a-z0-9]+", document.page_content.lower()))
        document_frequency.update(terms.intersection(present_terms))
    inverse_document_frequency = {
        term: math.log((len(all_documents) + 1) / (document_frequency[term] + 1)) + 1
        for term in terms
    }
    sparse = sorted(
        (
            (document, sparse_score(document, terms, inverse_document_frequency))
            for document in all_documents
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:SPARSE_CANDIDATE_COUNT]

    candidates: dict[str, tuple[Document, float]] = {}
    for rank, (document, _) in enumerate(dense, start=1):
        key = document_key(document)
        candidates[key] = (document, candidates.get(key, (document, 0.0))[1] + 1 / (60 + rank))
    for rank, (document, score) in enumerate(sparse, start=1):
        if score == 0:
            break
        key = document_key(document)
        candidates[key] = (
            document,
            candidates.get(key, (document, 0.0))[1]
            + SPARSE_RRF_WEIGHT / (60 + rank),
        )

    ranked_documents = [
        document
        for document, _ in sorted(candidates.values(), key=lambda item: item[1], reverse=True)[:k]
    ]
    # Preserve the best exact-term result so a precise technical answer is not
    # displaced by several broadly similar dense matches.
    if sparse and sparse[0][1] > 0:
        strongest_sparse_match = sparse[0][0]
        if document_key(strongest_sparse_match) not in {
            document_key(document) for document in ranked_documents
        }:
            ranked_documents = ranked_documents[: k - 1] + [strongest_sparse_match]
    return ranked_documents


def retrieve(state: RAGState) -> dict:
    docs = hybrid_search(state["question"])
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    context = "\n\n".join(
        f"Source: {doc.metadata['source']}\n{doc.page_content}"
        for doc in state["documents"]
    )
    messages = [
        SystemMessage(
            content=(
                "Answer only from the supplied PMC article excerpts. Give a clear, "
                "concise summary of what the excerpts say. Do not include inline "
                "citations, reference lists, or source filenames; source links are "
                "added separately. If the excerpts do not answer the question, say so. "
                "Do not make inferences or claim a cause, relationship, or conclusion "
                "that is not explicitly stated in the excerpts. "
                "This is research support, not medical advice."
            )
        ),
        HumanMessage(content=f"Question: {state['question']}\n\nExcerpts:\n{context}"),
    ]
    summary = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=nebius_api_key(),
        base_url=NEBIUS_BASE_URL,
    ).invoke(messages).content
    sources = {}
    for document in state["documents"]:
        source_name = document.metadata["source"]
        sources[source_name] = document.metadata.get("source_url")
    source_links = "\n".join(
        f"- [{source_name}]({url})" if url else f"- {source_name} (uploaded file)"
        for source_name, url in sorted(sources.items())
    )
    answer = f"{summary}\n\n### Read the source material\n{source_links}"
    return {"answer": answer}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(question: str) -> None:
    load_dotenv()
    result = build_graph().invoke({"question": question})
    print(result["answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG over PMC MND articles")
    parser.add_argument("command", choices=["index", "preview-chunks", "ask"])
    parser.add_argument("question", nargs="?", help="Question to ask (required for ask)")
    parser.add_argument("--limit", type=int, default=5, help="Number of chunks to preview")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic"],
        default="recursive",
        help="Chunking strategy for index or preview (default: recursive)",
    )
    args = parser.parse_args()

    if args.command == "index":
        index_documents(args.strategy)
    elif args.command == "preview-chunks":
        preview_chunks(args.limit, args.strategy)
    elif not args.question:
        parser.error("ask requires a question")
    else:
        ask(args.question)
