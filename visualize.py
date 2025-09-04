# -*- coding: utf-8 -*-

import os
import glob
import warnings
from typing import List, Tuple

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

DATA_DIR = "data"
EMBED_MODEL_PATH = "embedding_model/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250

CATEGORY_COLORS = {
    "company":   "#1f77b4",  # mavi
    "contracts": "#ff7f0e",  # turuncu
    "employees": "#2ca02c",  # yeşil
    "products":  "#d62728",  # kırmızı
}

def load_all_documents(data_dir: str) -> List[Document]:
    docs: List[Document] = []

    for pdf_path in glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True):
        try:
            pages = PyMuPDFLoader(pdf_path).load()
            for p in pages:
                p.metadata["source"] = pdf_path
            docs.extend(pages)
        except Exception as e:
            print(f"[WARN] PDF load failed: {pdf_path} -> {e}")

    for pattern in ("*.md", "*.markdown", "*.txt"):
        for txt_path in glob.glob(os.path.join(data_dir, "**", pattern), recursive=True):
            try:
                pages = TextLoader(txt_path, autodetect_encoding=True).load()
                for p in pages:
                    p.metadata["source"] = txt_path
                docs.extend(pages)
            except Exception as e:
                print(f"[WARN] Text load failed: {txt_path} -> {e}")

    print(f"Loaded documents: {len(docs)}")
    return docs


def infer_category_from_source(source_path: str, data_root: str) -> str:
    try:
        rel = os.path.relpath(source_path, data_root)
        top = rel.split(os.sep, 1)[0].lower()
        return top if top in CATEGORY_COLORS else "other"
    except Exception:
        return "other"


def build_in_memory_faiss(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks


def extract_vectors_and_meta(vs: FAISS, data_root: str) -> Tuple[np.ndarray, list, list, list]:

    vectors = []
    documents = []
    categories = []
    sources = []

    total = vs.index.ntotal
    if total == 0:
        return np.empty((0,)), [], [], []

    for i in range(total):
        vec = vs.index.reconstruct(i)
        vectors.append(vec)

        doc_id = vs.index_to_docstore_id[i]
        document = vs.docstore.search(doc_id)

        content = getattr(document, "page_content", "")
        src = (document.metadata or {}).get("source", "")

        cat = infer_category_from_source(src, data_root)

        documents.append(content)
        categories.append(cat)
        sources.append(src)

    return np.array(vectors), documents, categories, sources


def tsne_reduce(vectors: np.ndarray, dimensions: int = 2, random_state: int = 42) -> np.ndarray:
    if len(vectors) < 2:
        return np.empty((0, dimensions))
    perplexity = max(2, min(30, len(vectors) - 1))
    tsne = TSNE(n_components=dimensions, random_state=random_state, perplexity=perplexity, init="random")
    return tsne.fit_transform(vectors)


def make_plot(reduced: np.ndarray, documents: list, categories: list, sources: list, dimensions: int = 2):
    if reduced.size == 0:
        print("Not enough vectors to visualize.")
        return

    colors = [CATEGORY_COLORS.get(cat, "#7f7f7f") for cat in categories]  # other=grey

    def snip(t: str, n: int = 120) -> str:
        t = t.replace("\n", " ").strip()
        return (t[:n] + "…") if len(t) > n else t

    hover_texts = [
        f"Category: {cat}<br>File: {os.path.relpath(src, DATA_DIR)}<br>Text: {snip(doc)}"
        for cat, src, doc in zip(categories, sources, documents)
    ]

    if dimensions == 2:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers",
                    marker=dict(size=5, color=colors, opacity=0.85),
                    text=hover_texts,
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title="2D Vector Visualization (t-SNE)",
            width=900,
            height=650,
            margin=dict(r=20, b=20, l=20, t=50),
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=colors, opacity=0.85),
                    text=hover_texts,
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title="3D Vector Visualization (t-SNE)",
            width=1000,
            height=750,
            margin=dict(r=20, b=20, l=20, t=50),
        )

    legend_items = "<br>".join([f"<b>{k}</b>: <span style='color:{v}'>■</span>" for k, v in CATEGORY_COLORS.items()])
    fig.update_layout(
        title=f"{fig.layout.title.text}<br><sup>{legend_items}</sup>"
    )

    fig.show()


def visualize_data(dimensions: int = 2):

    docs = load_all_documents(DATA_DIR)
    if not docs:
        print("No documents found under 'data/'.")
        return

    vs, _ = build_in_memory_faiss(docs)
    vectors, documents, categories, sources = extract_vectors_and_meta(vs, DATA_DIR)

    if vectors.shape[0] < 2:
        print("Not enough vectors to visualize.")
        return

    reduced = tsne_reduce(vectors, dimensions=dimensions)
    make_plot(reduced, documents, categories, sources, dimensions=dimensions)


if __name__ == "__main__":
    #visualize_data(dimensions=2)
    visualize_data(dimensions=3)
