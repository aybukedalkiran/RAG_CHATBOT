# -*- coding: utf-8 -*-

import os
import glob
import json
import pickle
import logging
import warnings
import tempfile
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List

import gradio as gr

from langchain.schema import Document, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.llms import OllamaLLM
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader

# ----------------------------------------------------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

try:
    from gradio.chat_interface import ChatInterface as _CI
    _orig_msg2dict = _CI._message_as_message_dict

    def _safe_msg2dict(self, msg, role):
        if msg is None:
            return []
        if isinstance(msg, list):
            out = []
            for m in msg:
                if m is None:
                    continue
                if not isinstance(m, dict):
                    m = {"text": str(m), "files": []}
                else:
                    m = {"text": m.get("text") or "", "files": m.get("files") or []}
                out.extend(_orig_msg2dict(self, m, role))
            return out
        if not isinstance(msg, dict):
            msg = {"text": str(msg), "files": []}
        else:
            msg = {"text": msg.get("text") or "", "files": msg.get("files") or []}
        return _orig_msg2dict(self, msg, role)

    _CI._message_as_message_dict = _safe_msg2dict
except Exception as e:
    logging.getLogger(__name__).warning(f"Gradio hotfix failed: {e}")

# ----------------------------------------------------------------------------------------------------------------------

MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:latest")
DATA_DIR = "data"
EMBED_MODEL_PATH = "embedding_model/"
INDEX_DIR = "vec_index/all"
BM25_PATH = "bm25_retriever/all.pkl"

HIST_DIR = "history_all"
LOG_DIR  = "logs_all"
LOG_FILE = "chat.jsonl"

os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)

rot = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8")
rot.setFormatter(logging.Formatter("%(message)s"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = [rot]
logger.propagate = False

def _hist_path(session_id: str) -> str:
    return os.path.join(HIST_DIR, f"{session_id}.json")

def load_history_from_disk(session_id: str):
    p = _hist_path(session_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _atomic_write_json_array(path: str, arr):
    d = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False, encoding="utf-8") as tf:
        json.dump(arr, tf, ensure_ascii=False, indent=2)
        tf.flush(); os.fsync(tf.fileno())
        tmp = tf.name
    os.replace(tmp, path)

def save_history_to_disk(session_id: str, history):
    _atomic_write_json_array(_hist_path(session_id), history)

def append_session_log(session_id: str, entry: dict):
    p = os.path.join(LOG_DIR, f"{session_id}.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
        except Exception:
            logs = []
    else:
        logs = []
    logs.append(entry)
    _atomic_write_json_array(p, logs)

def log_global_json(entry: dict):
    logger.info(json.dumps(entry, ensure_ascii=False))

def log_interaction(session_id: str, user_question: str, answer_text: str, phase: str="final", error: str|None=None):
    entry = {
        "ts": datetime.now().astimezone().isoformat(),
        "session_id": session_id,
        "phase": phase,
        "user": user_question,
        "answer": answer_text,
        "error": error,
    }
    log_global_json(entry)
    append_session_log(session_id, entry)

# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=250, separators=["\n\n", "\n", ".", " ", ""]
)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)

def build_or_load_indices(docs: List[Document]):

    if os.path.exists(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, "index.pkl")):
        print("Loading FAISS index...")
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating FAISS index...")
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(INDEX_DIR)

    if os.path.exists(BM25_PATH):
        print("Loading BM25 retriever...")
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
    else:
        print("Creating BM25 retriever...")
        chunks = splitter.split_documents(docs)
        bm25 = BM25Retriever.from_documents(chunks)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

    dense = vs.as_retriever(search_kwargs={"k": 20})
    bm25.k = 20
    return dense, bm25


class SimpleHybridRetriever(BaseRetriever):
    dense: BaseRetriever = Field(...)
    sparse: BaseRetriever = Field(...)
    k: int = Field(default=10)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        d_docs = self.dense.get_relevant_documents(query)
        s_docs = self.sparse.get_relevant_documents(query)

        unique, seen = [], set()
        di, si, turn_dense = 0, 0, True

        while len(unique) < self.k and (di < len(d_docs) or si < len(s_docs)):
            if turn_dense and di < len(d_docs):
                cand = d_docs[di]; di += 1
            elif (not turn_dense) and si < len(s_docs):
                cand = s_docs[si]; si += 1
            else:
                if di < len(d_docs):
                    cand = d_docs[di]; di += 1
                elif si < len(s_docs):
                    cand = s_docs[si]; si += 1
                else:
                    break
            key = cand.page_content
            if key not in seen:
                seen.add(key)
                unique.append(cand)
            turn_dense = not turn_dense

        return unique

# ----------------------------------------------------------------------------------------------------------------------
llm = OllamaLLM(model=MODEL)

_all_docs = load_all_documents(DATA_DIR)
dense_ret, sparse_ret = build_or_load_indices(_all_docs)
hybrid = SimpleHybridRetriever(dense=dense_ret, sparse=sparse_ret, k=20)

# ----------------------------------------------------------------------------------------------------------------------

def chat(message, history, request: gr.Request):
    session_id = getattr(request, "session_hash", None) or "anonymous"
    disk_history = load_history_from_disk(session_id)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    trimmed = disk_history[-6:] if len(disk_history) > 6 else disk_history
    for msg in trimmed:
        if msg.get("role") == "user":
            memory.chat_memory.add_message(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            memory.chat_memory.add_message(AIMessage(content=msg.get("content", "")))

    try:
        if message is None:
            yield "Hello! How can I help you today?"
            return
        user_question = message.get("text") if isinstance(message, dict) else str(message or "")
        if not (user_question or "").strip():
            yield "Hello! How can I help you today?"
            return

        question = (
            "Answer in English. Use only the provided documents. "
            "If the answer is not in the retrieved documents, say you couldn't find it succinctly. "
            + user_question
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=hybrid,
            memory=memory,
            callbacks=[StdOutCallbackHandler()],
        )
        out = chain.invoke({"question": question})
        answer = out.get("answer", "").strip()

        memory.chat_memory.add_message(HumanMessage(content=user_question))
        memory.chat_memory.add_message(AIMessage(content=answer))
        new_hist = disk_history + [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": answer},
        ]
        save_history_to_disk(session_id, new_hist)

        log_interaction(session_id=session_id, user_question=user_question, answer_text=answer, phase="final", error=None)

        yield answer
        return

    except Exception as e:
        err = f"Error: {e}"
        try:
            log_interaction(session_id=session_id, user_question=user_question, answer_text="", phase="error", error=str(e))
        except Exception:
            pass
        new_hist = disk_history + [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": err},
        ]
        save_history_to_disk(session_id, new_hist)
        yield err
        return

# ----------------------------------------------------------------------------------------------------------------------
def _welcome():
    return [{"role": "assistant", "content": "Hello! How can I help you today?"}]

with gr.Blocks() as view:
    chatbot = gr.Chatbot(type="messages", height=525)
    textbox = gr.MultimodalTextbox(
        value={"text": "", "files": []},
        file_count="multiple",
        placeholder="Type your message…"
    )

    gr.ChatInterface(
        fn=chat,
        type="messages",
        chatbot=chatbot,
        textbox=textbox,
        concurrency_limit=20
    )
    view.load(_welcome, outputs=chatbot)

view.queue(
    default_concurrency_limit=20,
    max_size=100
    ).launch(server_name="127.0.0.1", server_port=7860, show_api=True)
