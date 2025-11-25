#!/usr/bin/env python3
# Shebang for cross-platform execution (Windows ignores but it's fine)

"""
Improved_RAG_Local_Windows.py
"""
# Module docstring describing file purpose

import os
# OS module for filesystem operations
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

'''
0 = all logs
1 = filter INFO
2 = filter INFO & WARNING (only ERROR)
3 = only FATAL
'''
import sys
# System module for interpreter interactions

import json
# JSON for metadata and simple persistence

import time
# Time for performance measurements

import argparse
# Argparse for CLI

import glob
# Glob for file discovery

import sqlite3
# SQLite for embedding and metadata caching

import multiprocessing as mp
# Multiprocessing for parallel embedding & extraction

from pathlib import Path
# Pathlib for path manipulation convenience

from typing import List, Tuple, Optional, Dict, Any
# Typing hints

import numpy as np
# NumPy for numeric arrays

import faiss
# FAISS for vector index (faiss-cpu recommended on Windows)
# Note: on Windows use faiss-cpu wheel

import torch
# Torch for device detection and fallback

from sentence_transformers import SentenceTransformer
# SentenceTransformers for embeddings

import fitz
# PyMuPDF for fast PDF extraction
import re

import tiktoken
# tiktoken for token counting and chunking by tokens

# Optional imports that improve functionality if present (non-fatal)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False
# PyArrow for optional fast metadata storage

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
# psutil for system resource sampling if available

# -----------------------------
# Configuration (tweak for Windows)
# -----------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
# Default to a fast MiniLM model; replace with local higher-quality if available

LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3")
# Ollama LLM model name used for generation

INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", "rag_index"))
# Directory to store index and caches

EMBED_CACHE_DB = INDEX_DIR / "embed_cache.sqlite"
# SQLite DB to cache embeddings

METADATA_JSON = INDEX_DIR / "metadata.jsonl"
# Line-delimited metadata for incremental indexing

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
# Path for FAISS index file

CHUNK_TOKEN_SIZE = int(os.environ.get("CHUNK_TOKEN_SIZE", "500"))
# Chunk size measured in tokens (more robust than chars)

CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "64"))
# Token overlap between adjacent chunks

EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))
# Embedding batch size

USE_GPU = torch.cuda.is_available()
# Detect GPU availability

DEVICE = "cuda" if USE_GPU else "cpu"
# Device string for models

FAISS_INDEX_TYPE = os.environ.get("FAISS_INDEX_TYPE", "HNSW")  # Options: FLAT, IVF, HNSW
# Default to HNSW for fast recall & low latency on CPU

N_M = 32
# HNSW M parameter (connectivity)

EF_CONSTRUCTION = 200
# HNSW efConstruction param

EF_SEARCH = 50
# HNSW efSearch runtime parameter

# Ensure index dir exists
INDEX_DIR.mkdir(parents=True, exist_ok=True)
# Create index directory if missing

# -----------------------------
# Utility: Tokenization and Chunking
# -----------------------------
_tokenizer_cache: Dict[str, Any] = {}
# Cache for tiktoken encoders

def get_tokenizer(model_name: str = "cl100k_base"):
    """
    Get tiktoken tokenizer for a model or fallback to cl100k_base.
    """
    # Use tiktoken to count tokens; cache encoder
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    _tokenizer_cache[model_name] = enc
    return enc
# Function caches tokenizer

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    Count tokens safely.
    """
    enc = get_tokenizer(model_name)
    return len(enc.encode(text))
# Returns token length

def chunk_by_tokens(text: str, chunk_size: int = CHUNK_TOKEN_SIZE, overlap: int = CHUNK_OVERLAP_TOKENS, model_name: str = "cl100k_base") -> List[str]:
    """
    Token-aware chunking that preserves sentence boundaries when possible.
    """
    enc = get_tokenizer(model_name)
    toks = enc.encode(text)
    if not toks:
        return []
    chunks = []
    start = 0
    length = len(toks)
    while start < length:
        end = min(start + chunk_size, length)
        # Try to extend end to nearest newline or sentence boundary in a small window
        window_end = min(end + 50, length)
        if window_end > end:
            segment = enc.decode(toks[end:window_end])
            # prefer newline or period boundary
            newline_pos = segment.find("\n")
            period_pos = segment.find(". ")
            if newline_pos != -1:
                end = end + newline_pos
            elif period_pos != -1:
                end = end + period_pos + 1
        chunk_text = enc.decode(toks[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks
# Token-based chunking with light sentence-boundary heuristics

# -----------------------------
# Embedding cache: SQLite
# -----------------------------
def init_embed_cache(db_path: Path = EMBED_CACHE_DB):
    """
    Initialize SQLite DB for embedding cache.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,
        model TEXT,
        vector BLOB,
        dim INTEGER,
        created_at REAL
    )
    """)
    conn.commit()
    return conn
# Creates table for embeddings

def fetch_cached_embedding(conn: sqlite3.Connection, key: str, model_name: str) -> Optional[np.ndarray]:
    """
    Return cached embedding if present.
    """
    cur = conn.cursor()
    cur.execute("SELECT vector, dim FROM embeddings WHERE id = ? AND model = ?", (key, model_name))
    row = cur.fetchone()
    if not row:
        return None
    blob, dim = row
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        return None
    return arr
# Fetch embedding from SQLite and reconstruct numpy array

def cache_embedding(conn: sqlite3.Connection, key: str, model_name: str, vec: np.ndarray):
    """
    Store embedding in cache.
    """
    cur = conn.cursor()
    blob = vec.astype(np.float32).tobytes()
    cur.execute("REPLACE INTO embeddings (id, model, vector, dim, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, model_name, sqlite3.Binary(blob), int(vec.size), time.time()))
    conn.commit()
# Write embedding into cache

# -----------------------------
# Embedding Manager (parallel friendly)
# -----------------------------
class EmbeddingService:
    """
    Wraps SentenceTransformer model with optional caching and multiprocessing-friendly API.
    """
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE, cache_db: Path = EMBED_CACHE_DB):
        self.model_name = model_name
        self.device = device
        self.cache_db = cache_db
        print(f"Loading embedding model {model_name} on {device} ...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.to(device)
        self.conn = init_embed_cache(cache_db)
        print("Embedding model loaded and cache initialized.")
    # Init loads model and opens SQLite

    def _text_key(self, text: str) -> str:
        """
        Deterministic key for caching (sha256 based).
        """
        import hashlib
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))
        h.update(text.encode("utf-8"))
        return h.hexdigest()
    # Generate sha256 key

    def embed_batch(self, texts: List[str], batch_size: int = EMBED_BATCH) -> np.ndarray:
        """
        Embed a batch with local caching per text.
        """
        results = []
        to_compute = []
        to_compute_indices = []
        # First check cache
        for i, t in enumerate(texts):
            key = self._text_key(t)
            cached = fetch_cached_embedding(self.conn, key, self.model_name)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                to_compute.append(t)
                to_compute_indices.append(i)
        # Compute missing embeddings
        if to_compute:
            computed = self.model.encode(to_compute, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
            # store them back
            for i, vec in enumerate(computed):
                idx = to_compute_indices[i]
                results[idx] = vec
                cache_embedding(self.conn, self._text_key(to_compute[i]), self.model_name, vec)
        return np.vstack(results).astype(np.float32)
    # Embeds with caching and returns numpy array

# -----------------------------
# Document Extraction Utilities (robust)
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    """
    Robust PDF extraction using PyMuPDF with fallback strategies.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Failed to open PDF {path}: {e}")
        return ""
    text_parts = []
    for p in range(len(doc)):
        try:
            page = doc.load_page(p)
            txt = page.get_text("text")
            if not txt or len(txt.strip()) < 20:
                # Try blocks or dict for scanned layout
                txt = page.get_text("blocks")
                if isinstance(txt, list):
                    # blocks -> join text parts
                    txt = "\n".join([b[4] for b in txt if len(b) >= 5 and isinstance(b[4], str)])
            text_parts.append(txt)
        except Exception as e:
            print(f"Page {p} extraction failed: {e}")
            continue
    doc.close()
    return "\n\n".join([p for p in text_parts if p and len(p.strip()) > 0])
# Extract text from PDF pages with fallback

def read_text_file(path: str) -> str:
    """
    Read text-like files safely.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return ""
# Safe text file reading

def extract_text(path: str) -> str:
    """
    Dispatch extraction based on extension.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in [".txt", ".md", ".py", ".json", ".csv", ".log"]:
        return read_text_file(path)
    # Unknown types: attempt read as text
    return read_text_file(path)
# Dispatch function

# -----------------------------
# Index Creation & Update
# -----------------------------
def create_faiss_index(vectors: np.ndarray, index_path: Path = FAISS_INDEX_PATH, index_type: str = FAISS_INDEX_TYPE) -> faiss.Index:
    """
    Create or overwrite FAISS index with chosen strategy.
    """
    d = vectors.shape[1]
    if index_type.upper() == "FLAT":
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
    elif index_type.upper() == "IVF":
        nlist = min(max(256, int(np.sqrt(len(vectors)))), 65536)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
    else:
        # HNSW for CPU - fast and memory friendly
        index = faiss.IndexHNSWFlat(d, N_M)
        # Set construction params if available
        if hasattr(index, "efConstruction"):
            index.hnsw.efConstruction = EF_CONSTRUCTION
        index.add(vectors)
    # Persist index to disk
    faiss.write_index(index, str(index_path))
    # Set search-time ef if HNSW
    if isinstance(index, faiss.IndexHNSWFlat):
        index.hnsw.efSearch = EF_SEARCH
    return index
# Build FAISS index and persist

def load_faiss_index(index_path: Path = FAISS_INDEX_PATH) -> faiss.Index:
    """
    Load FAISS index from disk.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(str(index_path))
# Load persisted FAISS index

# -----------------------------
# Incremental Indexing Pipeline
# -----------------------------
def index_folder(folder: str, rebuild: bool = False, recursive: bool = True, extensions: List[str] = None):
    """
    Index a folder incrementally with embedding cache and optional rebuild.
    """
    if extensions is None:
        extensions = ["*.pdf", "*.txt", "*.md", "*.py", "*.json", "*.csv", "*.log"]
    # Discover files
    patterns = []
    for ext in extensions:
        patterns.append(os.path.join(folder, "**", ext) if recursive else os.path.join(folder, ext))
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=recursive))
    files = sorted(set(files))
    if not files:
        print("No files found to index.")
        return
    # Load or create metadata store
    if rebuild and METADATA_JSON.exists():
        METADATA_JSON.unlink()
    metadata_list = []
    # We'll collect chunks and their metadata, then embed in batches
    all_chunks: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []
    enc = get_tokenizer()
    for path in files:
        try:
            text = extract_text(path)
        except Exception as e:
            print(f"Extraction failed for {path}: {e}")
            continue
        if not text or len(text.strip()) < 20:
            continue
        # Split into paragraphs first for semantic boundaries
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        # Build tokenwise chunks from paragraphs, preferring to keep paragraphs whole when possible
        for para in paragraphs:
            para_tokens = enc.encode(para)
            if len(para_tokens) <= CHUNK_TOKEN_SIZE:
                chunks = [enc.decode(para_tokens)]
            else:
                # Break long paragraph into token chunks
                chunks = chunk_by_tokens(para, CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS)
            for c in chunks:
                all_chunks.append(c)
                chunk_meta.append({"path": path, "text_preview": c[:400], "tokens": count_tokens(c)})
        print(f"Processed {path}: paragraphs {len(paragraphs)} -> chunks {len(all_chunks)}")
    if not all_chunks:
        print("No chunks created.")
        return
    # Initialize embedding service
    emb_service = EmbeddingService()
    # Compute embeddings in batches while using caching
    vectors = []
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i:i+EMBED_BATCH]
        emb = emb_service.embed_batch(batch, batch_size=EMBED_BATCH)
        vectors.append(emb)
    vectors = np.vstack(vectors).astype(np.float32)
    # Persist metadata as newline-delimited JSON
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    # Create FAISS index
    print(f"Creating FAISS index ({FAISS_INDEX_TYPE}) with {vectors.shape[0]} vectors of dim {vectors.shape[1]} ...")
    idx = create_faiss_index(vectors, FAISS_INDEX_PATH, index_type=FAISS_INDEX_TYPE)
    print("Indexing complete.")
# Index folder and create index

# -----------------------------
# Retrieval & Reranking (light)
# -----------------------------
def load_metadata_lines(metadata_path: Path = METADATA_JSON) -> List[Dict[str, Any]]:
    """
    Load line-delimited metadata into list.
    """
    if not metadata_path.exists():
        return []
    metas = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                metas.append(json.loads(line))
            except Exception:
                continue
    return metas
# Loads metadata list

def hybrid_search(query: str, k: int = 5, use_faiss: bool = True, rerank: bool = False) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Perform semantic search with FAISS and return top-k with optional simple lexical boosting.
    """
    # Load index & metadata
    index = load_faiss_index()
    metas = load_metadata_lines()
    # Embed query
    emb_service = EmbeddingService()
    qvec = emb_service.embed_batch([query])[0].reshape(1, -1)
    # Search
    D, I = index.search(qvec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        meta = metas[idx].copy()
        meta["score"] = float(dist)
        results.append((float(dist), meta))
    # Optionally do simple lexical boost: increase score if query tokens appear in preview
    if rerank:
        qlow = query.lower()
        boosted = []
        for score, meta in results:
            boost = 0.0
            if qlow in meta.get("text_preview", "").lower():
                boost += 0.2
            boosted.append((score - boost, meta))
        boosted.sort(key=lambda x: x[0])
        results = boosted
    return results
# Hybrid search using FAISS; can enable rerank

# -----------------------------
# CLI Entrypoint
# -----------------------------
def main():
    """
    CLI to run indexing and queries.
    """
    parser = argparse.ArgumentParser(description="Improved Local RAG Indexer (Windows-optimized)")
    parser.add_argument("--index", action="store_true", help="Build or update index")
    parser.add_argument("--folder", type=str, default=".", help="Folder to index")
    parser.add_argument("--query", type=str, help="Run a query against index")
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild metadata/index from scratch")
    args = parser.parse_args()
    if args.index:
        print("Starting indexing ...")
        index_folder(args.folder, rebuild=args.rebuild)
    if args.query:
        print("Running query ...")
        start = time.time()
        results = hybrid_search(args.query, k=args.k, rerank=True)
        elapsed = time.time() - start
        print(f"Query took {elapsed:.3f}s and returned {len(results)} hits")
        for i, (score, meta) in enumerate(results, 1):
            print(f"\n--- Result {i} (score {score:.4f}) ---")
            print(f"Path: {meta.get('path')}")
            print(f"Preview: {meta.get('text_preview')[:800]}...")
# CLI finished

if __name__ == "__main__":
    main()
# Run main when executed as script