'''
What You've Built So Far 🎯
In simple words: You've created a "smart document search" system that can:

Read your documents (PDFs, text files, code files, etc.)
Break them into small chunks (pieces of ~1000 characters)
Convert text into numbers (embeddings) - like giving each chunk a unique "fingerprint"
Store these fingerprints in a searchable index (FAISS database)
Answer questions by:

Converting your question into a fingerprint
Finding chunks with similar fingerprints (semantic search)
Sending those chunks to an AI model (Gemma3) to generate an answer
Think of it like: Instead of searching for exact keywords (like Google), you're searching for meaning. If you ask "how to cook pasta", it can find chunks mentioning "boiling noodles" even without the word "pasta".
What Makes This Powerful 💪
Works offline - No internet needed, all runs on your PC
Private - Your documents never leave your computer
Free - No API costs
Fast - FAISS makes searching millions of documents instant
Smart - Understands context, not just keywords
'''

#!/usr/bin/env python3
"""
Ollama-based RAG (Retrieval-Augmented Generation) system.
Index documents and query them using local LLMs via Ollama.
"""
from ollama import embed, chat, pull
import os
import glob
import json
import numpy as np
import faiss
from pypdf import PdfReader
from typing import List, Tuple, Dict

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "mxbai-embed-large"
TEXT_MODEL = "gemma3"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"


def extract_text_from_file(path: str) -> str:
    """Extract text from supported file types."""
    ext = os.path.splitext(path)[1].lower()

    if ext in [".txt", ".md", ".py", ".json", ".csv", ".log"]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            return ""

    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"Warning: Could not parse PDF {path}: {e}")
            return ""

    return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or len(text.strip()) == 0:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        if end >= length:
            break

        start = end - overlap

    return chunks


def index_folder(
        folder_path: str,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        embed_model: str = EMBED_MODEL,
        text_extensions: List[str] = None
) -> None:
    """Build FAISS index from documents in a folder."""
    if text_extensions is None:
        text_extensions = ["*.txt", "*.md", "*.py", "*.json", "*.csv", "*.pdf", "*.log"]

    # Find all matching files
    files = []
    for pattern in text_extensions:
        files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))

    print(f"Found {len(files)} files to process.")

    if not files:
        raise RuntimeError(f"No files found in {folder_path} matching patterns: {text_extensions}")

    all_embeddings = []
    metadata = []

    for file_idx, path in enumerate(files, 1):
        print(f"Processing [{file_idx}/{len(files)}]: {os.path.basename(path)}")

        text = extract_text_from_file(path)
        if not text:
            print(f"  Skipping (no text extracted)")
            continue

        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            try:
                # FIXED: embed() returns {"embeddings": [[...]]} not {"embedding": [...]}
                emb_resp = embed(model=embed_model, input=chunk)

                # Handle both response formats for compatibility
                if "embeddings" in emb_resp:
                    vec = np.array(emb_resp["embeddings"][0], dtype="float32")
                elif "embedding" in emb_resp:
                    vec = np.array(emb_resp["embedding"], dtype="float32")
                else:
                    print(f"  Warning: Unexpected response format: {emb_resp.keys()}")
                    continue

                all_embeddings.append(vec)
                metadata.append({
                    "path": path,
                    "chunk_index": i,
                    "text": chunk[:1000]  # Store preview
                })

            except Exception as e:
                print(f"  Embedding failed for chunk {i}: {e}")
                continue

    if len(all_embeddings) == 0:
        raise RuntimeError(
            "No embeddings created. Check that:\n"
            f"  1. Ollama is running (try: ollama list)\n"
            f"  2. Model '{embed_model}' is available (try: ollama pull {embed_model})\n"
            f"  3. Files contain readable text"
        )

    # Build FAISS index
    d = all_embeddings[0].shape[0]
    xb = np.stack(all_embeddings)

    index = faiss.IndexFlatL2(d)
    index.add(xb)

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Index created with {index.ntotal} vectors")
    print(f"  Saved to: {index_path}")
    print(f"  Metadata: {metadata_path}")


def load_index(
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH
) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """Load FAISS index and metadata from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run with --index first."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. Run with --index first."
        )

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def query_index(
        question: str,
        k: int = 5,
        embed_model: str = EMBED_MODEL,
        text_model: str = TEXT_MODEL,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH
) -> str:
    """Query indexed documents using RAG."""
    print(f"Loading index...")
    index, metadata = load_index(index_path, metadata_path)

    print(f"Embedding query...")
    # FIXED: Same fix for query embedding
    q_emb_resp = embed(model=embed_model, input=question)

    if "embeddings" in q_emb_resp:
        q_vec = np.array(q_emb_resp["embeddings"][0], dtype="float32")
    elif "embedding" in q_emb_resp:
        q_vec = np.array(q_emb_resp["embedding"], dtype="float32")
    else:
        raise RuntimeError(f"Unexpected embedding response: {q_emb_resp.keys()}")

    q_vec = q_vec.reshape(1, -1)

    print(f"Searching for top {k} chunks...")
    D, I = index.search(q_vec, k)

    # Gather results
    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])

    if not results:
        return "No relevant documents found."

    # Build context
    context = "\n\n---\n\n".join([
        f"File: {r['path']}\nChunk: {r['chunk_index']}\nText:\n{r['text']}"
        for r in results
    ])

    # Generate answer
    prompt = (
        f"You are given the following extracted document chunks as context:\n\n"
        f"{context}\n\n"
        f"Answer the question concisely and cite which file/chunk you used when relevant.\n\n"
        f"Question: {question}"
    )

    print(f"Generating answer with {text_model}...")
    resp = chat(model=text_model, messages=[{"role": "user", "content": prompt}])

    return resp["message"]["content"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index folder and query documents with Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a folder
  python script.py --index --folder /path/to/docs

  # Query indexed documents
  python script.py --query "What is the main topic?"

  # Index and query
  python script.py --index --folder ./docs --query "Summarize the key points"
        """
    )

    parser.add_argument(
        "--index",
        action="store_true",
        help="Create/rebuild index from folder"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=".",
        help="Folder to index (default: current directory)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a question against indexed documents"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )

    args = parser.parse_args()

    if not args.index and not args.query:
        parser.print_help()
        exit(0)

    if args.index:
        print("Checking models...")

        # Try to pull models
        for model_name in [EMBED_MODEL, TEXT_MODEL]:
            try:
                print(f"  Pulling {model_name}...")
                pull(model_name)
                print(f"  ✓ {model_name} ready")
            except Exception as e:
                print(f"  Warning: Could not pull {model_name}: {e}")
                print(f"  Ensure it's available locally: ollama pull {model_name}")

        print(f"\nIndexing folder: {args.folder}")
        index_folder(args.folder)

    if args.query:
        answer = query_index(args.query, k=args.k)
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60 + "\n")
        print(answer)
        print()
