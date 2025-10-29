#!/usr/bin/env python3
"""
Optimized Ollama-based RAG system for large log files and documents.
Includes streaming, batch processing, and smart log parsing.
"""
from ollama import embed, chat, pull
import os
import glob
import json
import numpy as np
import faiss
from pypdf import PdfReader
from typing import List, Tuple, Dict, Optional
from multiprocessing.pool import ThreadPool
import re
from datetime import datetime

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LOG_LINES_PER_CHUNK = 50  # For log files: group N lines per chunk
EMBED_MODEL = "mxbai-embed-large"
TEXT_MODEL = "gemma3"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"
BATCH_SIZE = 16  # Embeddings per batch
MAX_FILE_SIZE_MB = 500  # Stream files larger than this


def parse_log_line(line: str) -> Optional[Dict]:
    """
    Parse common log formats and extract metadata.
    Supports: Apache, JSON, Python logging, syslog, etc.
    """
    line = line.strip()
    if not line:
        return None

    # Try JSON format first
    if line.startswith('{'):
        try:
            return json.loads(line)
        except:
            pass

    # Common log patterns
    patterns = [
        # ISO timestamp with level: 2024-01-15 10:23:45 ERROR [module] message
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(\w+)\s+\[(.*?)\]\s+(.*)',
        # Apache/Nginx: 127.0.0.1 - - [15/Jan/2024:10:23:45] "GET /api" 200
        r'([\d\.]+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)',
        # Syslog: Jan 15 10:23:45 hostname process[pid]: message
        r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(.*)',
        # Simple: LEVEL: message
        r'(\w+):\s+(.*)',
    ]

    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            return {"raw": line, "parsed": True, "groups": match.groups()}

    return {"raw": line, "parsed": False}


def chunk_log_file(filepath: str, lines_per_chunk: int = LOG_LINES_PER_CHUNK) -> List[Dict]:
    """
    Stream and chunk log file line by line.
    Returns list of chunk dictionaries with metadata.
    """
    chunks = []
    current_chunk_lines = []
    current_chunk_metadata = {
        "start_line": 0,
        "timestamps": [],
        "log_levels": set()
    }

    line_number = 0

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_number += 1
                parsed = parse_log_line(line)

                if parsed:
                    current_chunk_lines.append(parsed.get("raw", line))

                    # Extract metadata
                    if parsed.get("parsed") and "groups" in parsed:
                        groups = parsed["groups"]
                        # Try to extract timestamp and level
                        for group in groups:
                            if any(level in str(group).upper() for level in
                                   ["ERROR", "WARN", "INFO", "DEBUG", "FATAL"]):
                                current_chunk_metadata["log_levels"].add(str(group).upper())

                # Create chunk when reaching threshold
                if len(current_chunk_lines) >= lines_per_chunk:
                    chunk_text = "\n".join(current_chunk_lines)
                    chunks.append({
                        "text": chunk_text,
                        "start_line": current_chunk_metadata["start_line"],
                        "end_line": line_number,
                        "log_levels": list(current_chunk_metadata["log_levels"]),
                        "line_count": len(current_chunk_lines)
                    })

                    # Reset for next chunk
                    current_chunk_lines = []
                    current_chunk_metadata = {
                        "start_line": line_number + 1,
                        "timestamps": [],
                        "log_levels": set()
                    }

        # Add remaining lines as final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append({
                "text": chunk_text,
                "start_line": current_chunk_metadata["start_line"],
                "end_line": line_number,
                "log_levels": list(current_chunk_metadata["log_levels"]),
                "line_count": len(current_chunk_lines)
            })

    except Exception as e:
        print(f"Error reading log file {filepath}: {e}")
        return []

    return chunks


def extract_text_from_file(path: str, streaming: bool = False) -> str:
    """Extract text with optional streaming for large files."""
    ext = os.path.splitext(path)[1].lower()
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    # For log files, always use streaming/chunking
    if ext in [".log"]:
        return None  # Signal to use chunk_log_file instead

    if ext in [".txt", ".md", ".py", ".json", ".csv"]:
        try:
            # Stream large files
            if file_size_mb > MAX_FILE_SIZE_MB or streaming:
                chunks = []
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    while True:
                        chunk = f.read(CHUNK_SIZE * 10)  # Read 10 chunks at a time
                        if not chunk:
                            break
                        chunks.append(chunk)
                return "".join(chunks)
            else:
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
    """Standard text chunking for non-log files."""
    if not text or len(text.strip()) == 0:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = end - overlap
    return chunks


def embedding_worker(embed_model: str, chunk: str):
    """Worker function for parallel embedding generation."""
    try:
        emb_resp = embed(model=embed_model, input=chunk)
        if "embeddings" in emb_resp:
            vec = np.array(emb_resp["embeddings"][0], dtype="float32")
        elif "embedding" in emb_resp:
            vec = np.array(emb_resp["embedding"], dtype="float32")
        else:
            return None
        return vec
    except Exception as e:
        print(f"  Embedding failed: {str(e)[:100]}")
        return None


def index_folder(
        folder_path: str,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        embed_model: str = EMBED_MODEL,
        text_extensions: List[str] = None
) -> None:
    """Index all documents with optimized log file handling."""
    if text_extensions is None:
        text_extensions = ["*.txt", "*.md", "*.py", "*.json", "*.csv", "*.pdf", "*.log"]

    files = []
    for pattern in text_extensions:
        files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))

    print(f"Found {len(files)} files to process.")
    if not files:
        raise RuntimeError(f"No files found in {folder_path} matching patterns: {text_extensions}")

    metadata = []
    index = None
    d = None
    total_chunks = 0

    for file_idx, path in enumerate(files, 1):
        ext = os.path.splitext(path)[1].lower()
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        print(f"\nProcessing [{file_idx}/{len(files)}]: {os.path.basename(path)} ({file_size_mb:.2f} MB)")

        # Handle log files specially
        if ext == ".log":
            print(f"  Using streaming log parser...")
            log_chunks = chunk_log_file(path, LOG_LINES_PER_CHUNK)

            if not log_chunks:
                print("  No valid log entries found")
                continue

            print(f"  Created {len(log_chunks)} log chunks")

            # Process log chunks in batches
            for i in range(0, len(log_chunks), BATCH_SIZE):
                batch = log_chunks[i:i + BATCH_SIZE]
                batch_texts = [chunk["text"] for chunk in batch]

                # Generate embeddings in parallel
                with ThreadPool(min(len(batch_texts), 8)) as pool:
                    emb_results = pool.starmap(embedding_worker,
                                               [(embed_model, text) for text in batch_texts])

                valid_vecs = [vec for vec in emb_results if vec is not None]
                if not valid_vecs:
                    continue

                # Initialize index on first valid embedding
                if index is None:
                    d = valid_vecs[0].shape[0]
                    index = faiss.IndexFlatL2(d)

                # Add to index
                xb = np.stack(valid_vecs)
                index.add(xb)

                # Store metadata
                for j, (vec, log_chunk) in enumerate(zip(emb_results, batch)):
                    if vec is not None:
                        metadata.append({
                            "path": path,
                            "chunk_index": i + j,
                            "text": log_chunk["text"][:1000],  # Preview
                            "file_type": "log",
                            "start_line": log_chunk["start_line"],
                            "end_line": log_chunk["end_line"],
                            "log_levels": log_chunk["log_levels"],
                            "line_count": log_chunk["line_count"]
                        })
                        total_chunks += 1

                if (i + BATCH_SIZE) % (BATCH_SIZE * 5) == 0:
                    print(f"    Processed {min(i + BATCH_SIZE, len(log_chunks))}/{len(log_chunks)} chunks...")

        else:
            # Handle regular files
            text = extract_text_from_file(path)
            if not text:
                print("  Skipping (no text extracted)")
                continue

            chunks = chunk_text(text)
            print(f"  Created {len(chunks)} chunks")

            # Process in batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]

                with ThreadPool(min(len(batch_chunks), 8)) as pool:
                    emb_results = pool.starmap(embedding_worker,
                                               [(embed_model, chunk) for chunk in batch_chunks])

                valid_vecs = [vec for vec in emb_results if vec is not None]
                if not valid_vecs:
                    continue

                if index is None:
                    d = valid_vecs[0].shape[0]
                    index = faiss.IndexFlatL2(d)

                xb = np.stack(valid_vecs)
                index.add(xb)

                for j, vec in enumerate(emb_results):
                    if vec is not None:
                        metadata.append({
                            "path": path,
                            "chunk_index": i + j,
                            "text": batch_chunks[j][:1000],
                            "file_type": ext[1:]  # Remove leading dot
                        })
                        total_chunks += 1

    if index is None or index.ntotal == 0:
        raise RuntimeError(
            "No embeddings created. Check that:\n"
            f"  1. Ollama is running (try: ollama list)\n"
            f"  2. Model '{embed_model}' is available (try: ollama pull {embed_model})\n"
            f"  3. Files contain readable text"
        )

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Index created successfully!")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Index file: {index_path}")
    print(f"  Metadata file: {metadata_path}")
    print(f"{'=' * 60}")


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
        k: int = 8,
        embed_model: str = EMBED_MODEL,
        text_model: str = TEXT_MODEL,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        filter_log_level: Optional[str] = None
) -> str:
    """Query with optional log level filtering."""
    print("Loading index...")
    index, metadata = load_index(index_path, metadata_path)

    print("Embedding query...")
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

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            result = metadata[idx]

            # Apply log level filter if specified
            if filter_log_level and result.get("file_type") == "log":
                if filter_log_level.upper() not in result.get("log_levels", []):
                    continue

            results.append(result)

    if not results:
        return "No relevant documents found."

    # Build context with enhanced metadata for logs
    context_parts = []
    for r in results:
        if r.get("file_type") == "log":
            context_parts.append(
                f"File: {r['path']}\n"
                f"Lines: {r.get('start_line', '?')}-{r.get('end_line', '?')}\n"
                f"Log Levels: {', '.join(r.get('log_levels', []))}\n"
                f"Text:\n{r['text']}"
            )
        else:
            context_parts.append(
                f"File: {r['path']}\n"
                f"Chunk: {r['chunk_index']}\n"
                f"Text:\n{r['text']}"
            )

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"You are given the following extracted document chunks as context:\n\n"
        f"{context}\n\n"
        f"Answer the question concisely and cite which file/chunk you used when relevant.\n"
        f"For log files, mention line numbers and log levels if applicable.\n\n"
        f"Question: {question}"
    )

    print(f"Generating answer with {text_model}...")
    resp = chat(model=text_model, messages=[{"role": "user", "content": prompt}])

    return resp["message"]["content"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index folder (including large log files) and query with Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a folder with logs
  python script.py --index --folder /var/log

  # Query all documents
  python script.py --query "What errors occurred today?"

  # Query with log level filter
  python script.py --query "Show database issues" --log-level ERROR

  # Adjust chunk retrieval
  python script.py --query "Summarize activity" --k 10
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
        default=8,
        help="Number of chunks to retrieve (default: 8)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["ERROR", "WARN", "INFO", "DEBUG", "FATAL"],
        help="Filter log results by level"
    )

    args = parser.parse_args()

    if not args.index and not args.query:
        parser.print_help()
        exit(0)

    if args.index:
        print("Checking models...")
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
        answer = query_index(args.query, k=args.k, filter_log_level=args.log_level)
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60 + "\n")
        print(answer)
        print()
