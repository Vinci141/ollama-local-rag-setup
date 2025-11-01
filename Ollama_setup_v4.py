#!/usr/bin/env python3
"""
RAG system with comprehensive performance monitoring.
Tracks processing time, resource usage, and query performance.

Understanding What Each Metric Tells YouBefore we add the monitoring code, let's understand why each metric matters.
Processing time tells you how long it takes to index your documents - this helps you estimate how long larger batches will take.
Chunks created shows how your documents are being split up, which affects search granularity.
Index size reveals the storage overhead of your system.
CPU usage indicates whether you're bottlenecked by processing power or something else like disk speed.
Query response time is the most user-facing metric - it determines whether your system feels snappy or sluggish.
v4:In Ollama's Python client, options like temperature must be passed inside an options dictionary.
Add RAG system with performance monitoring"""
import glob
import json
import os
import re
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Dict, Optional

import faiss
import numpy as np
import psutil  # For CPU and memory monitoring
import tiktoken
from ollama import embed, chat, pull
from pypdf import PdfReader

# Configuration
CHUNK_SIZE = 500  # 1000
CHUNK_OVERLAP = 200
LOG_LINES_PER_CHUNK = 50
EMBED_MODEL = "mxbai-embed-large"
TEXT_MODEL = "gemma3"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"
PERFORMANCE_LOG = "performance_metrics.json"
BATCH_SIZE = 16
MAX_FILE_SIZE_MB = 500


def count_tokens(text: str, model_name='gemma3') -> int:
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # fallback tokenizer
        tokenizer = tiktoken.get_encoding('cl100k_base')
    return len(tokenizer.encode(text))


def build_prompt_within_token_limit(context_chunks, question, max_tokens=3000):  # Max should be upto 16K
    separator = "\n\n---\n\n"
    prompt_intro = "You are given the following extracted document chunks as context:\n\n"
    prompt_question = f"\n\nQuestion: {question}"

    allowed_tokens = max_tokens - count_tokens(prompt_intro) - count_tokens(prompt_question) - 100  # buffer

    selected_chunks = []
    token_count = 0
    for chunk in context_chunks:
        chunk_tokens = count_tokens(chunk)
        if token_count + chunk_tokens > allowed_tokens:
            break
        selected_chunks.append(chunk)
        token_count += chunk_tokens

    context = separator.join(selected_chunks)
    prompt = f"{prompt_intro}{context}{prompt_question}"
    return prompt


def query_index(
        question: str,
        k: int = 8,
        embed_model: str = EMBED_MODEL,
        text_model: str = TEXT_MODEL,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        filter_log_level: Optional[str] = None
) -> str:
    """Query with performance tracking and token-aware prompt building."""

    query_start = time.time()
    success = False

    try:
        print("Loading index...")
        index, metadata = load_index(index_path, metadata_path)

        print("Embedding query...")
        q_emb_resp = embed(model=embed_model, input=question)
        if "embeddings" in q_emb_resp:
            q_vec = np.array(q_emb_resp["embeddings"][0], dtype="float32")
        elif "embedding" in q_emb_resp:
            q_vec = np.array(q_emb_resp["embedding"], dtype="float32")
        else:
            raise RuntimeError(f"Unexpected embedding response")

        q_vec = q_vec.reshape(1, -1)

        print(f"Searching for top {k} chunks...")
        D, I = index.search(q_vec, k)

        results = []
        for idx in I[0]:
            if 0 <= idx < len(metadata):
                result = metadata[idx]

                if filter_log_level and result.get("file_type") == "log":
                    if filter_log_level.upper() not in result.get("log_levels", []):
                        continue

                results.append(result)

        if not results:
            return "No relevant documents found."

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

        # Build prompt respecting token limit
        prompt = build_prompt_within_token_limit(context_parts, question)

        print(f"Generating answer with {text_model}...")
        # commenting as it is giving error -TypeError: Client.chat() got an unexpected keyword argument 'temperature'
        # resp = chat(
        #     model=text_model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.2  # Controlled temperature for factual answers
        # )

        resp = chat(
            model=text_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,  # Controlled temperature for factual answers
                "top_p": 0.9,  # Optional: nucleus sampling
                "top_k": 40  # Optional: top-k sampling
            }
        )
        success = True
        return resp["message"]["content"]

    finally:
        query_duration = time.time() - query_start
        perf_monitor.record_query(question, query_duration, k, success)
        print(f"\n⏱️  Query completed in {query_duration:.3f} seconds")


class PerformanceMonitor:
    """
    Tracks and reports performance metrics for the RAG system.
    Think of this as your system's health monitor - it keeps track of everything
    that matters for understanding how well your system performs.
    """

    def __init__(self):
        self.metrics = {
            "indexing": {
                "start_time": None,
                "end_time": None,
                "duration_seconds": 0,
                "files_processed": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "bytes_processed": 0,
                "index_size_bytes": 0,
                "avg_cpu_percent": 0,
                "peak_memory_mb": 0,
                "chunks_per_second": 0
            },
            "queries": []
        }

        # CPU monitoring - we'll sample this periodically
        self.cpu_samples = []
        self.memory_samples = []
        self.process = psutil.Process()

    def start_indexing(self):
        """Call this when indexing begins."""
        self.metrics["indexing"]["start_time"] = datetime.now().isoformat()
        print(f"\n{'=' * 60}")
        print(f"📊 Performance Monitoring Started")
        print(f"{'=' * 60}")

    def sample_resources(self):
        """
        Sample CPU and memory usage. This is like taking your pulse -
        you do it periodically to see how hard the system is working.
        """
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB

            self.cpu_samples.append(cpu)
            self.memory_samples.append(memory)
        except:
            pass  # Don't let monitoring crash the main process

    def record_file_processed(self, filepath: str, chunks: int):
        """Record that we've processed a file and how many chunks it created."""
        self.metrics["indexing"]["files_processed"] += 1
        self.metrics["indexing"]["chunks_created"] += chunks

        try:
            file_size = os.path.getsize(filepath)
            self.metrics["indexing"]["bytes_processed"] += file_size
        except:
            pass

    def end_indexing(self):
        """
        Call this when indexing completes. This calculates all the final
        statistics and rates like chunks per second.
        """
        self.metrics["indexing"]["end_time"] = datetime.now().isoformat()

        start = datetime.fromisoformat(self.metrics["indexing"]["start_time"])
        end = datetime.fromisoformat(self.metrics["indexing"]["end_time"])
        duration = (end - start).total_seconds()

        self.metrics["indexing"]["duration_seconds"] = round(duration, 2)

        # Calculate chunks per second (throughput metric)
        if duration > 0:
            self.metrics["indexing"]["chunks_per_second"] = round(
                self.metrics["indexing"]["chunks_created"] / duration, 2
            )

        # Calculate average CPU usage across all samples
        if self.cpu_samples:
            self.metrics["indexing"]["avg_cpu_percent"] = round(
                sum(self.cpu_samples) / len(self.cpu_samples), 1
            )

        # Record peak memory usage
        if self.memory_samples:
            self.metrics["indexing"]["peak_memory_mb"] = round(
                max(self.memory_samples), 1
            )

        # Get index file size
        try:
            self.metrics["indexing"]["index_size_bytes"] = os.path.getsize(INDEX_PATH)
        except:
            pass

    def record_query(self, question: str, duration: float, chunks_retrieved: int, success: bool):
        """
        Record metrics for a single query. Over time, this builds up statistics
        about typical query performance.
        """
        self.metrics["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "question": question[:100],  # Truncate long questions
            "duration_seconds": round(duration, 3),
            "chunks_retrieved": chunks_retrieved,
            "success": success
        })

    def print_indexing_report(self):
        """
        Display a nice summary of indexing performance. This is like a report card
        for your indexing job - it tells you everything important at a glance.
        """
        m = self.metrics["indexing"]

        print(f"\n{'=' * 60}")
        print(f"📊 INDEXING PERFORMANCE REPORT")
        print(f"{'=' * 60}\n")

        # Time metrics
        print(f"⏱️  Processing Time")
        print(f"   Total Duration: {m['duration_seconds']} seconds ({m['duration_seconds'] / 60:.1f} minutes)")
        print(f"   Start: {m['start_time']}")
        print(f"   End: {m['end_time']}\n")

        # Volume metrics
        print(f"📦 Data Processed")
        print(f"   Files Processed: {m['files_processed']}")
        print(f"   Chunks Created: {m['chunks_created']:,}")
        print(f"   Embeddings Generated: {m['embeddings_generated']:,}")
        print(f"   Data Volume: {m['bytes_processed'] / (1024 * 1024):.2f} MB\n")

        # Performance metrics
        print(f"⚡ Performance")
        print(f"   Throughput: {m['chunks_per_second']:.2f} chunks/second")
        print(f"   Average CPU Usage: {m['avg_cpu_percent']}%")
        print(f"   Peak Memory Usage: {m['peak_memory_mb']:.1f} MB\n")

        # Storage metrics
        print(f"💾 Storage")
        print(f"   Index Size: {m['index_size_bytes'] / (1024 * 1024):.2f} MB")
        if m['bytes_processed'] > 0:
            compression_ratio = m['index_size_bytes'] / m['bytes_processed']
            print(f"   Compression Ratio: {compression_ratio:.2%} of original")

        print(f"\n{'=' * 60}\n")

    def print_query_stats(self):
        """Show statistics about query performance across all queries."""
        if not self.metrics["queries"]:
            print("No queries recorded yet.")
            return

        queries = self.metrics["queries"]
        durations = [q["duration_seconds"] for q in queries]

        print(f"\n{'=' * 60}")
        print(f"🔍 QUERY PERFORMANCE STATISTICS")
        print(f"{'=' * 60}\n")

        print(f"Total Queries: {len(queries)}")
        print(f"Average Response Time: {sum(durations) / len(durations):.3f} seconds")
        print(f"Fastest Query: {min(durations):.3f} seconds")
        print(f"Slowest Query: {max(durations):.3f} seconds")
        print(f"Success Rate: {sum(1 for q in queries if q['success']) / len(queries) * 100:.1f}%")

        print(f"\n{'=' * 60}\n")

    def save_to_file(self, filepath: str = PERFORMANCE_LOG):
        """
        Save all metrics to a JSON file for later analysis. This is useful
        for tracking performance over time or comparing different configurations.
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Performance metrics saved to: {filepath}")


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


def parse_log_line(line: str) -> Optional[Dict]:
    """Parse common log formats and extract metadata."""
    line = line.strip()
    if not line:
        return None

    if line.startswith('{'):
        try:
            return json.loads(line)
        except:
            pass

    patterns = [
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(\w+)\s+\[(.*?)\]\s+(.*)',
        r'([\d\.]+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)',
        r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(.*)',
        r'(\w+):\s+(.*)',
    ]

    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            return {"raw": line, "parsed": True, "groups": match.groups()}

    return {"raw": line, "parsed": False}


def chunk_log_file(filepath: str, lines_per_chunk: int = LOG_LINES_PER_CHUNK) -> List[Dict]:
    """Stream and chunk log file line by line."""
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
                # Sample resources every 100 lines to avoid overhead
                if line_number % 100 == 0:
                    perf_monitor.sample_resources()

                line_number += 1
                parsed = parse_log_line(line)

                if parsed:
                    current_chunk_lines.append(parsed.get("raw", line))

                    if parsed.get("parsed") and "groups" in parsed:
                        groups = parsed["groups"]
                        for group in groups:
                            if any(level in str(group).upper() for level in
                                   ["ERROR", "WARN", "INFO", "DEBUG", "FATAL"]):
                                current_chunk_metadata["log_levels"].add(str(group).upper())

                if len(current_chunk_lines) >= lines_per_chunk:
                    chunk_text = "\n".join(current_chunk_lines)
                    chunks.append({
                        "text": chunk_text,
                        "start_line": current_chunk_metadata["start_line"],
                        "end_line": line_number,
                        "log_levels": list(current_chunk_metadata["log_levels"]),
                        "line_count": len(current_chunk_lines)
                    })

                    current_chunk_lines = []
                    current_chunk_metadata = {
                        "start_line": line_number + 1,
                        "timestamps": [],
                        "log_levels": set()
                    }

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

    if ext in [".log"]:
        return None

    if ext in [".txt", ".md", ".py", ".json", ".csv"]:
        try:
            if file_size_mb > MAX_FILE_SIZE_MB or streaming:
                chunks = []
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    while True:
                        chunk = f.read(CHUNK_SIZE * 10)
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
    """Index all documents with performance monitoring."""

    # Start performance monitoring
    perf_monitor.start_indexing()

    if text_extensions is None:
        text_extensions = ["*.txt", "*.md", "*.py", "*.json", "*.csv", "*.pdf", "*.log"]

    files = []
    for pattern in text_extensions:
        files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))

    print(f"Found {len(files)} files to process.")
    if not files:
        raise RuntimeError(f"No files found in {folder_path}")

    metadata = []
    index = None
    d = None
    total_chunks = 0

    for file_idx, path in enumerate(files, 1):
        ext = os.path.splitext(path)[1].lower()
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        print(f"\nProcessing [{file_idx}/{len(files)}]: {os.path.basename(path)} ({file_size_mb:.2f} MB)")

        # Sample resources at the start of each file
        perf_monitor.sample_resources()

        if ext == ".log":
            print(f"  Using streaming log parser...")
            log_chunks = chunk_log_file(path, LOG_LINES_PER_CHUNK)

            if not log_chunks:
                print("  No valid log entries found")
                continue

            print(f"  Created {len(log_chunks)} log chunks")
            perf_monitor.record_file_processed(path, len(log_chunks))

            for i in range(0, len(log_chunks), BATCH_SIZE):
                batch = log_chunks[i:i + BATCH_SIZE]
                batch_texts = [chunk["text"] for chunk in batch]

                with ThreadPool(min(len(batch_texts), 8)) as pool:
                    emb_results = pool.starmap(embedding_worker,
                                               [(embed_model, text) for text in batch_texts])

                valid_vecs = [vec for vec in emb_results if vec is not None]
                if not valid_vecs:
                    continue

                if index is None:
                    d = valid_vecs[0].shape[0]
                    index = faiss.IndexFlatL2(d)

                xb = np.stack(valid_vecs)
                index.add(xb)
                perf_monitor.metrics["indexing"]["embeddings_generated"] += len(valid_vecs)

                for j, (vec, log_chunk) in enumerate(zip(emb_results, batch)):
                    if vec is not None:
                        metadata.append({
                            "path": path,
                            "chunk_index": i + j,
                            "text": log_chunk["text"][:1000],
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
            text = extract_text_from_file(path)
            if not text:
                print("  Skipping (no text extracted)")
                continue

            chunks = chunk_text(text)
            print(f"  Created {len(chunks)} chunks")
            perf_monitor.record_file_processed(path, len(chunks))

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
                perf_monitor.metrics["indexing"]["embeddings_generated"] += len(valid_vecs)

                for j, vec in enumerate(emb_results):
                    if vec is not None:
                        metadata.append({
                            "path": path,
                            "chunk_index": i + j,
                            "text": batch_chunks[j][:1000],
                            "file_type": ext[1:]
                        })
                        total_chunks += 1

    if index is None or index.ntotal == 0:
        raise RuntimeError("No embeddings created. Check Ollama is running.")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # End performance monitoring and print report
    perf_monitor.end_indexing()
    perf_monitor.print_indexing_report()
    perf_monitor.save_to_file()


def load_index(
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH
) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """Load FAISS index and metadata from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found at {index_path}. Run with --index first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run with --index first.")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG system with performance monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--index", action="store_true", help="Create/rebuild index")
    parser.add_argument("--folder", type=str, default=".", help="Folder to index")
    parser.add_argument("--query", type=str, help="Query the indexed documents")
    parser.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve")
    parser.add_argument("--log-level", type=str,
                        choices=["ERROR", "WARN", "INFO", "DEBUG", "FATAL"],
                        help="Filter by log level")
    parser.add_argument("--show-stats", action="store_true",
                        help="Show performance statistics from previous queries")

    args = parser.parse_args()

    if args.show_stats:
        # Load and display stats from saved metrics
        try:
            with open(PERFORMANCE_LOG, 'r') as f:
                perf_monitor.metrics = json.load(f)
            perf_monitor.print_indexing_report()
            perf_monitor.print_query_stats()
        except FileNotFoundError:
            print("No performance metrics found. Run indexing or queries first.")
        exit(0)

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

        print(f"\nIndexing folder: {args.folder}")
        index_folder(args.folder)

    if args.query:
        answer = query_index(args.query, k=args.k, filter_log_level=args.log_level)
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60 + "\n")
        print(answer)
        print()

        # Show query statistics
        perf_monitor.print_query_stats()
