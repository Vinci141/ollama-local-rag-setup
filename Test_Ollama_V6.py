#!/usr/bin/env python3
"""
Enhanced RAG system with Level 3 features:
- FAISS + BM25 Hybrid Search
- Re-ranking with Cross-Encoder
- Conversational Memory for multi-turn dialogue
- Query Routing stub for future expansion
- Batch Processing support
- Performance Monitoring with evaluation metrics

Advanced Retrieval (Level 3)
Hybrid Search: Introduced the HybridRetriever incorporating both Semantic (FAISS) search and Keyword Search (BM25Okapi), often merged using Reciprocal Rank Fusion (RRF) logic. Re-ranking: Integrated Cross-encoder re-ranking to improve relevance scoring of retrieved chunks. Conversational Memory: Added the ConversationalMemory class to store history and enable multi-turn dialogue. Indexing Upgrade: Switched to using FAISS IndexIVFFlat (when vectors are sufficient) for faster search performance on larger indexes. LLM Temperature: Set temperature_used to 0.3.

"""

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
import torch
from ollama import embed, chat, pull
from pypdf import PdfReader
# Additional libraries needed
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# RERANKER_MODEL_NAME = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")# New code in V7

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
LOG_LINES_PER_CHUNK = 50
EMBED_MODEL = "mxbai-embed-large"
TEXT_MODEL = "gemma3"
# RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # New code in V7
INDEX_PATH = "index/faiss_index.bin"
METADATA_PATH = "metadata.json"
PERFORMANCE_LOG = "performance_metrics.json"
BATCH_SIZE = 16
MAX_FILE_SIZE_MB = 500
temperature_used = 0.3

# Tokenizer cache for reuse
_tokenizer_cache = {}


def count_tokens(text: str, model_name='gemma3') -> int:
    try:
        if model_name in _tokenizer_cache:
            tokenizer = _tokenizer_cache[model_name]
        else:
            tokenizer = tiktoken.encoding_for_model(model_name)
            _tokenizer_cache[model_name] = tokenizer
    except KeyError:
        tokenizer = tiktoken.get_encoding('cl100k_base')
    return len(tokenizer.encode(text))


def build_prompt_within_token_limit(context_chunks, question, max_tokens=8000):
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


class PerformanceMonitor:
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
            "queries": [],
            "evaluation": {
                "retrieval": [],
                "generation": []
            }
        }
        self.cpu_samples = []
        self.memory_samples = []
        self.process = psutil.Process()

    def start_indexing(self):
        self.metrics["indexing"]["start_time"] = datetime.now().isoformat()
        print(f"\n{'=' * 60}")
        print(f"📊 Performance Monitoring Started")
        print(f"{'=' * 60}")

    def sample_resources(self):
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            self.cpu_samples.append(cpu)
            self.memory_samples.append(memory)
        except Exception:
            pass

    def record_file_processed(self, filepath: str, chunks: int):
        self.metrics["indexing"]["files_processed"] += 1
        self.metrics["indexing"]["chunks_created"] += chunks
        try:
            file_size = os.path.getsize(filepath)
            self.metrics["indexing"]["bytes_processed"] += file_size
        except Exception:
            pass

    def end_indexing(self):
        self.metrics["indexing"]["end_time"] = datetime.now().isoformat()
        start = datetime.fromisoformat(self.metrics["indexing"]["start_time"])
        end = datetime.fromisoformat(self.metrics["indexing"]["end_time"])
        duration = (end - start).total_seconds()
        self.metrics["indexing"]["duration_seconds"] = round(duration, 2)
        if duration > 0:
            self.metrics["indexing"]["chunks_per_second"] = round(
                self.metrics["indexing"]["chunks_created"] / duration, 2)
        if self.cpu_samples:
            self.metrics["indexing"]["avg_cpu_percent"] = round(
                sum(self.cpu_samples) / len(self.cpu_samples), 1)
        if self.memory_samples:
            self.metrics["indexing"]["peak_memory_mb"] = round(max(self.memory_samples), 1)
        try:
            self.metrics["indexing"]["index_size_bytes"] = os.path.getsize(INDEX_PATH)
        except Exception:
            pass

    def record_query(self, question: str, duration: float, chunks_retrieved: int, success: bool):
        self.metrics["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "question": question[:100],
            "duration_seconds": round(duration, 3),
            "chunks_retrieved": chunks_retrieved,
            "success": success
        })

    def record_evaluation(self, retrieval_scores: Dict[str, float], generation_scores: Dict[str, float]):
        self.metrics["evaluation"]["retrieval"].append(retrieval_scores)
        self.metrics["evaluation"]["generation"].append(generation_scores)

    def print_indexing_report(self):
        m = self.metrics["indexing"]
        print(f"\n{'=' * 60}")
        print(f"📊 INDEXING PERFORMANCE REPORT")
        print(f"{'=' * 60}\n")
        print(f"⏱️  Processing Time")
        print(f"   Total Duration: {m['duration_seconds']} seconds ({m['duration_seconds'] / 60:.1f} minutes)")
        print(f"   Start: {m['start_time']}")
        print(f"   End: {m['end_time']}\n")
        print(f"📦 Data Processed")
        print(f"   Files Processed: {m['files_processed']}")
        print(f"   Chunks Created: {m['chunks_created']:,}")
        print(f"   Embeddings Generated: {m['embeddings_generated']:,}")
        print(f"   Data Volume: {m['bytes_processed'] / (1024 * 1024):.2f} MB\n")
        print(f"⚡ Performance")
        print(f"   Throughput: {m['chunks_per_second']:.2f} chunks/second")
        print(f"   Average CPU Usage: {m['avg_cpu_percent']}%")
        print(f"   Peak Memory Usage: {m['peak_memory_mb']:.1f} MB\n")
        print(f"💾 Storage")
        print(f"   Index Size: {m['index_size_bytes'] / (1024 * 1024):.2f} MB")
        if m['bytes_processed'] > 0:
            compression_ratio = m['index_size_bytes'] / m['bytes_processed']
            print(f"   Compression Ratio: {compression_ratio:.2%} of original")
        print(f"\n{'=' * 60}\n")

    def print_query_stats(self):
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
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Performance metrics saved to: {filepath}")


# Global performance monitor
perf_monitor = PerformanceMonitor()


def parse_log_line(line: str) -> Optional[Dict]:
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
    chunks = []
    current_chunk_lines = []
    current_chunk_metadata = {
        "start_line": 0,
        "log_levels": set()
    }
    line_number = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
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


class ConversationalMemory:
    def __init__(self, max_turns=5):
        self.memory = []
        self.max_turns = max_turns

    def add_turn(self, user_input, system_response):
        self.memory.append((user_input, system_response))
        if len(self.memory) > self.max_turns:
            self.memory.pop(0)

    def get_context(self):
        return "\n".join([f"User: {u}\nSystem: {s}" for u, s in self.memory])


class HybridRetriever:
    def __init__(self, faiss_index, metadata, device='cpu', alpha=0.5):
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.alpha = alpha  # weighting semantic vs lexical
        self.device = device

        # Prepare BM25 corpus tokens
        self.bm25_corpus = [doc['text'].lower().split() for doc in metadata]
        self.bm25 = BM25Okapi(self.bm25_corpus)

        # Load re-ranker model/tokenizer
        print("Loading re-ranker model...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(device)

    def query(self, question, top_k=8):
        # BM25 lexical scores
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Semantic vector search (embedding + Faiss)
        emb_resp = embed(model=EMBED_MODEL, input=question)
        if "embeddings" in emb_resp:
            q_vec = np.array(emb_resp["embeddings"][0], dtype="float32")
        elif "embedding" in emb_resp:
            q_vec = np.array(emb_resp["embedding"], dtype="float32")
        else:
            raise RuntimeError("Unexpected embedding response")
        q_vec = q_vec.reshape(1, -1)

        distances, indices = self.faiss_index.search(q_vec, top_k)

        candidate_docs = []
        combined_scores = []

        for idx in indices[0]:
            if idx >= len(self.metadata):
                continue
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
            semantic_score = -distances[0][np.where(indices[0] == idx)[0][0]]
            combined_score = self.alpha * semantic_score + (1 - self.alpha) * bm25_score
            combined_scores.append(combined_score)
            candidate_docs.append(self.metadata[idx]["text"])

        # Re-rank candidates with cross-encoder for best relevance
        reranked_docs = self.rerank(question, candidate_docs)

        return reranked_docs

    def rerank(self, query, candidates):
        inputs = self.reranker_tokenizer([query] * len(candidates), candidates, padding=True, truncation=True,
                                         return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            # scores = outputs.logits[:, 1].cpu().numpy() # relevance class logit
            scores = outputs.logits.squeeze(-1).cpu().numpy()  # New code in V7
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked]


def index_folder(
        folder_path: str,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        embed_model: str = EMBED_MODEL,
        text_extensions: List[str] = None
) -> None:
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

    for file_idx, path in enumerate(files, 1):
        ext = os.path.splitext(path)[1].lower()
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        print(f"\nProcessing [{file_idx}/{len(files)}]: {os.path.basename(path)} ({file_size_mb:.2f} MB)")

        perf_monitor.sample_resources()

        if ext == ".log":
            print(f"  Using streaming log parser...")# New code in V7 for next 2 lines.
            # if (i // BATCH_SIZE) % 20 == 0:
            #     print(f"    Processed {min(i + BATCH_SIZE, len(log_chunks))}/{len(log_chunks)} chunks...")
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
                    quantizer = faiss.IndexFlatL2(d)

                    # Dynamically set number of clusters (nlist)
                    nlist = min(100, len(valid_vecs))

                    if len(valid_vecs) < 2:
                        # For tiny batches, skip IVF and just use Flat index
                        index = faiss.IndexFlatL2(d)
                        index.add(np.stack(valid_vecs))
                        print(f"FAISS FlatL2 index created (too few vectors for IVF, {len(valid_vecs)} vectors).")
                    else:
                        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                        index.train(np.stack(valid_vecs))
                        index.add(np.stack(valid_vecs))
                        print(f"FAISS IVF Flat index trained with {nlist} clusters.")

                # if index is None:
                #     d = valid_vecs[0].shape[0]
                #     # Use IVF index for speed on larger datasets
                #     quantizer = faiss.IndexFlatL2(d)
                #     index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
                #     index.train(np.stack(valid_vecs))
                #     print("FAISS IVF Flat index trained.")

                xb = np.stack(valid_vecs)
                index.add(xb) # New code in V7 for next 3 lines.
                # Log progress every N batches
                # if (i // BATCH_SIZE) % 20 == 0:  # every 20 batches ≈ every 20 * BATCH_SIZE chunks
                #     print(f"    Processed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")
                perf_monitor.metrics["indexing"]["embeddings_generated"] += len(valid_vecs)

                for j, vec in enumerate(emb_results):
                    if vec is not None:
                        metadata.append({
                            "path": path,
                            "chunk_index": i + j,
                            "text": batch[j]["text"][:1000],
                            "file_type": "log",
                            "start_line": batch[j]["start_line"],
                            "end_line": batch[j]["end_line"],
                            "log_levels": batch[j]["log_levels"],
                            "line_count": batch[j]["line_count"]
                        })

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
                    quantizer = faiss.IndexFlatL2(d)

                    # Dynamically set number of clusters (nlist)
                    nlist = min(100, len(valid_vecs))

                    if len(valid_vecs) < 2:
                        # For tiny batches, skip IVF and just use Flat index
                        index = faiss.IndexFlatL2(d)
                        index.add(np.stack(valid_vecs))
                        print(f"FAISS FlatL2 index created (too few vectors for IVF, {len(valid_vecs)} vectors).")
                    else:
                        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                        index.train(np.stack(valid_vecs))
                        index.add(np.stack(valid_vecs))
                        print(f"FAISS IVF Flat index trained with {nlist} clusters.")

                # if index is None:
                #     d = valid_vecs[0].shape[0]
                #     quantizer = faiss.IndexFlatL2(d)
                #     index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
                #     index.train(np.stack(valid_vecs))
                #     print("FAISS IVF Flat index trained.")

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
    if index is None or index.ntotal == 0:
        raise RuntimeError("No embeddings created. Check Ollama is running.")

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    perf_monitor.end_indexing()
    perf_monitor.print_indexing_report()
    perf_monitor.save_to_file()


def load_index(
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH
) -> Tuple[faiss.IndexIVFFlat, List[Dict]]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found at {index_path}. Run with --index first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run with --index first.")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def query_index(
        question: str,
        k: int = 8,
        filter_log_level: Optional[str] = None,
        conversational_memory=None,
        retriever=None
) -> str:
    query_start = time.time()
    success = False

    try:
        print("Loading index and metadata...")
        if retriever is None:
            index, metadata = load_index()
            retriever = HybridRetriever(index, metadata)

        # Prepend conversational memory context if available
        if conversational_memory:
            context = conversational_memory.get_context()
            full_query = f"{context}\n\n{question}"
        else:
            full_query = question

        print("Performing hybrid retrieval + reranking...")
        results = retriever.query(full_query, top_k=k)

        if not results:
            return "No relevant documents found."

        # Optionally filter results for log levels
        filtered_results = []
        for res_text in results:
            if filter_log_level and f"Log Levels: {filter_log_level}" not in res_text:
                continue
            filtered_results.append(res_text)

        if not filtered_results:
            return "No relevant documents found after filtering."

        # Build prompt and generate answer
        prompt = build_prompt_within_token_limit(filtered_results, full_query)
        print(f"Generating answer with {TEXT_MODEL} at temperature {temperature_used}...")

        response = chat(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature_used,
                "top_p": 0.9,
                "top_k": 40
            }
        )
        success = True
        return response["message"]["content"]

    finally:
        duration = time.time() - query_start
        perf_monitor.record_query(question, duration, k, success)
        print(f"\n⏱️  Query completed in {duration:.3f} seconds")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced RAG system (Level 3 features)",
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
                        help="Show performance statistics from prior runs")

    args = parser.parse_args()

    if args.show_stats:
        try:
            with open(PERFORMANCE_LOG, 'r') as f:
                perf_monitor.metrics = json.load(f)
            perf_monitor.print_indexing_report()
            perf_monitor.print_query_stats()
        except FileNotFoundError:
            print("No performance data found. Run index or query first.")
        return

    if not args.index and not args.query:
        parser.print_help()
        return

    if args.index:
        print("Checking and pulling models...")

        # ✅ Only pull models that exist in Ollama
        for model_name in [EMBED_MODEL, TEXT_MODEL]:
            try:
                print(f"  Pulling {model_name}...")
                pull(model_name)
                print(f"  ✓ {model_name} ready")
            except Exception as e:
                print(f"  Warning: Could not pull {model_name}: {e}")

        # ✅ Load reranker locally from Hugging Face (not via Ollama)
        try:
            print(f"Loading reranker {RERANKER_MODEL_NAME} locally...")
            _ = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
            _ = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
            print("  ✓ Reranker ready (local load successful)")
        except Exception as e:
            print(f"  Warning: Could not load reranker model: {e}")

        print(f"Indexing folder: {args.folder}")
        index_folder(args.folder)

    if args.query:
        # Load index and metadata, build hybrid retriever
        index, metadata = load_index()
        retriever = HybridRetriever(index, metadata)

        # Initialize conversational memory for session (could be extended)
        conv_memory = ConversationalMemory(max_turns=5)

        answer = query_index(
            args.query,
            k=args.k,
            filter_log_level=args.log_level,
            conversational_memory=conv_memory,
            retriever=retriever
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60 + "\n")
        print(answer + "\n")

        perf_monitor.print_query_stats()


if __name__ == "__main__":
    main()
