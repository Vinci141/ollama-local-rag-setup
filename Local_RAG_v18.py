#!/usr/bin/env python3
"""
🚀 TOON-Enhanced RAG System - Version 18.0 (Multi-Query Retrieval)

NEW FEATURES (18.0):
✅ Multi-Query Retrieval - Decomposes complex questions into sub-queries
✅ Parallel sub-query execution for faster retrieval
✅ Intelligent result merging with deduplication
✅ Enhanced re-ranking across all sub-query results
✅ Automatic multi-query detection (complex vs simple questions)

NEW FEATURES (17.0):
✅ Async/await for all I/O operations (Ollama, file reading)
✅ ProcessPoolExecutor for CPU-intensive tasks (chunking, BM25 indexing)
✅ Disk-based caching for embeddings and LLM responses (diskcache)
✅ Reciprocal Rank Fusion (RRF) for hybrid search merging
✅ Query refinement via LLM before retrieval
✅ Cross-encoder re-ranking (top 20 → best 5)
✅ Modular architecture: Config, IngestionEngine, RetrievalEngine, RAGPipeline
✅ Structured logging with per-step latency tracking
✅ Comprehensive error handling (Ollama timeouts, FAISS dimension mismatches)
✅ Optimized FAISS index selection (IndexFlatIP for small, IVFFlat for large)

RETAINED FEATURES:
✅ TOON (Task-Oriented Orchestration Network) integration
✅ Knowledge graph for file relationships
✅ Hybrid FAISS + BM25S retrieval
✅ Robust PDF/DOCX/XLSX extraction
✅ PII detection and redaction
✅ Performance monitoring and audit logging

Dependencies:
pip install sentence-transformers torch faiss-cpu numpy pymupdf bm25s PyStemmer psutil transformers ollama networkx spacy aiohttp aiofiles diskcache python-docx pandas openpyxl
python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import os
import sys
import warnings

# Suppress TensorFlow warnings (must be set before imports)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fix Windows console encoding for UTF-8/emoji support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suppress harmless "resource module not available on Windows" warning from bm25s
warnings.filterwarnings("ignore", message="resource module not available")

import asyncio
import glob
import hashlib
import json
import logging
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiofiles
import aiohttp
import diskcache
import docx
import faiss
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import spacy
import torch
import bm25s
from Stemmer import Stemmer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent


# ============================================================================
# VERSION DETECTION
# ============================================================================

def get_version_from_filename() -> str:
    """
    Extract version number from the script filename (e.g., 'Local_RAG_v18.py' -> '18').
    Falls back to '18' if version cannot be detected.
    """
    # Try multiple methods to get the script path
    script_path = None
    
    # Method 1: Use __file__ if available (most reliable)
    try:
        if '__file__' in globals():
            script_path = __file__
    except NameError:
        pass
    
    # Method 2: Use sys.argv[0]
    if not script_path and sys.argv:
        script_path = sys.argv[0]
    
    # Method 3: Use inspect module as last resort
    if not script_path:
        try:
            import inspect
            script_path = inspect.getfile(inspect.currentframe())
        except:
            pass
    
    if script_path:
        script_name = os.path.basename(script_path)
        # Try to extract version from filename (e.g., Local_RAG_v18.py -> 18)
        match = re.search(r'[vV](\d+)', script_name)
        if match:
            return match.group(1)
    
    # Default fallback
    return "18"


def get_dynamic_directories() -> Tuple[str, str]:
    """
    Generate dynamic directory names based on script version.
    Returns: (index_dir, cache_dir)
    """
    version = get_version_from_filename()
    return f"index_v{version}", f"cache_v{version}"


def remove_empty_directory(dir_path: str) -> bool:
    """
    Remove a directory if it exists and is empty.
    Returns True if directory was removed, False otherwise.
    """
    try:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Check if directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                return True
    except OSError:
        pass
    return False


# ============================================================================
# CONFIGURATION (Centralized Dataclass)
# ============================================================================

@dataclass
class Config:
    """Centralized configuration for the RAG system."""
    
    # Model paths (local)
    embed_model_name: str = "local_models/all-MiniLM-L6-v2"
    reranker_model_name: str = "local_models/ms-marco-TinyBERT-L2-v2"
    text_model: str = "gemma3"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300  # seconds (5 min for first load)
    
    # Chunking parameters
    chunk_size: int = 2000
    chunk_overlap: int = 250
    log_lines_per_chunk: int = 100
    
    # Batch processing
    batch_size: int = 64
    embedding_batch_size: int = 32
    
    # Retrieval parameters
    top_k_retrieval: int = 20  # Initial retrieval before re-ranking
    top_k_final: int = 5       # Final results after re-ranking
    rrf_k: int = 60            # RRF constant (typically 60)
    
    # Multi-query parameters
    enable_multi_query: bool = True  # Enable multi-query decomposition
    max_sub_queries: int = 3          # Maximum sub-queries to generate
    multi_query_threshold: int = 15   # Min query length to trigger multi-query
    
    # Paths (dynamically set based on script version in __post_init__)
    index_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    index_path: str = field(init=False)
    metadata_path: str = field(init=False)
    file_registry_path: str = field(init=False)  # Tracks indexed files
    knowledge_graph_path: str = "knowledge_graph.json"
    performance_log: str = "performance_metrics.json"
    audit_log: str = "audit_log.json"
    
    # Limits
    max_file_size_mb: int = 500
    max_tokens: int = 16000
    
    # Generation parameters
    temperature: float = 0.2
    top_p: float = 0.9
    top_k_generation: int = 40
    
    # Device settings
    device: str = field(init=False)
    use_gpu: bool = field(init=False)
    
    # Cache settings
    cache_size_limit: int = 10 * 1024 * 1024 * 1024  # 10GB
    embedding_cache_enabled: bool = True
    llm_cache_enabled: bool = True
    
    # Parallel processing
    max_workers: int = field(init=False)
    
    def __post_init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.max_workers = min(os.cpu_count() or 4, 8)
        
        # Set dynamic directory names if not already set
        if self.index_dir is None:
            self.index_dir = get_dynamic_directories()[0]
        if self.cache_dir is None:
            self.cache_dir = get_dynamic_directories()[1]
        
        self.index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.index_dir, "metadata.json")
        self.file_registry_path = os.path.join(self.index_dir, "file_registry.json")
        
        # Note: Directories are created only when actually needed, not here
        # This prevents creating empty directories for operations like --warmup, --cache-stats


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LatencyLogger:
    """Structured logger with per-step latency tracking."""
    
    def __init__(self, name: str = "RAGv17"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self._step_times: Dict[str, float] = {}
    
    def start_step(self, step_name: str) -> None:
        """Start timing a step."""
        self._step_times[step_name] = time.perf_counter()
        self.logger.info(f"⏱️  Starting: {step_name}")
    
    def end_step(self, step_name: str) -> float:
        """End timing a step and log the duration."""
        if step_name not in self._step_times:
            self.logger.warning(f"Step '{step_name}' was never started")
            return 0.0
        
        duration = time.perf_counter() - self._step_times[step_name]
        self.logger.info(f"✅ Completed: {step_name} ({duration:.3f}s)")
        del self._step_times[step_name]
        return duration
    
    def info(self, msg: str) -> None:
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        self.logger.error(msg)
    
    def debug(self, msg: str) -> None:
        self.logger.debug(msg)


# Global logger instance
logger = LatencyLogger()


# ============================================================================
# CACHING LAYER
# ============================================================================

class CacheManager:
    """Disk-based caching for embeddings and LLM responses."""
    
    def __init__(self, config: Config):
        self.config = config
        self._embedding_cache: Optional[diskcache.Cache] = None
        self._llm_cache: Optional[diskcache.Cache] = None
        
        if config.embedding_cache_enabled:
            self._embedding_cache = diskcache.Cache(
                os.path.join(config.cache_dir, "embeddings"),
                size_limit=config.cache_size_limit // 2
            )
        
        if config.llm_cache_enabled:
            self._llm_cache = diskcache.Cache(
                os.path.join(config.cache_dir, "llm_responses"),
                size_limit=config.cache_size_limit // 2
            )
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding."""
        if not self._embedding_cache:
            return None
        
        key = self._hash_text(text)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return np.frombuffer(cached, dtype=np.float32)
        return None
    
    def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        if not self._embedding_cache:
            return
        
        key = self._hash_text(text)
        self._embedding_cache.set(key, embedding.tobytes())
    
    def get_llm_response(self, prompt: str, model: str) -> Optional[str]:
        """Retrieve cached LLM response."""
        if not self._llm_cache:
            return None
        
        key = self._hash_text(f"{model}:{prompt}")
        return self._llm_cache.get(key)
    
    def set_llm_response(self, prompt: str, model: str, response: str) -> None:
        """Cache an LLM response."""
        if not self._llm_cache:
            return
        
        key = self._hash_text(f"{model}:{prompt}")
        self._llm_cache.set(key, response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {}
        if self._embedding_cache:
            stats['embedding_cache_size'] = len(self._embedding_cache)
        if self._llm_cache:
            stats['llm_cache_size'] = len(self._llm_cache)
        return stats
    
    def clear(self) -> None:
        """Clear all caches."""
        if self._embedding_cache:
            self._embedding_cache.clear()
        if self._llm_cache:
            self._llm_cache.clear()
    
    def close(self) -> None:
        """Close cache connections."""
        if self._embedding_cache:
            self._embedding_cache.close()
        if self._llm_cache:
            self._llm_cache.close()


# ============================================================================
# FILE REGISTRY (Track indexed files for incremental indexing)
# ============================================================================

class FileRegistry:
    """Tracks indexed files to enable incremental indexing."""
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.files: Dict[str, Dict[str, Any]] = {}  # path -> {mtime, size, hash, chunk_indices}
        self.load()
    
    def load(self) -> None:
        """Load registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.files = json.load(f)
            except Exception:
                self.files = {}
    
    def save(self) -> None:
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.files, f, indent=2)
    
    def get_file_hash(self, path: str) -> str:
        """Get fast hash of file (first 8KB + size)."""
        try:
            size = os.path.getsize(path)
            with open(path, 'rb') as f:
                header = f.read(8192)
            return hashlib.md5(header + str(size).encode()).hexdigest()
        except Exception:
            return ""
    
    def is_file_changed(self, path: str) -> bool:
        """Check if file is new or modified."""
        if path not in self.files:
            return True
        
        try:
            current_mtime = os.path.getmtime(path)
            current_size = os.path.getsize(path)
            
            stored = self.files[path]
            if stored.get('mtime') != current_mtime or stored.get('size') != current_size:
                # Size or mtime changed, verify with hash
                current_hash = self.get_file_hash(path)
                return current_hash != stored.get('hash')
            
            return False
        except Exception:
            return True
    
    def register_file(self, path: str, chunk_start_idx: int, chunk_count: int) -> None:
        """Register a file as indexed."""
        self.files[path] = {
            'mtime': os.path.getmtime(path),
            'size': os.path.getsize(path),
            'hash': self.get_file_hash(path),
            'chunk_start_idx': chunk_start_idx,
            'chunk_count': chunk_count,
            'indexed_at': datetime.now().isoformat()
        }
    
    def get_files_to_reindex(self, current_files: List[str]) -> Tuple[List[str], List[str]]:
        """
        Compare current files with registry.
        Returns: (new_or_modified_files, deleted_files)
        """
        current_set = set(current_files)
        indexed_set = set(self.files.keys())
        
        # Files that are new or modified
        new_or_modified = [f for f in current_files if self.is_file_changed(f)]
        
        # Files that were deleted
        deleted = list(indexed_set - current_set)
        
        return new_or_modified, deleted
    
    def remove_file(self, path: str) -> Optional[Dict]:
        """Remove file from registry, return its info."""
        return self.files.pop(path, None)
    
    def clear(self) -> None:
        """Clear registry for full rebuild."""
        self.files = {}
        self.save()


# ============================================================================
# ASYNC OLLAMA CLIENT
# ============================================================================

class AsyncOllamaClient:
    """Async client for Ollama API with timeout handling."""
    
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.ollama_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """Generate text using Ollama API asynchronously."""
        model = model or self.config.text_model
        
        # Check cache first
        if use_cache:
            cached = self.cache.get_llm_response(prompt, model)
            if cached:
                logger.debug(f"LLM cache hit for prompt hash")
                return cached
        
        session = await self._get_session()
        url = f"{self.config.ollama_base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k_generation
            }
        }
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error ({response.status}): {error_text}")
                
                result = await response.json()
                generated_text = result.get("response", "")
                
                # Cache the response
                if use_cache and generated_text:
                    self.cache.set_llm_response(prompt, model, generated_text)
                
                return generated_text
                
        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after {self.config.ollama_timeout}s")
            raise TimeoutError(f"Ollama request timed out after {self.config.ollama_timeout} seconds")
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """Chat completion using Ollama API asynchronously."""
        model = model or self.config.text_model
        
        # Create cache key from messages
        cache_key = json.dumps(messages, sort_keys=True)
        
        if use_cache:
            cached = self.cache.get_llm_response(cache_key, model)
            if cached:
                logger.debug("LLM cache hit for chat")
                return cached
        
        session = await self._get_session()
        url = f"{self.config.ollama_base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k_generation
            }
        }
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error ({response.status}): {error_text}")
                
                result = await response.json()
                generated_text = result.get("message", {}).get("content", "")
                
                if use_cache and generated_text:
                    self.cache.set_llm_response(cache_key, model, generated_text)
                
                return generated_text
                
        except asyncio.TimeoutError:
            logger.error(f"Ollama chat timed out after {self.config.ollama_timeout}s")
            raise TimeoutError(f"Ollama chat timed out after {self.config.ollama_timeout} seconds")
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ============================================================================
# TEXT EXTRACTION (Async File Reading)
# ============================================================================

async def read_file_async(path: str) -> str:
    """Read file content asynchronously."""
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return await f.read()
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return ""


def extract_text_from_pdf_robust(path: str) -> str:
    """Enhanced PDF text extraction with multiple strategies."""
    doc = None
    text = ""
    
    try:
        fitz.TOOLS.reset_mupdf_warnings()
        doc = fitz.open(path)
        
        total_pages = len(doc)
        
        # Strategy 1: Standard extraction
        try:
            text_parts = []
            for page_num in range(total_pages):
                page_text = doc.load_page(page_num).get_text("text")
                if page_text:
                    text_parts.append(page_text)
            
            text = "\n\n".join(text_parts)
            if len(text.strip()) > 10:
                doc.close()
                return text
        except Exception:
            pass
        
        # Strategy 2: Block-based extraction
        try:
            text_parts = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) >= 5 and isinstance(block[4], str):
                        text_parts.append(block[4])
            
            text = "\n".join(text_parts)
            if len(text.strip()) > 10:
                doc.close()
                return text
        except Exception:
            pass
        
        doc.close()
        
    except Exception as e:
        logger.warning(f"PDF extraction failed for {path}: {e}")
        if doc:
            doc.close()
    
    return text


def extract_text_from_docx(path: str) -> str:
    """Extract text from Word documents."""
    try:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.warning(f"Could not read DOCX {path}: {e}")
        return ""


def extract_text_from_xlsx(path: str) -> str:
    """Extract text from Excel files."""
    try:
        df = pd.read_excel(path, sheet_name=None)
        text_parts = []
        for sheet, data in df.items():
            text_parts.append(f"Sheet: {sheet}\n")
            text_parts.append(data.to_string(index=False))
            text_parts.append("\n\n")
        return "".join(text_parts)
    except Exception as e:
        logger.warning(f"Could not read XLSX {path}: {e}")
        return ""


async def extract_text_from_file(path: str) -> str:
    """Extract text from various file formats asynchronously."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext in [".log"]:
        return ""  # Handled separately
    
    # Text files - async read
    if ext in [".txt", ".md", ".py", ".json", ".csv"]:
        return await read_file_async(path)
    
    # PDF files - sync (fitz not async-compatible)
    if ext == ".pdf":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_text_from_pdf_robust, path)
    
    # Word files
    if ext == ".docx":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_text_from_docx, path)
    
    # Excel files
    if ext == ".xlsx":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_text_from_xlsx, path)
    
    return ""


# ============================================================================
# CHUNKING (CPU-Intensive - Runs in ProcessPoolExecutor)
# ============================================================================

def chunk_text_worker(args: Tuple[str, int, int]) -> List[str]:
    """Worker function for parallel text chunking."""
    text, chunk_size, overlap = args
    
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


def chunk_log_file_worker(args: Tuple[str, int]) -> List[Dict]:
    """Worker function for parallel log file chunking."""
    filepath, lines_per_chunk = args
    chunks = []
    current_chunk_lines = []
    current_chunk_metadata = {"start_line": 0, "log_levels": set()}
    line_number = 0
    
    log_patterns = [
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(\w+)\s+\[(.*?)\]\s+(.*)',
        r'([\d\.]+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)',
        r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(.*)',
        r'(\w+):\s+(.*)',
    ]
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:
                    continue
                
                current_chunk_lines.append(line)
                
                # Detect log levels
                for pattern in log_patterns:
                    match = re.match(pattern, line)
                    if match:
                        for group in match.groups():
                            group_upper = str(group).upper()
                            for level in ["ERROR", "WARN", "INFO", "DEBUG", "FATAL"]:
                                if level in group_upper:
                                    current_chunk_metadata["log_levels"].add(level)
                        break
                
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
                    current_chunk_metadata = {"start_line": line_number + 1, "log_levels": set()}
        
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
        pass
    
    return chunks


# ============================================================================
# EMBEDDING MANAGER (With Caching)
# ============================================================================

class EmbeddingManager:
    """Manages embeddings with caching support."""
    
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
    
    def load_model(self) -> None:
        """Load the embedding model."""
        if self.model is not None:
            return
        
        logger.start_step("Loading embedding model")
        self.model = SentenceTransformer(self.config.embed_model_name)
        self.model.to(self.config.device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.end_step("Loading embedding model")
        logger.info(f"Embedding model loaded on {self.config.device.upper()} (dim={self._dimension})")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self.load_model()
        return self._dimension
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed texts with caching."""
        if self.model is None:
            self.load_model()
        
        batch_size = batch_size or self.config.embedding_batch_size
        
        # Check cache for each text
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cached = self.cache.get_embedding(text)
            if cached is not None and len(cached) == self._dimension:
                embeddings.append((i, cached))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=batch_size,
                show_progress_bar=show_progress and len(texts_to_embed) > 10,
                convert_to_numpy=True,
                device=self.config.device
            )
            
            # Cache new embeddings
            for idx, (text_idx, text) in enumerate(zip(text_indices, texts_to_embed)):
                emb = new_embeddings[idx]
                self.cache.set_embedding(text, emb)
                embeddings.append((text_idx, emb))
        
        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in embeddings])
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed_texts([text], show_progress=False)[0]


# ============================================================================
# CROSS-ENCODER RE-RANKER
# ============================================================================

class CrossEncoderReranker:
    """Lightweight cross-encoder for re-ranking retrieved documents."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
    
    def load_model(self) -> None:
        """Load the cross-encoder model."""
        if self.model is not None:
            return
        
        logger.start_step("Loading cross-encoder model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.reranker_model_name
        ).to(self.config.device)
        self.model.eval()
        logger.end_step("Loading cross-encoder model")
        logger.info(f"Cross-encoder loaded on {self.config.device.upper()}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Re-rank documents using cross-encoder scores."""
        if not documents:
            return []
        
        if self.model is None:
            self.load_model()
        
        # Prepare inputs
        inputs = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.config.device)
        
        # Get scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        # Handle single document case
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        
        # Rank by score
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ============================================================================
# RETRIEVAL ENGINE (Hybrid Search with RRF)
# ============================================================================

class RetrievalEngine:
    """
    Hybrid retrieval engine combining FAISS (dense) and BM25 (sparse) 
    with Reciprocal Rank Fusion (RRF) for result merging.
    """
    
    def __init__(
        self, 
        config: Config,
        embedding_manager: EmbeddingManager,
        reranker: CrossEncoderReranker
    ):
        self.config = config
        self.embedding_manager = embedding_manager
        self.reranker = reranker
        
        self.faiss_index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.bm25: Optional[bm25s.BM25] = None
        self.stemmer: Optional[Stemmer] = None
    
    def build_index(self, chunks: List[str], metadata: List[Dict]) -> None:
        """Build FAISS and BM25 indices."""
        if len(chunks) != len(metadata):
            raise ValueError(f"Chunks ({len(chunks)}) and metadata ({len(metadata)}) length mismatch")
        
        self.metadata = metadata
        
        # Generate embeddings
        logger.start_step("Generating embeddings")
        embeddings = self.embedding_manager.embed_texts(chunks)
        logger.end_step("Generating embeddings")
        
        # Validate dimensions
        expected_dim = self.embedding_manager.dimension
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}"
            )
        
        # Build FAISS index
        logger.start_step("Building FAISS index")
        self.faiss_index = self._create_faiss_index(embeddings)
        logger.end_step("Building FAISS index")
        
        # Build BM25 index
        logger.start_step("Building BM25 index")
        self.stemmer = Stemmer("english")
        corpus_tokens = bm25s.tokenize(chunks, stemmer=self.stemmer)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)
        logger.end_step("Building BM25 index")
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create optimized FAISS index based on dataset size."""
        d = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Normalize embeddings for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(embeddings)
        
        if n_vectors < 10000:
            # Use exact search for small datasets
            index = faiss.IndexFlatIP(d)
            index.add(embeddings.astype(np.float32))
            logger.info(f"Created FAISS IndexFlatIP for {n_vectors} vectors")
        else:
            # Use IVF for larger datasets
            nlist = min(int(4 * np.sqrt(n_vectors)), 256)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(20, nlist)
            logger.info(f"Created FAISS IVFFlat with {nlist} clusters for {n_vectors} vectors")
        
        return index
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        faiss.write_index(self.faiss_index, index_path)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """Load index and metadata from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        logger.start_step("Loading FAISS index")
        self.faiss_index = faiss.read_index(index_path)
        logger.end_step("Loading FAISS index")
        
        # Validate dimensions
        index_dim = self.faiss_index.d
        expected_dim = self.embedding_manager.dimension
        if index_dim != expected_dim:
            raise ValueError(
                f"FAISS index dimension ({index_dim}) doesn't match "
                f"embedding model dimension ({expected_dim})"
            )
        
        logger.start_step("Loading metadata")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.end_step("Loading metadata")
        
        # Build BM25 index
        logger.start_step("Building BM25 index")
        self.stemmer = Stemmer("english")
        corpus_texts = [doc['text'] for doc in self.metadata]
        corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=self.stemmer)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)
        logger.end_step("Building BM25 index")
    
    def _reciprocal_rank_fusion(
        self, 
        rankings: List[List[int]], 
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Merge multiple rankings using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank_i)) for each ranking list
        """
        rrf_scores: Dict[int, float] = {}
        
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
    
    def retrieve(
        self, 
        query: str, 
        top_k_retrieval: Optional[int] = None,
        top_k_final: Optional[int] = None,
        rerank: bool = True
    ) -> List[Tuple[str, Dict, float]]:
        """
        Hybrid retrieval with RRF fusion and cross-encoder re-ranking.
        
        Returns: List of (text, metadata, score) tuples
        """
        top_k_retrieval = top_k_retrieval or self.config.top_k_retrieval
        top_k_final = top_k_final or self.config.top_k_final
        
        if not self.faiss_index or not self.bm25:
            raise RuntimeError("Index not loaded. Call load() or build_index() first.")
        
        n_docs = len(self.metadata)
        max_results = min(top_k_retrieval, n_docs)
        
        # FAISS (dense) retrieval
        logger.start_step("FAISS retrieval")
        query_embedding = self.embedding_manager.embed_single(query).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        _, faiss_indices = self.faiss_index.search(query_embedding.astype(np.float32), max_results)
        faiss_ranking = [int(idx) for idx in faiss_indices[0] if 0 <= idx < n_docs]
        faiss_time = logger.end_step("FAISS retrieval")
        
        # BM25 (sparse) retrieval
        logger.start_step("BM25 retrieval")
        query_tokens = bm25s.tokenize([query], stemmer=self.stemmer)
        bm25_results, _ = self.bm25.retrieve(query_tokens, k=max_results)
        bm25_ranking = [int(idx) for idx in bm25_results[0] if 0 <= idx < n_docs]
        bm25_time = logger.end_step("BM25 retrieval")
        
        # Reciprocal Rank Fusion
        logger.start_step("RRF fusion")
        fused_results = self._reciprocal_rank_fusion(
            [faiss_ranking, bm25_ranking], 
            k=self.config.rrf_k
        )
        rrf_time = logger.end_step("RRF fusion")
        
        # Get top candidates for re-ranking
        candidate_indices = [idx for idx, _ in fused_results[:top_k_retrieval]]
        candidates = [(self.metadata[idx]['text'], self.metadata[idx]) for idx in candidate_indices]
        
        if not candidates:
            return []
        
        # Cross-encoder re-ranking
        if rerank and len(candidates) > top_k_final:
            logger.start_step("Cross-encoder re-ranking")
            candidate_texts = [c[0] for c in candidates]
            reranked = self.reranker.rerank(query, candidate_texts, top_k_final)
            rerank_time = logger.end_step("Cross-encoder re-ranking")
            
            # Map back to metadata
            text_to_meta = {c[0]: c[1] for c in candidates}
            results = [(text, text_to_meta[text], score) for text, score in reranked]
        else:
            # Take top results without re-ranking
            results = [
                (self.metadata[idx]['text'], self.metadata[idx], score) 
                for idx, score in fused_results[:top_k_final]
            ]
        
        logger.info(f"Retrieved {len(results)} documents (FAISS: {faiss_time:.3f}s, BM25: {bm25_time:.3f}s)")
        return results


# ============================================================================
# INGESTION ENGINE
# ============================================================================

class IngestionEngine:
    """Handles document ingestion with parallel processing."""
    
    def __init__(self, config: Config, embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        self._pii_detector: Optional[PIIDetector] = None
    
    @property
    def pii_detector(self) -> 'PIIDetector':
        """Lazy load PII detector."""
        if self._pii_detector is None:
            self._pii_detector = PIIDetector()
        return self._pii_detector
    
    async def ingest_files(
        self, 
        files: List[str],
        knowledge_graph: Optional['KnowledgeGraph'] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Ingest specific files asynchronously.
        
        Args:
            files: List of file paths to process
            knowledge_graph: Optional knowledge graph to update
        
        Returns: (chunks, metadata)
        """
        if not files:
            return [], []
        
        logger.info(f"Processing {len(files)} files")
        
        all_chunks: List[str] = []
        all_metadata: List[Dict] = []
        
        for idx, path in enumerate(files, 1):
            ext = os.path.splitext(path)[1].lower()
            file_size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
            
            logger.info(f"[{idx}/{len(files)}] Processing: {os.path.basename(path)} ({file_size_mb:.2f} MB)")
            
            try:
                if ext == ".log":
                    loop = asyncio.get_event_loop()
                    with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                        log_chunks = await loop.run_in_executor(
                            executor,
                            chunk_log_file_worker,
                            (path, self.config.log_lines_per_chunk)
                        )
                    
                    for chunk in log_chunks:
                        all_chunks.append(chunk["text"])
                        all_metadata.append({
                            "path": path,
                            "text": chunk["text"][:1000],
                            "file_type": "log",
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "log_levels": chunk["log_levels"]
                        })
                    
                    if knowledge_graph:
                        knowledge_graph.add_file_node(path, {
                            'file_type': 'log',
                            'chunk_count': len(log_chunks)
                        })
                else:
                    text = await extract_text_from_file(path)
                    if not text or len(text.strip()) < 10:
                        logger.warning(f"   Skipping (no text extracted)")
                        continue
                    
                    try:
                        text = self.pii_detector.redact_pii(text)
                    except Exception as e:
                        logger.warning(f"   PII redaction failed: {e}")
                    
                    loop = asyncio.get_event_loop()
                    with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                        chunks = await loop.run_in_executor(
                            executor,
                            chunk_text_worker,
                            (text, self.config.chunk_size, self.config.chunk_overlap)
                        )
                    
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_metadata.append({
                            "path": path,
                            "text": chunk[:1000],
                            "file_type": ext[1:] if ext else "unknown"
                        })
                    
                    if knowledge_graph:
                        knowledge_graph.add_file_node(path, {
                            'file_type': ext[1:] if ext else "unknown",
                            'chunk_count': len(chunks)
                        })
                    
                    logger.info(f"   Created {len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"   Error processing file: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks, all_metadata


# ============================================================================
# QUERY REFINEMENT
# ============================================================================

class QueryRefiner:
    """Refines user queries for better retrieval."""
    
    REFINEMENT_PROMPT = """You are a search query optimizer. Your task is to rewrite the user's question to be more effective for semantic search.

Rules:
1. Keep the core meaning but make it more specific
2. Add relevant keywords that might appear in documents
3. Remove filler words and conversational elements
4. Keep it concise (1-2 sentences max)
5. Output ONLY the refined query, nothing else

Original query: {query}

Refined query:"""
    
    def __init__(self, ollama_client: AsyncOllamaClient):
        self.ollama = ollama_client
    
    async def refine(self, query: str) -> str:
        """Refine a query for better retrieval."""
        try:
            prompt = self.REFINEMENT_PROMPT.format(query=query)
            refined = await self.ollama.generate(prompt, use_cache=True)
            refined = refined.strip().strip('"\'')
            
            # Fallback to original if refinement is too short or failed
            if len(refined) < 5 or len(refined) > len(query) * 3:
                return query
            
            logger.info(f"Query refined: '{query}' → '{refined}'")
            return refined
            
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return query


# ============================================================================
# MULTI-QUERY RETRIEVAL
# ============================================================================

class MultiQueryRetriever:
    """
    Decomposes complex questions into multiple sub-queries for better retrieval.
    Handles questions like "Compare X and Y" or "What changed between A and B?"
    """
    
    DECOMPOSE_PROMPT = """You are a query decomposition expert. Break complex questions into simpler search queries.

Rules:
1. If the question asks to compare/contrast, create separate queries for each item
2. If asking about changes/differences, create queries for each time period/version
3. If asking multiple things, split into focused sub-queries
4. Each sub-query should be a complete, searchable question
5. Output ONLY the sub-queries, one per line, no numbering or bullets
6. Maximum {max_queries} sub-queries

Example:
Question: "Compare machine learning and deep learning approaches"
Sub-queries:
What is machine learning?
What is deep learning?
How do machine learning and deep learning differ?

Question: {query}

Sub-queries:"""
    
    def __init__(self, ollama_client: AsyncOllamaClient, retrieval_engine: 'RetrievalEngine', config: Config):
        self.ollama = ollama_client
        self.retrieval_engine = retrieval_engine
        self.config = config
    
    def _should_use_multi_query(self, query: str) -> bool:
        """Determine if query is complex enough for multi-query decomposition."""
        if not self.config.enable_multi_query:
            return False
        
        # Simple heuristics
        query_lower = query.lower()
        complex_indicators = [
            'compare', 'contrast', 'difference', 'versus', 'vs', 'between',
            'and', 'or', 'both', 'multiple', 'various', 'different',
            'change', 'changed', 'update', 'updated', 'before', 'after'
        ]
        
        has_complex_words = any(word in query_lower for word in complex_indicators)
        is_long = len(query.split()) >= self.config.multi_query_threshold
        
        return has_complex_words or is_long
    
    async def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        try:
            prompt = self.DECOMPOSE_PROMPT.format(
                query=query,
                max_queries=self.config.max_sub_queries
            )
            
            response = await self.ollama.generate(prompt, use_cache=True)
            
            # Parse sub-queries (one per line)
            sub_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering/bullets
                line = re.sub(r'^[\d\.\-\*\)]+[\s\)]*', '', line)
                line = line.strip()
                
                if line and len(line) > 10:  # Valid query
                    # Remove question marks if at start (LLM sometimes adds them)
                    if line.startswith('?'):
                        line = line[1:].strip()
                    sub_queries.append(line)
            
            # Limit to max_sub_queries
            sub_queries = sub_queries[:self.config.max_sub_queries]
            
            # Fallback: if decomposition failed, use original
            if not sub_queries or len(sub_queries) < 2:
                logger.debug("Multi-query decomposition produced < 2 queries, using original")
                return [query]
            
            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries, 1):
                logger.info(f"  {i}. {sq}")
            
            return sub_queries
            
        except Exception as e:
            logger.warning(f"Multi-query decomposition failed: {e}")
            return [query]
    
    async def retrieve_multi_query(
        self, 
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_final: Optional[int] = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Retrieve using multi-query approach.
        
        Returns: List of (text, metadata, score) tuples
        """
        if not self._should_use_multi_query(query):
            # Simple query - use standard retrieval
            return self.retrieval_engine.retrieve(
                query,
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
                rerank=True
            )
        
        # Decompose query
        logger.start_step("Multi-query decomposition")
        sub_queries = await self.decompose_query(query)
        logger.end_step("Multi-query decomposition")
        
        if len(sub_queries) == 1:
            # Decomposition didn't help, use standard retrieval
            return self.retrieval_engine.retrieve(
                sub_queries[0],
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
                rerank=True
            )
        
        # Retrieve for each sub-query in parallel
        logger.start_step("Parallel sub-query retrieval")
        top_k_retrieval = top_k_retrieval or self.config.top_k_retrieval
        
        # Create tasks for parallel execution (retrieval is CPU-bound, use ThreadPoolExecutor)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self.retrieval_engine.retrieve,
                    sq,
                    top_k_retrieval,
                    top_k_retrieval,  # Get more results per sub-query
                    True  # rerank
                )
                for sq in sub_queries
            ]
            
            # Wait for all retrievals
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.warning(f"Sub-query {i+1} retrieval failed: {result}")
            else:
                valid_results.extend(result)
        
        logger.end_step("Parallel sub-query retrieval")
        
        if not valid_results:
            return []
        
        # Deduplicate results (same text chunk)
        logger.start_step("Result deduplication")
        seen_texts = set()
        unique_results = []
        
        for text, metadata, score in valid_results:
            # Use first 200 chars as dedup key
            text_key = text[:200].strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append((text, metadata, score))
        
        logger.info(f"Deduplicated: {len(valid_results)} → {len(unique_results)} results")
        logger.end_step("Result deduplication")
        
        # Final re-ranking across all sub-query results
        top_k_final = top_k_final or self.config.top_k_final
        
        if len(unique_results) <= top_k_final:
            # Not enough results to re-rank
            return unique_results
        
        logger.start_step("Cross-query re-ranking")
        candidate_texts = [text for text, _, _ in unique_results]
        reranked = self.retrieval_engine.reranker.rerank(query, candidate_texts, top_k_final)
        logger.end_step("Cross-query re-ranking")
        
        # Map back to metadata (use text as key since meta is a dict and not hashable)
        text_to_result = {text: (text, meta, score) for text, meta, score in unique_results}
        final_results = []
        for text, score in reranked:
            # Find matching result
            if text in text_to_result:
                orig_text, orig_meta, orig_score = text_to_result[text]
                final_results.append((orig_text, orig_meta, float(score)))
        
        return final_results


# ============================================================================
# PII DETECTOR (Retained from v16)
# ============================================================================

class PIIDetector:
    """Detect and optionally redact PII from text."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Detect PII entities in text."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        pii_entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD"]:
                pii_entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return pii_entities
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        pii_entities = self.detect_pii(text)
        
        for entity in sorted(pii_entities, key=lambda x: x['start'], reverse=True):
            text = text[:entity['start']] + "[REDACTED]" + text[entity['end']:]
        
        return text


# ============================================================================
# KNOWLEDGE GRAPH (Retained from v16)
# ============================================================================

class KnowledgeGraph:
    """Manage relationships between files and concepts."""
    
    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        self.graph = nx.Graph()
        self.file_nodes: Dict[str, str] = {}
        self.concept_nodes: Dict[str, str] = {}
        
        if os.path.exists(graph_path):
            self.load()
    
    def add_file_node(self, file_path: str, metadata: Dict) -> str:
        """Register a file as a knowledge node."""
        node_id = f"file_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
        
        self.graph.add_node(
            node_id,
            type='file',
            path=file_path,
            file_type=metadata.get('file_type', 'unknown'),
            chunk_count=metadata.get('chunk_count', 0),
            indexed_at=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self.file_nodes[file_path] = node_id
        return node_id
    
    def add_concept_node(self, concept: str, files: List[str]) -> None:
        """Add a concept and link it to files."""
        concept_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()[:8]}"
        
        if concept_id not in self.graph:
            self.graph.add_node(
                concept_id,
                type='concept',
                name=concept,
                created_at=datetime.now().isoformat()
            )
            self.concept_nodes[concept] = concept_id
        
        for file_path in files:
            if file_path in self.file_nodes:
                self.graph.add_edge(concept_id, self.file_nodes[file_path], relationship='mentioned_in')
    
    def get_related_files(self, file_path: str, max_depth: int = 2) -> List[str]:
        """Get files related to a given file."""
        node_id = self.file_nodes.get(file_path)
        if not node_id:
            return []
        
        related = []
        for neighbor in nx.single_source_shortest_path_length(self.graph, node_id, cutoff=max_depth).keys():
            if self.graph.nodes[neighbor]['type'] == 'file':
                related.append(self.graph.nodes[neighbor]['path'])
        
        return related
    
    def save(self) -> None:
        """Save knowledge graph to file."""
        data = {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
        
        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self) -> None:
        """Load knowledge graph from file."""
        with open(self.graph_path, 'r') as f:
            data = json.load(f)
        
        self.graph = nx.Graph()
        for node_id, attrs in data['nodes']:
            self.graph.add_node(node_id, **attrs)
            if attrs['type'] == 'file':
                self.file_nodes[attrs['path']] = node_id
            elif attrs['type'] == 'concept':
                self.concept_nodes[attrs['name']] = node_id
        
        for source, target, attrs in data['edges']:
            self.graph.add_edge(source, target, **attrs)


# ============================================================================
# RAG PIPELINE (Main Orchestrator)
# ============================================================================

class RAGPipeline:
    """
    Main RAG pipeline orchestrating all components.
    
    Features:
    - Async query processing
    - Query refinement before retrieval
    - Hybrid retrieval with RRF
    - Cross-encoder re-ranking
    - Response caching
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize components
        self.cache_manager = CacheManager(self.config)
        self.ollama_client = AsyncOllamaClient(self.config, self.cache_manager)
        self.embedding_manager = EmbeddingManager(self.config, self.cache_manager)
        self.reranker = CrossEncoderReranker(self.config)
        self.retrieval_engine = RetrievalEngine(
            self.config, 
            self.embedding_manager, 
            self.reranker
        )
        self.ingestion_engine = IngestionEngine(self.config, self.embedding_manager)
        self.query_refiner = QueryRefiner(self.ollama_client)
        self.multi_query_retriever = MultiQueryRetriever(
            self.ollama_client,
            self.retrieval_engine,
            self.config
        )
        self.knowledge_graph = KnowledgeGraph(self.config.knowledge_graph_path)
        self.file_registry = FileRegistry(self.config.file_registry_path)
        
        self._is_loaded = False
    
    async def index_folder(self, folder_path: str, force_rebuild: bool = False) -> None:
        """
        Index documents in a folder with incremental support.
        
        Args:
            folder_path: Path to folder containing documents
            force_rebuild: If True, rebuild entire index from scratch
        """
        logger.info(f"{'=' * 70}")
        logger.info(f"RAG v18 - Indexing: {folder_path}")
        logger.info(f"{'=' * 70}")
        
        total_start = time.perf_counter()
        
        # Load embedding model
        self.embedding_manager.load_model()
        
        # Find all files
        text_extensions = ["*.txt", "*.md", "*.py", "*.json", "*.csv", "*.pdf", "*.log", "*.docx", "*.xlsx"]
        all_files = []
        for pattern in text_extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))
        
        if not all_files:
            raise RuntimeError(f"No files found in {folder_path}")
        
        # Check for existing index
        has_existing_index = (
            os.path.exists(self.config.index_path) and 
            os.path.exists(self.config.metadata_path) and
            not force_rebuild
        )
        
        if has_existing_index:
            # Incremental indexing
            logger.info("Checking for file changes...")
            new_or_modified, deleted = self.file_registry.get_files_to_reindex(all_files)
            
            if not new_or_modified and not deleted:
                logger.info("No changes detected. Index is up to date.")
                self._is_loaded = True
                return
            
            logger.info(f"Files to process: {len(new_or_modified)} new/modified, {len(deleted)} deleted")
            
            # Load existing index
            await self.load_index()
            
            if new_or_modified:
                # Process only changed files
                logger.start_step("Processing changed files")
                new_chunks, new_metadata = await self.ingestion_engine.ingest_files(
                    new_or_modified, 
                    self.knowledge_graph
                )
                logger.end_step("Processing changed files")
                
                if new_chunks:
                    # Generate embeddings for new chunks
                    logger.start_step("Generating embeddings")
                    new_embeddings = self.embedding_manager.embed_texts(new_chunks)
                    logger.end_step("Generating embeddings")
                    
                    # Add to existing index
                    logger.start_step("Updating index")
                    faiss.normalize_L2(new_embeddings)
                    chunk_start_idx = len(self.retrieval_engine.metadata)
                    self.retrieval_engine.faiss_index.add(new_embeddings.astype(np.float32))
                    self.retrieval_engine.metadata.extend(new_metadata)
                    
                    # Register files
                    file_chunks = {}
                    for i, meta in enumerate(new_metadata):
                        path = meta.get('path', '')
                        if path not in file_chunks:
                            file_chunks[path] = {'start': chunk_start_idx + i, 'count': 0}
                        file_chunks[path]['count'] += 1
                    
                    for path, info in file_chunks.items():
                        self.file_registry.register_file(path, info['start'], info['count'])
                    
                    # Rebuild BM25 with all texts
                    all_texts = [doc['text'] for doc in self.retrieval_engine.metadata]
                    self.retrieval_engine.stemmer = Stemmer("english")
                    corpus_tokens = bm25s.tokenize(all_texts, stemmer=self.retrieval_engine.stemmer)
                    self.retrieval_engine.bm25 = bm25s.BM25()
                    self.retrieval_engine.bm25.index(corpus_tokens)
                    logger.end_step("Updating index")
            
            # Handle deleted files (mark in registry, keep in index for now)
            for deleted_path in deleted:
                self.file_registry.remove_file(deleted_path)
            
        else:
            # Full rebuild
            if force_rebuild:
                logger.info("Force rebuild requested. Clearing existing index...")
                self.file_registry.clear()
            
            logger.start_step("Document ingestion")
            chunks, metadata = await self.ingestion_engine.ingest_files(
                all_files, 
                self.knowledge_graph
            )
            logger.end_step("Document ingestion")
            
            # Build indices
            logger.start_step("Index building")
            self.retrieval_engine.build_index(chunks, metadata)
            logger.end_step("Index building")
            
            # Register all files
            file_chunks = {}
            for i, meta in enumerate(metadata):
                path = meta.get('path', '')
                if path not in file_chunks:
                    file_chunks[path] = {'start': i, 'count': 0}
                file_chunks[path]['count'] += 1
            
            for path, info in file_chunks.items():
                self.file_registry.register_file(path, info['start'], info['count'])
        
        # Save everything
        logger.start_step("Saving indices")
        self.retrieval_engine.save(self.config.index_path, self.config.metadata_path)
        self.knowledge_graph.save()
        self.file_registry.save()
        logger.end_step("Saving indices")
        
        total_time = time.perf_counter() - total_start
        
        logger.info(f"{'=' * 70}")
        logger.info(f"Indexing complete in {total_time:.2f}s")
        logger.info(f"   Total chunks: {len(self.retrieval_engine.metadata)}")
        logger.info(f"   Indexed files: {len(self.file_registry.files)}")
        logger.info(f"   Index: {self.config.index_path}")
        logger.info(f"{'=' * 70}")
        
        self._is_loaded = True
    
    async def load_index(self) -> None:
        """Load existing index from disk."""
        if self._is_loaded:
            return
        
        logger.info("Loading existing index...")
        
        # Load embedding model
        self.embedding_manager.load_model()
        
        # Load indices
        self.retrieval_engine.load(self.config.index_path, self.config.metadata_path)
        
        # Load knowledge graph
        if os.path.exists(self.config.knowledge_graph_path):
            self.knowledge_graph.load()
        
        # Load reranker
        self.reranker.load_model()
        
        self._is_loaded = True
        logger.info("Index loaded successfully")
    
    def _build_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Build the prompt for the LLM."""
        separator = "\n\n---\n\n"
        context = separator.join(context_chunks)
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    async def query(
        self, 
        question: str, 
        refine_query: bool = True,
        top_k: Optional[int] = None,
        use_multi_query: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline.
        
        Steps:
        1. Query refinement (optional)
        2. Multi-query decomposition (if complex question)
        3. Parallel hybrid retrieval (FAISS + BM25 + RRF)
        4. Result deduplication and cross-query re-ranking
        5. LLM generation
        
        Returns:
            Dict with 'answer', 'sources', 'latency', 'sub_queries' keys
        """
        if not self._is_loaded:
            await self.load_index()
        
        total_start = time.perf_counter()
        latency = {}
        
        # Step 1: Query refinement
        if refine_query:
            logger.start_step("Query refinement")
            refined_query = await self.query_refiner.refine(question)
            latency['refinement'] = logger.end_step("Query refinement")
        else:
            refined_query = question
            latency['refinement'] = 0.0
        
        # Step 2 & 3: Multi-query retrieval (automatically uses standard if simple)
        logger.start_step("Retrieval")
        top_k = top_k or self.config.top_k_final
        
        # Temporarily disable multi-query if requested
        original_enable = self.config.enable_multi_query
        if use_multi_query is False:
            self.config.enable_multi_query = False
        
        try:
            results = await self.multi_query_retriever.retrieve_multi_query(
                refined_query,
                top_k_retrieval=self.config.top_k_retrieval,
                top_k_final=top_k
            )
        finally:
            # Restore original setting
            self.config.enable_multi_query = original_enable
        
        latency['retrieval'] = logger.end_step("Retrieval")
        
        if not results:
            return {
                'answer': "No relevant documents found.",
                'sources': [],
                'latency': latency,
                'refined_query': refined_query,
                'sub_queries': []
            }
        
        # Extract context and sources
        context_chunks = [text for text, _, _ in results]
        sources = [
            {
                'path': meta.get('path', 'unknown'),
                'score': float(score),
                'preview': text[:200] + '...' if len(text) > 200 else text
            }
            for text, meta, score in results
        ]
        
        # Step 4: LLM generation
        logger.start_step("LLM generation")
        prompt = self._build_prompt(question, context_chunks)
        
        try:
            answer = await self.ollama_client.chat(
                messages=[{"role": "user", "content": prompt}],
                use_cache=True
            )
        except (TimeoutError, ConnectionError) as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating response: {e}"
        
        latency['generation'] = logger.end_step("LLM generation")
        
        # Total latency
        latency['total'] = time.perf_counter() - total_start
        
        # Get sub-queries if multi-query was used
        sub_queries = []
        if self.multi_query_retriever._should_use_multi_query(refined_query):
            sub_queries = await self.multi_query_retriever.decompose_query(refined_query)
            if len(sub_queries) == 1:
                sub_queries = []  # Single query, not really multi-query
        
        logger.info(f"Query completed in {latency['total']:.3f}s")
        
        return {
            'answer': answer,
            'sources': sources,
            'latency': latency,
            'refined_query': refined_query,
            'sub_queries': sub_queries
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.ollama_client.close()
        self.cache_manager.close()


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Track and report performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "queries": [],
            "indexing": {}
        }
        self.process = psutil.Process()
    
    def record_query(self, question: str, latency: Dict[str, float], success: bool) -> None:
        """Record query metrics."""
        self.metrics["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "question": question[:100],
            "latency": latency,
            "success": success,
            "memory_mb": self.process.memory_info().rss / (1024 * 1024)
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        if not self.metrics["queries"]:
            return {"message": "No queries recorded"}
        
        queries = self.metrics["queries"]
        total_latencies = [q["latency"]["total"] for q in queries]
        
        return {
            "total_queries": len(queries),
            "avg_latency": sum(total_latencies) / len(total_latencies),
            "min_latency": min(total_latencies),
            "max_latency": max(total_latencies),
            "success_rate": sum(1 for q in queries if q["success"]) / len(queries) * 100
        }
    
    def save(self, path: str) -> None:
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# ============================================================================
# FILE WATCHER (Auto-indexing)
# ============================================================================

class AutoIndexHandler(FileSystemEventHandler):
    """Watches folder for changes and triggers re-indexing."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.py', '.json', '.csv', '.pdf', '.log', '.docx', '.xlsx'}
    
    def __init__(self, folder_path: str, debounce_seconds: float = 5.0):
        super().__init__()
        self.folder_path = folder_path
        self.debounce_seconds = debounce_seconds
        self._pending_reindex = False
        self._last_event_time = 0.0
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
    
    def _is_supported_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def on_created(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            logger.info(f"File created: {os.path.basename(event.src_path)}")
            self._schedule_reindex()
    
    def on_modified(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            logger.info(f"File modified: {os.path.basename(event.src_path)}")
            self._schedule_reindex()
    
    def on_deleted(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            logger.info(f"File deleted: {os.path.basename(event.src_path)}")
            self._schedule_reindex()
    
    def _schedule_reindex(self):
        self._last_event_time = time.time()
        self._pending_reindex = True
    
    def should_reindex(self) -> bool:
        """Check if enough time has passed since last event (debounce)."""
        if not self._pending_reindex:
            return False
        if time.time() - self._last_event_time >= self.debounce_seconds:
            self._pending_reindex = False
            return True
        return False


async def watch_folder(folder_path: str, pipeline: RAGPipeline) -> None:
    """Watch folder for changes and auto-reindex."""
    
    # Initial index
    logger.info(f"Initial indexing of: {folder_path}")
    await pipeline.index_folder(folder_path)
    
    # Setup watcher
    handler = AutoIndexHandler(folder_path, debounce_seconds=5.0)
    observer = Observer()
    observer.schedule(handler, folder_path, recursive=True)
    observer.start()
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Watching for changes: {folder_path}")
    logger.info(f"Press Ctrl+C to stop")
    logger.info(f"{'=' * 70}\n")
    
    try:
        while True:
            await asyncio.sleep(1)
            
            if handler.should_reindex():
                logger.info("Changes detected, re-indexing...")
                try:
                    await pipeline.index_folder(folder_path)
                except Exception as e:
                    logger.error(f"Re-indexing failed: {e}")
    
    except KeyboardInterrupt:
        logger.info("\nStopping file watcher...")
    finally:
        observer.stop()
        observer.join()


async def run_benchmark(pipeline: RAGPipeline, queries: List[str]) -> None:
    """Run benchmark with multiple queries."""
    perf_monitor = PerformanceMonitor()
    
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: Running {len(queries)} queries")
    print(f"{'=' * 70}\n")
    
    await pipeline.load_index()
    
    total_start = time.perf_counter()
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query}")
        print("-" * 50)
        
        try:
            result = await pipeline.query(query, refine_query=True)
            perf_monitor.record_query(query, result['latency'], True)
            
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Total latency: {result['latency']['total']:.3f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            perf_monitor.record_query(query, {"total": 0}, False)
    
    total_time = time.perf_counter() - total_start
    
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    
    stats = perf_monitor.get_stats()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average latency: {stats['avg_latency']:.3f}s")
    print(f"Min latency: {stats['min_latency']:.3f}s")
    print(f"Max latency: {stats['max_latency']:.3f}s")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"{'=' * 70}\n")


async def interactive_mode(pipeline: RAGPipeline) -> None:
    """Run interactive query mode."""
    print(f"\n{'=' * 70}")
    print(f"Interactive Query Mode")
    print(f"Type 'quit' or 'exit' to stop")
    print(f"{'=' * 70}\n")
    
    await pipeline.load_index()
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            result = await pipeline.query(query, refine_query=True)
            
            print(f"\n{'-' * 50}")
            print("ANSWER:")
            print(f"{'-' * 50}")
            print(result['answer'])
            
            print(f"\n{'-' * 50}")
            print(f"Latency: {result['latency']['total']:.3f}s | Sources: {len(result['sources'])}")
            print(f"{'-' * 50}")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG System v18.0 - Multi-Query Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--index", action="store_true", help="Create/rebuild index (incremental)")
    parser.add_argument("--folder", type=str, default=".", help="Folder to index")
    parser.add_argument("--force-rebuild", action="store_true", help="Force full index rebuild (ignore cache)")
    parser.add_argument("--query", type=str, help="Query the indexed documents")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark with sample queries")
    parser.add_argument("--no-refine", action="store_true", help="Skip query refinement")
    parser.add_argument("--no-multi-query", action="store_true", help="Disable multi-query decomposition")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all caches")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--warmup", action="store_true", help="Pre-load LLM model into memory")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch folder and auto-reindex on changes")
    parser.add_argument("--index-dir", type=str, help="Custom index directory (default: index_v18)")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory (default: cache_v18)")
    
    args = parser.parse_args()
    
    # Initialize config and pipeline
    config = Config()
    
    # Override directories if specified via command line
    if args.index_dir:
        config.index_dir = args.index_dir
        # Recalculate paths based on new index_dir
        config.index_path = os.path.join(config.index_dir, "faiss_index.bin")
        config.metadata_path = os.path.join(config.index_dir, "metadata.json")
        config.file_registry_path = os.path.join(config.index_dir, "file_registry.json")
    
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    
    # Track directories that might be created during runtime for cleanup
    directories_created = set()
    
    print(f"\n{'=' * 70}")
    print(f"RAG System v18.0 - Multi-Query Retrieval")
    print(f"{'=' * 70}")
    print(f"Device: {config.device.upper()}")
    print(f"GPU Available: {config.use_gpu}")
    print(f"Max Workers: {config.max_workers}")
    print(f"Cache Enabled: Embeddings={config.embedding_cache_enabled}, LLM={config.llm_cache_enabled}")
    print(f"{'=' * 70}\n")
    
    pipeline = RAGPipeline(config)
    perf_monitor = PerformanceMonitor()
    
    try:
        if args.clear_cache:
            print("Clearing caches...")
            pipeline.cache_manager.clear()
            print("Caches cleared.")
            return
        
        if args.cache_stats:
            stats = pipeline.cache_manager.get_stats()
            print("Cache Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        if args.warmup:
            print("Warming up LLM model (this may take 1-2 minutes on first run)...")
            try:
                response = await pipeline.ollama_client.generate("Say 'ready' in one word.", use_cache=False)
                print(f"Model ready! Response: {response.strip()}")
            except Exception as e:
                print(f"Warmup failed: {e}")
            return
        
        if args.index:
            # Create index directory if it doesn't exist
            if not os.path.exists(config.index_dir):
                os.makedirs(config.index_dir, exist_ok=True)
                directories_created.add(config.index_dir)
            await pipeline.index_folder(args.folder, force_rebuild=args.force_rebuild)
        
        if args.query:
            # Check if index exists before querying
            if not os.path.exists(config.index_path):
                print(f"\n{'=' * 70}")
                print("ERROR: Index not found")
                print(f"{'=' * 70}")
                print(f"Index file not found: {config.index_path}")
                print(f"\nPlease create an index first using:")
                print(f"  python {os.path.basename(sys.argv[0])} --index --folder <folder_path>")
                print(f"{'=' * 70}\n")
                return
            result = await pipeline.query(
                args.query,
                refine_query=not args.no_refine,
                top_k=args.k,
                use_multi_query=not args.no_multi_query
            )
            
            perf_monitor.record_query(args.query, result['latency'], True)
            
            print(f"\n{'=' * 70}")
            print("ANSWER")
            print(f"{'=' * 70}\n")
            print(result['answer'])
            
            if result.get('sub_queries'):
                print(f"\n{'=' * 70}")
                print("SUB-QUERIES (Multi-Query Decomposition)")
                print(f"{'=' * 70}")
                for i, sq in enumerate(result['sub_queries'], 1):
                    print(f"  {i}. {sq}")
                print(f"{'=' * 70}\n")
            
            print(f"\n{'=' * 70}")
            print("SOURCES")
            print(f"{'=' * 70}\n")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['path']} (score: {source['score']:.4f})")
            
            print(f"\n{'=' * 70}")
            print("LATENCY")
            print(f"{'=' * 70}")
            for step, time_s in result['latency'].items():
                print(f"  {step}: {time_s:.3f}s")
            print(f"{'=' * 70}\n")
        
        if args.interactive:
            # Check if index exists before interactive mode
            if not os.path.exists(config.index_path):
                print(f"\n{'=' * 70}")
                print("ERROR: Index not found")
                print(f"{'=' * 70}")
                print(f"Index file not found: {config.index_path}")
                print(f"\nPlease create an index first using:")
                print(f"  python {os.path.basename(sys.argv[0])} --index --folder <folder_path>")
                print(f"{'=' * 70}\n")
                return
            await interactive_mode(pipeline)
        
        if args.benchmark:
            # Check if index exists before benchmark
            if not os.path.exists(config.index_path):
                print(f"\n{'=' * 70}")
                print("ERROR: Index not found")
                print(f"{'=' * 70}")
                print(f"Index file not found: {config.index_path}")
                print(f"\nPlease create an index first using:")
                print(f"  python {os.path.basename(sys.argv[0])} --index --folder <folder_path>")
                print(f"{'=' * 70}\n")
                return
            benchmark_queries = [
                "What is machine learning?",
                "How does neural network training work?",
                "Explain the concept of embeddings",
                "What are transformers in AI?",
                "How does RAG improve LLM accuracy?",
            ]
            await run_benchmark(pipeline, benchmark_queries)
        
        if args.watch:
            await watch_folder(args.folder, pipeline)
        
        if not any([args.index, args.query, args.interactive, args.benchmark, 
                    args.clear_cache, args.cache_stats, args.warmup, args.watch]):
            parser.print_help()
    
    finally:
        await pipeline.close()
        
        # Cleanup: Remove empty directories that were created but not used
        for dir_path in directories_created:
            if remove_empty_directory(dir_path):
                print(f"Removed unused empty directory: {dir_path}")
        
        # Also check cache directory - remove if empty (diskcache may have created it)
        # Check if cache_dir exists and is empty (or only contains empty subdirectories)
        if os.path.exists(config.cache_dir) and os.path.isdir(config.cache_dir):
            try:
                # Check if cache_dir is empty or only contains empty subdirectories
                contents = os.listdir(config.cache_dir)
                all_empty = True
                for item in contents:
                    item_path = os.path.join(config.cache_dir, item)
                    if os.path.isdir(item_path):
                        if os.listdir(item_path):  # Subdirectory has contents
                            all_empty = False
                            break
                    else:  # File exists
                        all_empty = False
                        break
                
                if all_empty:
                    # Remove empty subdirectories first
                    for item in contents:
                        item_path = os.path.join(config.cache_dir, item)
                        if os.path.isdir(item_path):
                            try:
                                os.rmdir(item_path)
                            except OSError:
                                pass
                    # Then remove cache_dir if it's now empty
                    if remove_empty_directory(config.cache_dir):
                        print(f"Removed unused empty cache directory: {config.cache_dir}")
            except OSError:
                pass


def run():
    """Synchronous wrapper for main()."""
    asyncio.run(main())


if __name__ == "__main__":
    run()

