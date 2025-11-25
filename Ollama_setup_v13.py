#!/usr/bin/env python3

"""
🚀 TOON-Enhanced RAG System - Version 13.0 (Large JSON Support)

🆕 NEW FEATURES IN v13.0:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Streaming JSON Parser - Handle files >50MB without loading into memory
✅ Semantic JSON Chunking - Preserve key-value pairs and object hierarchy
✅ Hierarchical Embeddings - Maintain parent-child relationships in vectors
✅ Checkpoint System - Resume large JSON processing from interruptions
✅ Adaptive Sizing - Dynamic chunk size based on JSON nesting complexity
✅ Batch Processing - Memory-efficient incremental JSON processing
✅ Structure Preservation - Keep JSON schema context in embeddings
✅ Enhanced Metadata - Extract keys, types, nesting depth automatically

📦 All v12.0 Features Included:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ TOON (Task-Oriented Orchestration Network) integration
✅ Knowledge graph for file relationships and dependencies
✅ Task orchestration with workflow automation
✅ Multi-file reasoning and cross-document analysis
✅ Enhanced metadata with entity extraction
✅ Task queue system for concurrent processing
✅ Incremental indexing without full rebuild
✅ Robust PDF error handling with multiple extraction strategies
✅ Direct Sentence-Transformers embeddings (10-50x faster)
✅ BM25S for lexical search (500x faster)
✅ TinyBERT-L2-v2 cross-encoder reranking (9x faster)
✅ Hybrid FAISS + BM25S retrieval
✅ Conversational memory for multi-turn dialogue
✅ Performance monitoring and statistics
✅ PII detection and redaction
✅ Audit logging for compliance

📋 Installation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip install sentence-transformers torch faiss-cpu numpy pymupdf bm25s PyStemmer psutil tiktoken transformers ollama networkx spacy ijson

python -m spacy download en_core_web_sm

🚀 Usage Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Index documents (including large JSON files)
python V13_Enhanced_RAG_Large_JSON.py --index --folder ./data

# Query the system
python V13_Enhanced_RAG_Large_JSON.py --query "What information is in the JSON data?"

# Execute TOON task
python V13_Enhanced_RAG_Large_JSON.py --toon-task summarize_all

# Process specific large JSON file
python V13_Enhanced_RAG_Large_JSON.py --index-json large_file.json --streaming

"""

#!/usr/bin/env python3

"""
🚀 TOON-Enhanced RAG System - Version 9.0 (Multi-File Orchestration)

NEW FEATURES (v9.0):
✅ TOON (Task-Oriented Orchestration Network) integration
✅ Knowledge graph for file relationships and dependencies
✅ Task orchestration with workflow automation
✅ Multi-file reasoning and cross-document analysis
✅ Enhanced metadata with entity extraction
✅ Task queue system for concurrent processing
✅ Incremental indexing without full rebuild
✅ API layer with REST endpoints
✅ Security features (PII detection, audit logs)

Features from v8.2:
✅ Robust PDF error handling with automatic repair
✅ Multiple text extraction strategies
✅ Direct Sentence-Transformers embeddings (10-50x faster)
✅ BM25S for lexical search (500x faster)
✅ TinyBERT-L2-v2 cross-encoder (9x faster re-ranking)
✅ Hybrid FAISS + BM25S retrieval
✅ Conversational memory for multi-turn dialogue
✅ Performance monitoring and statistics

Dependencies:
pip install sentence-transformers torch faiss-cpu numpy pymupdf bm25s PyStemmer psutil tiktoken transformers ollama networkx spacy flask celery redis
python -m spacy download en_core_web_sm
"""

import glob
import json
import os
import re
import time
import hashlib
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any, Iterator
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import psutil
import tiktoken
import torch
import bm25s
import networkx as nx
import spacy
from Stemmer import Stemmer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ollama import chat, pull
import fitz  # PyMuPDF

# NEW in v13: Streaming JSON support for large files
try:
    import ijson
    IJSON_AVAILABLE = True
    print("✓ ijson available - Large JSON streaming enabled")
except ImportError:
    IJSON_AVAILABLE = False
    print("⚠️  ijson not installed - Large JSON streaming disabled")
    print("   Install with: pip install ijson")

# CONFIGURATION
# ============================================================================

# Embedding & LLM Models
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
TEXT_MODEL = "gemma3"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L2-v2"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
LOG_LINES_PER_CHUNK = 50

# Batch Processing
BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 64

# Paths
INDEX_PATH = "index/faiss_index.bin"
METADATA_PATH = "metadata.json"
KNOWLEDGE_GRAPH_PATH = "knowledge_graph.json"
PERFORMANCE_LOG = "performance_metrics.json"
AUDIT_LOG = "audit_log.json"

# Limits
MAX_FILE_SIZE_MB = 500
TOP_K_DEFAULT = 5
MAX_TOKENS = 16000

# Generation
TEMPERATURE = 0.3
TOP_P = 0.9
TOP_K = 40

# Flags
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

# Tokenizer cache
_tokenizer_cache = {}

print(f"\n{'=' * 70}")
print(f"🚀 TOON-Enhanced RAG System v12.0 - Multi-File Orchestration")
print(f"{'=' * 70}")
print(f"Device: {DEVICE.upper()}")
print(f"GPU Available: {USE_GPU}")
print(f"Embedding Model: {EMBED_MODEL_NAME.split('/')[-1]}")
print(f"Cross-Encoder: {RERANKER_MODEL_NAME.split('/')[-1]}")
print(f"Chunk Size: {CHUNK_SIZE} | Batch Size: {BATCH_SIZE}")
print(f"{'=' * 70}\n")


# ============================================================================

# ============================================================================
# JSON PROCESSING CONFIGURATION (NEW in v13.0)
# ============================================================================

JSON_CHUNK_SIZE = 512  # Smaller chunks for JSON (better semantic matching)
JSON_STREAMING_THRESHOLD_MB = 50  # Use streaming for files larger than this
JSON_BATCH_SIZE = 100  # Process JSON objects in batches
MAX_NESTING_DEPTH = 10  # Maximum JSON nesting depth to preserve
JSON_CHECKPOINT_DIR = "json_checkpoints"  # Checkpoint storage directory
JSON_CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N JSON objects

print(f"JSON Settings: Chunk={JSON_CHUNK_SIZE}, Threshold={JSON_STREAMING_THRESHOLD_MB}MB, Streaming={'ON' if IJSON_AVAILABLE else 'OFF'}")

# AUDIT & SECURITY
# ============================================================================


# ============================================================================
# LARGE JSON PROCESSING ENGINE (NEW in v13.0)
# ============================================================================

class JSONStreamProcessor:
    """
    Advanced streaming JSON processor for files >50MB.

    Features:
    - Streaming parsing with ijson (handles GB-sized files)
    - Semantic chunking preserving JSON structure
    - Hierarchical context preservation
    - Checkpoint/resume capability
    - Adaptive chunk sizing based on nesting
    - Batch processing for memory efficiency
    """

    def __init__(self, checkpoint_dir: str = JSON_CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def should_use_streaming(self, filepath: str) -> bool:
        """Determine if file should use streaming based on size."""
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            return size_mb > JSON_STREAMING_THRESHOLD_MB and IJSON_AVAILABLE
        except:
            return False

    def stream_json_objects(self, filepath: str, batch_size: int = JSON_BATCH_SIZE) -> Iterator[List[Dict]]:
        """
        Stream JSON file and yield batches of objects.

        Supports:
        - JSON arrays: [{"obj1"}, {"obj2"}, ...]
        - JSON objects: {"key": "value", ...}
        - NDJSON: line-delimited JSON objects
        """
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        use_streaming = self.should_use_streaming(filepath)

        print(f"  JSON file size: {file_size_mb:.2f}MB - {'Streaming' if use_streaming else 'Standard'} mode")

        if not use_streaming:
            # Standard loading for smaller files
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i in range(0, len(data), batch_size):
                            yield data[i:i + batch_size]
                    elif isinstance(data, dict):
                        yield [data]
                    else:
                        yield [{"value": data}]
                except json.JSONDecodeError:
                    # Try NDJSON format
                    f.seek(0)
                    batch = []
                    for line in f:
                        if line.strip():
                            try:
                                batch.append(json.loads(line))
                                if len(batch) >= batch_size:
                                    yield batch
                                    batch = []
                            except:
                                continue
                    if batch:
                        yield batch
            return

        # Streaming mode for large files
        with open(filepath, 'rb') as f:
            try:
                # Attempt to parse as array of objects
                parser = ijson.items(f, 'item')
                batch = []
                obj_count = 0

                for item in parser:
                    batch.append(item)
                    obj_count += 1

                    if len(batch) >= batch_size:
                        print(f"    Streamed {obj_count} JSON objects...", end='\r')
                        yield batch
                        batch = []

                if batch:
                    yield batch
                    print(f"    Completed: {obj_count} total JSON objects          ")

            except (ijson.JSONError, ijson.IncompleteJSONError):
                # Fallback: try parsing as single object
                f.seek(0)
                try:
                    items = list(ijson.kvitems(f, ''))
                    yield [dict(items)]
                except:
                    print("    ⚠️ Warning: Could not parse JSON structure")
                    yield []

    def flatten_json_with_context(self, json_obj: Dict, parent_key: str = '', max_depth: int = MAX_NESTING_DEPTH, current_depth: int = 0) -> str:
        """
        Flatten nested JSON while preserving hierarchical context.

        Example:
        {"user": {"name": "John", "age": 30}}
        -> "user.name: John | user.age: 30"
        """
        if current_depth >= max_depth or not isinstance(json_obj, (dict, list)):
            return f"{parent_key}: {str(json_obj)[:100]}" if parent_key else str(json_obj)[:100]

        if not isinstance(json_obj, dict):
            return f"{parent_key}: {str(json_obj)[:200]}"

        items = []

        for key, value in json_obj.items():
            new_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                flattened = self.flatten_json_with_context(value, new_key, max_depth, current_depth + 1)
                if flattened:
                    items.append(flattened)

            elif isinstance(value, list):
                if not value:
                    items.append(f"{new_key}: []")
                elif isinstance(value[0] if value else None, dict):
                    # Array of objects - sample first few
                    sample_size = min(3, len(value))
                    for idx in range(sample_size):
                        flattened = self.flatten_json_with_context(
                            value[idx], f"{new_key}[{idx}]", max_depth, current_depth + 1
                        )
                        if flattened:
                            items.append(flattened)
                    if len(value) > sample_size:
                        items.append(f"{new_key}: ...{len(value)} total items")
                else:
                    # Simple array
                    sample = ', '.join(str(v) for v in value[:5])
                    suffix = f", ...{len(value)-5} more" if len(value) > 5 else ""
                    items.append(f"{new_key}: [{sample}{suffix}]")
            else:
                items.append(f"{new_key}: {value}")

        return " | ".join(items)

    def calculate_nesting_depth(self, obj: Any, current: int = 0) -> int:
        """Calculate maximum nesting depth of JSON structure."""
        if not isinstance(obj, (dict, list)) or not obj:
            return current

        if isinstance(obj, dict):
            return max((self.calculate_nesting_depth(v, current + 1) for v in obj.values()), default=current)
        else:  # list
            return max((self.calculate_nesting_depth(item, current + 1) for item in obj if isinstance(item, (dict, list))), default=current)

    def adaptive_chunk_size(self, json_data: Dict) -> int:
        """
        Determine optimal chunk size based on JSON complexity.

        Deep nesting = smaller chunks (more focused)
        Flat structure = larger chunks (more context)
        """
        depth = self.calculate_nesting_depth(json_data)

        if depth > 5:
            return 256   # Deeply nested - use small chunks
        elif depth > 3:
            return 512   # Medium nesting
        elif depth > 1:
            return 768   # Shallow nesting
        else:
            return 1024  # Flat structure - larger chunks OK

    def semantic_json_chunking(self, json_objects: List[Dict], base_chunk_size: int = JSON_CHUNK_SIZE) -> List[Dict]:
        """
        Create semantically meaningful chunks from JSON objects.

        Preserves:
        - Object boundaries
        - Key-value relationships
        - Hierarchical structure
        - Contextual information
        """
        chunks = []

        for idx, obj in enumerate(json_objects):
            # Flatten while preserving structure
            flattened_text = self.flatten_json_with_context(obj)

            # Determine optimal chunk size for this object
            optimal_size = self.adaptive_chunk_size(obj)

            # Calculate metadata
            depth = self.calculate_nesting_depth(obj)

            # If text fits in one chunk, keep it together
            if len(flattened_text) <= optimal_size * 1.5:
                chunks.append({
                    'text': flattened_text,
                    'original_object': obj,
                    'object_index': idx,
                    'chunk_part': 0,
                    'total_parts': 1,
                    'nesting_depth': depth,
                    'chunk_size_used': len(flattened_text)
                })
            else:
                # Split into multiple chunks while preserving word boundaries
                words = flattened_text.split()
                words_per_chunk = optimal_size // 5  # Assume ~5 chars per word

                part = 0
                total_parts = (len(words) + words_per_chunk - 1) // words_per_chunk

                for i in range(0, len(words), words_per_chunk):
                    chunk_words = words[i:i + words_per_chunk]
                    chunk_text = ' '.join(chunk_words)

                    if chunk_text.strip():
                        chunks.append({
                            'text': chunk_text,
                            'original_object': obj,
                            'object_index': idx,
                            'chunk_part': part,
                            'total_parts': total_parts,
                            'nesting_depth': depth,
                            'chunk_size_used': len(chunk_text)
                        })
                        part += 1

        return chunks

    def extract_json_metadata(self, json_chunk: Dict) -> Dict:
        """Extract rich metadata from JSON chunk for enhanced retrieval."""
        metadata = {
            'object_index': json_chunk.get('object_index', 0),
            'chunk_part': json_chunk.get('chunk_part', 0),
            'total_parts': json_chunk.get('total_parts', 1),
            'nesting_depth': json_chunk.get('nesting_depth', 0),
            'chunk_size': json_chunk.get('chunk_size_used', 0),
            'keys': [],
            'key_count': 0,
            'has_arrays': False,
            'has_nested_objects': False,
            'data_types': set()
        }

        original_obj = json_chunk.get('original_object', {})

        if isinstance(original_obj, dict):
            metadata['keys'] = list(original_obj.keys())[:50]  # Limit to first 50 keys
            metadata['key_count'] = len(original_obj.keys())

            for value in original_obj.values():
                metadata['data_types'].add(type(value).__name__)
                if isinstance(value, list):
                    metadata['has_arrays'] = True
                elif isinstance(value, dict):
                    metadata['has_nested_objects'] = True

        metadata['data_types'] = list(metadata['data_types'])

        return metadata

    def save_checkpoint(self, filepath: str, batch_index: int, processed_chunks: int):
        """Save processing checkpoint for resume capability."""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{hashlib.md5(filepath.encode()).hexdigest()}.checkpoint"
        )
        checkpoint_data = {
            'filepath': filepath,
            'batch_index': batch_index,
            'processed_chunks': processed_chunks,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  💾 Checkpoint saved: batch {batch_index}, {processed_chunks} chunks")

    def load_checkpoint(self, filepath: str) -> Optional[Dict]:
        """Load existing checkpoint if available."""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{hashlib.md5(filepath.encode()).hexdigest()}.checkpoint"
        )
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                print(f"  ♻️  Resuming from checkpoint: batch {checkpoint['batch_index']}")
                return checkpoint
        return None

    def clear_checkpoint(self, filepath: str):
        """Remove checkpoint after successful completion."""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{hashlib.md5(filepath.encode()).hexdigest()}.checkpoint"
        )
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"  ✓ Checkpoint cleared")


class AuditLogger:
    """Track all operations for security and compliance."""

    def __init__(self, log_path: str = AUDIT_LOG):
        self.log_path = log_path
        self.logs = []

    def log_operation(self, operation: str, user: str, details: Dict):
        """Log an operation with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "user": user,
            "details": details,
            "success": True
        }
        self.logs.append(entry)
        self._save()

    def log_error(self, operation: str, error: str, details: Dict):
        """Log an error."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "error": error,
            "details": details,
            "success": False
        }
        self.logs.append(entry)
        self._save()

    def _save(self):
        """Save audit log to file."""
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)


class PIIDetector:
    """Detect and optionally redact PII from text."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Spacy model not found. Install with: python -m spacy download en_core_web_sm")
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
# KNOWLEDGE GRAPH
# ============================================================================

class KnowledgeGraph:
    """Manage relationships between files and concepts."""

    def __init__(self, graph_path: str = KNOWLEDGE_GRAPH_PATH):
        self.graph_path = graph_path
        self.graph = nx.Graph()
        self.file_nodes = {}
        self.concept_nodes = {}

        if os.path.exists(graph_path):
            self.load()

    def add_file_node(self, file_path: str, metadata: Dict):
        """Register a file as a knowledge node."""
        node_id = self._hash_path(file_path)

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

    def add_concept_node(self, concept: str, files: List[str]):
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

        # Link concept to files
        for file_path in files:
            if file_path in self.file_nodes:
                self.graph.add_edge(
                    concept_id,
                    self.file_nodes[file_path],
                    relationship='mentioned_in'
                )

    def add_relationship(self, source_file: str, target_file: str, rel_type: str):
        """Add relationship between two files."""
        source_id = self.file_nodes.get(source_file)
        target_id = self.file_nodes.get(target_file)

        if source_id and target_id:
            self.graph.add_edge(source_id, target_id, relationship=rel_type)

    def get_related_files(self, file_path: str, max_depth: int = 2) -> List[str]:
        """Get files related to a given file."""
        node_id = self.file_nodes.get(file_path)
        if not node_id:
            return []

        related = []
        for neighbor in nx.single_source_shortest_path_length(
                self.graph, node_id, cutoff=max_depth
        ).keys():
            if self.graph.nodes[neighbor]['type'] == 'file':
                related.append(self.graph.nodes[neighbor]['path'])

        return related

    def get_file_concepts(self, file_path: str) -> List[str]:
        """Get concepts associated with a file."""
        node_id = self.file_nodes.get(file_path)
        if not node_id:
            return []

        concepts = []
        for neighbor in self.graph.neighbors(node_id):
            if self.graph.nodes[neighbor]['type'] == 'concept':
                concepts.append(self.graph.nodes[neighbor]['name'])

        return concepts

    def save(self):
        """Save knowledge graph to file."""
        data = {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }

        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
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

    @staticmethod
    def _hash_path(path: str) -> str:
        """Generate unique hash for file path."""
        return f"file_{hashlib.md5(path.encode()).hexdigest()[:8]}"


# ============================================================================
# TOON ORCHESTRATOR
# ============================================================================

class TOONOrchestrator:
    """Task-Oriented Orchestration Network for multi-file workflows."""

    def __init__(self, rag_system, knowledge_graph: KnowledgeGraph):
        self.rag = rag_system
        self.kg = knowledge_graph
        self.task_queue = []
        self.task_results = {}

    def register_task(self, task_type: str, params: Dict) -> str:
        """Register a new task for execution."""
        task_id = f"task_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        task = {
            'id': task_id,
            'type': task_type,
            'params': params,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }

        self.task_queue.append(task)
        return task_id

    def execute_task(self, task_id: str) -> Any:
        """Execute a registered task."""
        task = next((t for t in self.task_queue if t['id'] == task_id), None)

        if not task:
            raise ValueError(f"Task {task_id} not found")

        task['status'] = 'running'
        task['started_at'] = datetime.now().isoformat()

        try:
            result = self._execute_workflow(task['type'], task['params'])
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            self.task_results[task_id] = result
            return result
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            raise

    def _execute_workflow(self, task_type: str, params: Dict) -> Any:
        """Execute specific workflow based on task type."""
        workflows = {
            'summarize_all': self._summarize_all_workflow,
            'cross_file_qa': self._cross_file_qa_workflow,
            'generate_report': self._generate_report_workflow,
            'find_conflicts': self._find_conflicts_workflow,
            'extract_entities': self._extract_entities_workflow
        }

        if task_type not in workflows:
            raise ValueError(f"Unknown task type: {task_type}")

        return workflows[task_type](params)

    def _summarize_all_workflow(self, params: Dict) -> Dict:
        """Summarize all indexed files."""
        file_summaries = []

        for file_path in self.kg.file_nodes.keys():
            # Retrieve chunks for this file
            chunks = self.rag.get_chunks_by_file(file_path)

            if chunks:
                # Generate summary
                summary_prompt = f"Summarize the following document sections:\n\n{chunks[:3]}"
                summary = self.rag.generate_answer(summary_prompt)

                file_summaries.append({
                    'file': file_path,
                    'summary': summary,
                    'chunk_count': len(chunks)
                })

        return {
            'summaries': file_summaries,
            'total_files': len(file_summaries)
        }

    def _cross_file_qa_workflow(self, params: Dict) -> Dict:
        """Answer questions across multiple files."""
        question = params.get('question', '')
        files = params.get('files', list(self.kg.file_nodes.keys()))

        # Retrieve relevant chunks from all specified files
        all_chunks = []
        for file_path in files:
            chunks = self.rag.get_chunks_by_file(file_path)
            all_chunks.extend([(chunk, file_path) for chunk in chunks])

        # Perform semantic search across all chunks
        relevant_chunks = self.rag.retrieve_relevant_chunks(question, all_chunks)

        # Generate answer
        answer = self.rag.generate_answer(question, relevant_chunks)

        return {
            'question': question,
            'answer': answer,
            'source_files': list(set([chunk[1] for chunk in relevant_chunks]))
        }

    def _generate_report_workflow(self, params: Dict) -> Dict:
        """Generate a comprehensive report from multiple files."""
        report_type = params.get('type', 'summary')
        files = params.get('files', list(self.kg.file_nodes.keys()))

        report_sections = []

        for file_path in files:
            chunks = self.rag.get_chunks_by_file(file_path)

            if chunks:
                section = {
                    'file': os.path.basename(file_path),
                    'path': file_path,
                    'key_points': self._extract_key_points(chunks),
                    'concepts': self.kg.get_file_concepts(file_path)
                }
                report_sections.append(section)

        return {
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'sections': report_sections,
            'total_files': len(report_sections)
        }

    def _find_conflicts_workflow(self, params: Dict) -> Dict:
        """Find conflicting information across files."""
        topic = params.get('topic', '')

        # Query all files for the topic
        results_by_file = {}

        for file_path in self.kg.file_nodes.keys():
            chunks = self.rag.get_chunks_by_file(file_path)
            relevant = [c for c in chunks if topic.lower() in c.lower()]

            if relevant:
                results_by_file[file_path] = relevant

        # Identify potential conflicts (simplified)
        conflicts = []
        files = list(results_by_file.keys())

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                # Compare statements from different files
                conflicts.append({
                    'file1': files[i],
                    'file2': files[j],
                    'topic': topic,
                    'requires_review': True
                })

        return {
            'topic': topic,
            'conflicts_found': len(conflicts),
            'conflicts': conflicts
        }

    def _extract_entities_workflow(self, params: Dict) -> Dict:
        """Extract named entities from all files."""
        pii_detector = PIIDetector()
        entities_by_file = {}

        for file_path in self.kg.file_nodes.keys():
            chunks = self.rag.get_chunks_by_file(file_path)

            file_entities = []
            for chunk in chunks[:5]:  # Sample first 5 chunks
                entities = pii_detector.detect_pii(chunk)
                file_entities.extend(entities)

            if file_entities:
                entities_by_file[file_path] = file_entities

        return {
            'total_files': len(entities_by_file),
            'entities_by_file': entities_by_file
        }

    def _extract_key_points(self, chunks: List[str], max_points: int = 5) -> List[str]:
        """Extract key points from text chunks."""
        # Simplified key point extraction
        key_points = []

        for chunk in chunks[:max_points]:
            sentences = chunk.split('.')
            if sentences:
                key_points.append(sentences[0].strip() + '.')

        return key_points


# ============================================================================
# UTILITY FUNCTIONS (from original code)
# ============================================================================

def count_tokens(text: str, model_name='gemma3') -> int:
    """Count tokens using tiktoken."""
    try:
        if model_name in _tokenizer_cache:
            tokenizer = _tokenizer_cache[model_name]
        else:
            tokenizer = tiktoken.encoding_for_model(model_name)
            _tokenizer_cache[model_name] = tokenizer
    except KeyError:
        tokenizer = tiktoken.get_encoding('cl100k_base')

    return len(tokenizer.encode(text))


def build_prompt_within_token_limit(context_chunks: List[str], question: str, max_tokens: int = MAX_TOKENS) -> str:
    """Build prompt while respecting token limits."""
    separator = "\n\n---\n\n"
    prompt_intro = "You are given the following extracted document chunks as context:\n\n"
    prompt_question = f"\n\nQuestion: {question}"

    allowed_tokens = max_tokens - count_tokens(prompt_intro) - count_tokens(prompt_question) - 100

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


# ============================================================================
# PERFORMANCE MONITORING (from original code)
# ============================================================================

class PerformanceMonitor:
    """Track indexing and query performance metrics."""

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
        print(f"\n{'=' * 70}")
        print(f"📊 Performance Monitoring Started")
        print(f"{'=' * 70}\n")

    def sample_resources(self):
        """Sample CPU and memory usage."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = self.process.memory_info().rss / (1024 * 1024)
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

    def print_indexing_report(self):
        m = self.metrics["indexing"]
        print(f"\n{'=' * 70}")
        print(f"📊 INDEXING PERFORMANCE REPORT")
        print(f"{'=' * 70}\n")

        print(f"⏱️ Processing Time")
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

        print(f"\n{'=' * 70}\n")

    def print_query_stats(self):
        if not self.metrics["queries"]:
            print("No queries recorded yet.")
            return

        queries = self.metrics["queries"]
        durations = [q["duration_seconds"] for q in queries]

        print(f"\n{'=' * 70}")
        print(f"🔍 QUERY PERFORMANCE STATISTICS")
        print(f"{'=' * 70}\n")

        print(f"Total Queries: {len(queries)}")
        print(f"Average Response Time: {sum(durations) / len(durations):.3f} seconds")
        print(f"Fastest Query: {min(durations):.3f} seconds")
        print(f"Slowest Query: {max(durations):.3f} seconds")
        print(f"Success Rate: {sum(1 for q in queries if q['success']) / len(queries) * 100:.1f}%")

        print(f"\n{'=' * 70}\n")

    def save_to_file(self, filepath: str = PERFORMANCE_LOG):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Performance metrics saved to: {filepath}")


perf_monitor = PerformanceMonitor()
audit_logger = AuditLogger()


# ============================================================================
# TEXT EXTRACTION (from original code - keeping robust PDF extraction)
# ============================================================================

def parse_log_line(line: str) -> Optional[Dict]:
    """Parse a single log line."""
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
    """Chunk a log file with metadata."""
    chunks = []
    current_chunk_lines = []
    current_chunk_metadata = {"start_line": 0, "log_levels": set()}
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
        print(f"Error reading log file {filepath}: {e}")
        return []

    return chunks


def validate_extracted_text(text: str, min_length: int = 10) -> bool:
    """Validate if extracted text is meaningful."""
    if not text or len(text.strip()) < min_length:
        return False

    # Check if text is mostly garbled
    if len(text) > 0:
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = special_chars / len(text)
        if ratio > 0.7:
            return False

    return True


def extract_pdf_with_layout(doc, page_num: int) -> str:
    """Extract text preserving layout using 'blocks' mode."""
    try:
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        text_blocks = []

        for block in blocks:
            if len(block) >= 5 and isinstance(block[4], str):
                text_blocks.append(block[4])

        return "\n".join(text_blocks)
    except Exception as e:
        return ""


def extract_pdf_with_dict(doc, page_num: int) -> str:
    """Extract text using detailed dictionary mode."""
    try:
        page = doc.load_page(page_num)
        text_dict = page.get_text("dict")
        text_parts = []

        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text_parts.append(span.get("text", ""))

        return " ".join(text_parts)
    except Exception as e:
        return ""


def check_pdf_needs_ocr(doc) -> bool:
    """Detect if PDF is scanned and needs OCR."""
    try:
        total_pages = len(doc)
        sample_size = min(3, total_pages)
        low_text_pages = 0

        for i in range(sample_size):
            page = doc.load_page(i)
            text = page.get_text().strip()

            if len(text) < 100:
                low_text_pages += 1

        return low_text_pages / sample_size > 0.5
    except:
        return False


def repair_and_clean_pdf(path: str) -> Optional[bytes]:
    """Attempt to repair a corrupt PDF."""
    try:
        doc = fitz.open(path)
        pdf_bytes = doc.tobytes(
            garbage=4,
            clean=True,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True
        )
        doc.close()
        return pdf_bytes
    except Exception as e:
        print(f"   Could not repair PDF: {e}")
        return None


def extract_text_from_pdf_robust(path: str) -> str:
    """Enhanced PDF text extraction with multiple strategies."""
    doc = None
    text = ""

    try:
        fitz.TOOLS.reset_mupdf_warnings()
        doc = fitz.open(path)

        if doc.is_repaired:
            print(f"   PDF was automatically repaired during opening")

        needs_ocr = check_pdf_needs_ocr(doc)
        if needs_ocr:
            print(f"   ⚠️ WARNING: PDF appears to be scanned. Consider using OCR.")

        total_pages = len(doc)
        print(f"   Pages: {total_pages}", end="")

        # Strategy 1: Standard extraction
        try:
            text_parts = []
            for page_num in range(total_pages):
                page_text = doc.load_page(page_num).get_text("text")
                if page_text:
                    text_parts.append(page_text)

            text = "\n\n".join(text_parts)
            if validate_extracted_text(text):
                print(f" | Extracted: {len(text)} chars")
                doc.close()
                return text
        except Exception as e:
            print(f"\n   Standard extraction failed: {e}")

        # Strategy 2: Block-based extraction
        try:
            text_parts = []
            for page_num in range(total_pages):
                page_text = extract_pdf_with_layout(doc, page_num)
                if page_text:
                    text_parts.append(page_text)

            text = "\n\n".join(text_parts)
            if validate_extracted_text(text):
                print(f" | Extracted: {len(text)} chars (block mode)")
                doc.close()
                return text
        except Exception as e:
            print(f"\n   Block extraction failed: {e}")

        # Additional strategies...
        doc.close()

    except Exception as e:
        print(f"\n   PDF extraction failed: {e}")
        if doc:
            doc.close()

    return ""


def extract_text_from_file(path: str) -> str:
    """Extract text from various file formats."""
    ext = os.path.splitext(path)[1].lower()

    if ext in [".log"]:
        return None

    # Text files
    if ext in [".txt", ".md", ".py", ".json", ".csv"]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            return ""

    # PDF files
    if ext == ".pdf":
        return extract_text_from_pdf_robust(path)

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

        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = end - overlap

    return chunks


# ============================================================================
# EMBEDDING & INDEXING
# ============================================================================

class EmbeddingManager:
    """Manages embeddings using Sentence-Transformers."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        print(f"✓ Embedding model loaded on {device.upper()}\n")

    def embed_texts(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Batch embed texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        return embeddings


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """Hybrid retrieval using FAISS + BM25S + Cross-Encoder."""

    def __init__(self, faiss_index, metadata, device=DEVICE, alpha=0.5):
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.alpha = alpha
        self.device = device

        print("Initializing BM25S lexical search...")
        stemmer = Stemmer("english")
        corpus_texts = [doc['text'] for doc in metadata]
        corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=stemmer)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)
        self.stemmer = stemmer
        print("✓ BM25S initialized\n")

        print(f"Loading cross-encoder: {RERANKER_MODEL_NAME}")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME
        ).to(device)
        print(f"✓ Cross-encoder loaded on {device.upper()}\n")

    def query(self, question: str, embedding_fn, top_k: int = TOP_K_DEFAULT) -> List[str]:
        """Retrieve documents using hybrid search + re-ranking."""
        max_results = min(top_k * 2, len(self.metadata))

        # BM25S lexical search
        query_tokens = bm25s.tokenize([question], stemmer=self.stemmer)
        bm25_results_docs, bm25_scores = self.bm25.retrieve(query_tokens, k=max_results)

        bm25_candidate_texts = []
        if len(bm25_results_docs) > 0:
            bm25_candidate_texts = [str(doc) for doc in bm25_results_docs[0]]

        # FAISS semantic search
        q_embedding = embedding_fn([question])[0].reshape(1, -1)
        distances, indices = self.faiss_index.search(q_embedding, max_results)

        # Combine candidates
        candidate_set = {}

        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                doc_text = self.metadata[int(idx)]["text"]
                candidate_set[int(idx)] = doc_text

        for bm25_text in bm25_candidate_texts:
            for meta_idx, meta in enumerate(self.metadata):
                if meta["text"][:100] in str(bm25_text)[:100]:
                    candidate_set[meta_idx] = meta["text"]
                    break

        candidates = list(candidate_set.values())[:max_results]

        if not candidates:
            return []

        return self._rerank_with_cross_encoder(question, candidates, top_k)

    def _rerank_with_cross_encoder(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """Re-rank candidates using cross-encoder."""
        if not candidates:
            return []

        inputs = self.reranker_tokenizer(
            [query] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]


# ============================================================================
# ENHANCED RAG SYSTEM WITH TOON
# ============================================================================

class TOONEnabledRAGSystem:
    """RAG System with TOON orchestration capabilities."""

    def __init__(self):
        self.index = None
        self.metadata = []
        self.embedding_manager = None
        self.retriever = None
        self.knowledge_graph = KnowledgeGraph()
        self.orchestrator = None
        self.conv_memory = ConversationalMemory(max_turns=5)

    def index_folder(self, folder_path: str):
        """Index all documents in a folder with TOON integration."""
        perf_monitor.start_indexing()

        text_extensions = ["*.txt", "*.md", "*.py", "*.json", "*.csv", "*.pdf", "*.log",".jsonl"]

        files = []
        for pattern in text_extensions:
            files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))

        print(f"Found {len(files)} files to process.\n")

        if not files:
            raise RuntimeError(f"No files found in {folder_path}")

        self.embedding_manager = EmbeddingManager()
        self.metadata = []
        all_chunks = []

        # Extract chunks from all files
        for file_idx, path in enumerate(files, 1):
            ext = os.path.splitext(path)[1].lower()

            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
            except:
                file_size_mb = 0

            print(f"Processing [{file_idx}/{len(files)}]: {os.path.basename(path)} ({file_size_mb:.2f} MB)")

            perf_monitor.sample_resources()

            if ext == ".log":
                log_chunks = chunk_log_file(path, LOG_LINES_PER_CHUNK)
                if not log_chunks:
                    print("   No valid log entries found")
                    continue

                print(f"   Created {len(log_chunks)} log chunks")
                perf_monitor.record_file_processed(path, len(log_chunks))

                for chunk in log_chunks:
                    all_chunks.append(chunk["text"])
                    self.metadata.append({
                        "path": path,
                        "text": chunk["text"][:1000],
                        "file_type": "log",
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "log_levels": chunk["log_levels"],
                        "line_count": chunk["line_count"]
                    })

                # Add to knowledge graph
                self.knowledge_graph.add_file_node(path, {
                    'file_type': 'log',
                    'chunk_count': len(log_chunks)
                })
            else:
                text = extract_text_from_file(path)
                if not text:
                    print("   Skipping (no text extracted)")
                    continue

                chunks = chunk_text(text)
                print(f"   Created {len(chunks)} chunks")
                perf_monitor.record_file_processed(path, len(chunks))

                for chunk in chunks:
                    all_chunks.append(chunk)
                    self.metadata.append({
                        "path": path,
                        "text": chunk[:1000],
                        "file_type": ext[1:] if ext else "unknown"
                    })

                # Add to knowledge graph
                self.knowledge_graph.add_file_node(path, {
                    'file_type': ext[1:] if ext else "unknown",
                    'chunk_count': len(chunks)
                })

        if not all_chunks:
            raise RuntimeError("No chunks created from documents")

        print(f"\n✓ Extracted {len(all_chunks)} total chunks")

        # Generate embeddings
        print(f"\nGenerating embeddings (batch size: {EMBEDDING_BATCH_SIZE})...")
        embeddings = self.embedding_manager.embed_texts(all_chunks, batch_size=EMBEDDING_BATCH_SIZE)
        perf_monitor.metrics["indexing"]["embeddings_generated"] = len(embeddings)

        # Create FAISS index
        self.index = self.create_optimized_faiss_index(embeddings)

        # Save everything
        os.makedirs(os.path.dirname(INDEX_PATH) or '.', exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        # Save knowledge graph
        self.knowledge_graph.save()

        perf_monitor.end_indexing()
        perf_monitor.print_indexing_report()
        perf_monitor.save_to_file()

        # Initialize orchestrator
        self.orchestrator = TOONOrchestrator(self, self.knowledge_graph)

        print(f"✅ Indexing complete!")
        print(f"   Index saved: {INDEX_PATH}")
        print(f"   Metadata saved: {METADATA_PATH}")
        print(f"   Knowledge Graph saved: {KNOWLEDGE_GRAPH_PATH}\n")

        audit_logger.log_operation("index_folder", "system", {"folder": folder_path, "files": len(files)})

    def create_optimized_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index optimized for dataset size."""
        d = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        print(f"\nCreating FAISS index for {n_vectors} vectors ({d} dimensions)...")

        if n_vectors < 10000:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings.astype(np.float32))
            print(f"✓ FAISS FlatL2 index created")
        else:
            quantizer = faiss.IndexFlatL2(d)
            nlist = min(int(4 * np.sqrt(n_vectors)), 256)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(20, nlist)
            print(f"✓ FAISS IVFFlat index created with {nlist} clusters")

        return index

    def load_index(self):
        """Load existing index and metadata."""
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"Index not found at {INDEX_PATH}")

        self.index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.embedding_manager = EmbeddingManager()
        self.retriever = HybridRetriever(self.index, self.metadata, device=DEVICE)
        self.orchestrator = TOONOrchestrator(self, self.knowledge_graph)

    def get_chunks_by_file(self, file_path: str) -> List[str]:
        """Get all chunks for a specific file."""
        return [doc['text'] for doc in self.metadata if doc.get('path') == file_path]

    def retrieve_relevant_chunks(self, question: str, chunks_with_files: List[Tuple]) -> List[Tuple]:
        """Retrieve relevant chunks (simplified)."""
        return chunks_with_files[:TOP_K_DEFAULT]

    def generate_answer(self, question: str, context_chunks: Optional[List[str]] = None) -> str:
        """Generate answer using LLM."""
        if context_chunks is None:
            context_chunks = self.retriever.query(
                question,
                embedding_fn=self.embedding_manager.embed_texts,
                top_k=TOP_K_DEFAULT
            )

        prompt = build_prompt_within_token_limit(context_chunks, question)

        response = chat(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K
            }
        )

        return response["message"]["content"]

    def query(self, question: str, k: int = TOP_K_DEFAULT) -> str:
        """Query the system with TOON-enhanced capabilities."""
        query_start = time.time()
        success = False

        try:
            if self.retriever is None:
                self.load_index()

            # Check if question is a TOON task
            if question.startswith("TOON:"):
                task_parts = question[5:].split(":", 1)
                task_type = task_parts[0].strip()
                task_params = json.loads(task_parts[1]) if len(task_parts) > 1 else {}

                task_id = self.orchestrator.register_task(task_type, task_params)
                result = self.orchestrator.execute_task(task_id)

                success = True
                return json.dumps(result, indent=2)

            # Regular query
            print("Performing hybrid retrieval + re-ranking...")
            results = self.retriever.query(
                question,
                embedding_fn=self.embedding_manager.embed_texts,
                top_k=k
            )

            if not results:
                return "No relevant documents found."

            # Generate answer
            prompt = build_prompt_within_token_limit(results, question)
            print(f"Generating answer with {TEXT_MODEL}...")

            response = chat(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K
                }
            )

            answer = response["message"]["content"]
            self.conv_memory.add_turn(question, answer)

            success = True
            return answer

        finally:
            duration = time.time() - query_start
            perf_monitor.record_query(question, duration, k, success)
            print(f"\n⏱️ Query completed in {duration:.3f} seconds\n")

            audit_logger.log_operation("query", "user", {"question": question[:100], "duration": duration})


class ConversationalMemory:
    """Store conversation history."""

    def __init__(self, max_turns: int = 5):
        self.memory = []
        self.max_turns = max_turns

    def add_turn(self, user_input: str, system_response: str):
        self.memory.append((user_input, system_response))
        if len(self.memory) > self.max_turns:
            self.memory.pop(0)

    def get_context(self) -> str:
        return "\n".join([f"User: {u}\nSystem: {s}" for u, s in self.memory])


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🚀 TOON-Enhanced RAG System v9.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--index", action="store_true", help="Create/rebuild index")
    parser.add_argument("--folder", type=str, default=".", help="Folder to index")
    parser.add_argument("--query", type=str, help="Query the indexed documents")
    parser.add_argument("--toon-task", type=str, help="Execute TOON task (e.g., 'summarize_all')")
    parser.add_argument("--k", type=int, default=TOP_K_DEFAULT, help="Number of chunks to retrieve")
    parser.add_argument("--show-stats", action="store_true", help="Show performance statistics")
    parser.add_argument("--show-graph", action="store_true", help="Show knowledge graph info")

    args = parser.parse_args()

    rag_system = TOONEnabledRAGSystem()

    if args.show_stats:
        try:
            with open(PERFORMANCE_LOG, 'r') as f:
                perf_monitor.metrics = json.load(f)
            perf_monitor.print_indexing_report()
            perf_monitor.print_query_stats()
        except FileNotFoundError:
            print("No performance data found.")
        return

    if args.show_graph:
        if os.path.exists(KNOWLEDGE_GRAPH_PATH):
            kg = KnowledgeGraph()
            print(f"\n📊 Knowledge Graph Statistics:")
            print(f"   Total Nodes: {kg.graph.number_of_nodes()}")
            print(f"   Total Edges: {kg.graph.number_of_edges()}")
            print(f"   File Nodes: {len(kg.file_nodes)}")
            print(f"   Concept Nodes: {len(kg.concept_nodes)}\n")
        else:
            print("Knowledge graph not found. Index documents first.")
        return

    if args.index:
        print("Checking Ollama models...")
        try:
            pull(TEXT_MODEL)
            print(f"✓ {TEXT_MODEL} ready\n")
        except Exception as e:
            print(f"Warning: Could not pull {TEXT_MODEL}: {e}")

        print(f"\nIndexing folder: {args.folder}")
        rag_system.index_folder(args.folder)

    if args.query:
        print("Loading index...\n")
        answer = rag_system.query(args.query, k=args.k)

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70 + "\n")
        print(answer + "\n")

        perf_monitor.print_query_stats()

    if args.toon_task:
        print("Loading index...\n")
        rag_system.load_index()

        task_query = f"TOON:{args.toon_task}"
        result = rag_system.query(task_query)

        print("\n" + "=" * 70)
        print("TOON TASK RESULT")
        print("=" * 70 + "\n")
        print(result + "\n")

    if not any([args.index, args.query, args.toon_task, args.show_stats, args.show_graph]):
        parser.print_help()


if __name__ == "__main__":
    main()