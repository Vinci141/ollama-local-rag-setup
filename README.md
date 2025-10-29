# 🧠 Ollama Local RAG Setup
> Run a fully local Retrieval-Augmented Generation (RAG) system using Ollama — index and query your own documents directly on your machine.

---

## 📘 Overview
This project lets you build a **local knowledge assistant** powered by [Ollama](https://ollama.ai/).  
It reads and indexes your local files, creates embeddings, and enables **semantic search + Q&A** without sending data to the cloud.

✅ **Everything runs locally** — no external API keys or internet required.  
🧩 Ideal for privacy-sensitive projects, research, and offline data exploration.

---

## ⚙️ Features
- 🔍 **Automatic document indexing** with FAISS
- 🧬 **Embeddings** powered by `mxbai-embed-large` (Ollama)
- 💬 **Natural language querying** over your local files
- 💾 **Offline support** — runs entirely on your machine
- 🧰 **Modular design** — easy to extend with other models or data sources

---

## 🚀 Getting Started

### 1️⃣ Prerequisites
- [Ollama](https://ollama.ai/download) installed and running locally  
  > Verify with:
  > ```bash
  > ollama serve
  > ```
- Python ≥ 3.9
- `git`, `pip`, and a code editor (PyCharm, VS Code, etc.)
  

---

### 2️⃣ Clone the Repository
    bash
    git clone https://github.com/<your-username>/ollama-local-rag-setup.git
    cd ollama-local-rag-setup

###3️⃣ Create a Virtual Environment

    python -m venv .venv
    .\.venv\Scripts\activate      # On Windows
    source .venv/bin/activate     # On macOS / Linux

###4️⃣ Install Dependencies

    pip install -r requirements.txt

###5️⃣ Pull Required Ollama Models

    ollama pull mxbai-embed-large
    ollama pull llama3

6️⃣ Index Your Local Files

python Ollama_setup.py --index --folder "C:\path\to\your\docs"
  This step:

    Reads files from the given folder
    Generates embeddings
    Builds a FAISS vector index for fast retrieval
7️⃣ Ask Questions About Your Files

python 02_Ollama_setup.py --query "Summarize project goals"

📂 Project Structure

    ollama-local-rag-setup/
    │
    ├── Ollama_setup.py      # Main script for indexing and querying
    ├── requirements.txt         # Python dependencies
    ├── data/                    # Folder for source documents [Optionally use this in arguement -->C:\path\to\your\docs]
    ├── index/                   # Stored FAISS vector database
    └── README.md


    Content of requirement.txt

    # Core Ollama client
    ollama>=0.1.0
    
    # Vector database and numeric operations
    faiss-cpu>=1.7.4
    numpy>=1.24.0

    # PDF processing
    pypdf>=3.17.0

    # Note: These are already included in Python standard library
    # - os
    # - glob
    # - json
    # - typing
    # - argparse

🧩 Architecture

    User Query
      ↓
    Ollama LLM (e.g., Llama 3)
      ↓
    Relevant Context Retrieved from FAISS
      ↓
    Embedded via mxbai-embed-large
      ↓
    Response Generated → Displayed to User


💡 Example Output

    > python 02_Ollama_setup.py --query "What is this document about?"

Answer:
This document describes the internal architecture and deployment setup
for the RAG-based Ollama local environment...

🧰 Tech Stack

| Component             | Description                          |
| --------------------- | ------------------------------------ |
| **Ollama**            | Local LLM runtime                    |
| **mxbai-embed-large** | Embedding model for document vectors |
| **FAISS**             | Vector search engine                 |
| **Python**            | Glue logic for indexing and querying |


🔒 Privacy

All processing happens locally — no data leaves your machine.
Perfect for confidential files, research notes, or enterprise data.
