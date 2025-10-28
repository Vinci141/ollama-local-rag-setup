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
```bash
git clone https://github.com/<your-username>/ollama-local-rag-setup.git
cd ollama-local-rag-setup
