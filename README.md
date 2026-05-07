# 🔍 Victor RAG Document Q&A System

> **RAG (Retrieval-Augmented Generation) pipeline** that lets you upload PDFs and ask questions in natural language — powered by Claude (Anthropic) + LangChain + FAISS.

## 🎯 What This Does

Upload any PDF document and ask it questions in plain English. The system:

1. **Ingests** your PDF and splits it into chunks
2. **Embeds** each chunk using Anthropic embeddings  
3. **Stores** vectors in a FAISS index for fast similarity search
4. **Retrieves** the most relevant chunks for your query
5. **Generates** a grounded answer using Claude (claude-3-5-sonnet)

## 🏗️ Project Structure

```
victor-rag-document-qa/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── rag_pipeline.py      # Core RAG logic (ingest + query)
│   ├── embeddings.py        # Embedding utility functions
│   └── config.py            # App configuration
├── data/
│   ├── raw/                 # Place your PDF files here
│   └── vectorstore/         # Auto-generated FAISS index
├── notebooks/
│   └── 01_rag_exploration.ipynb
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

```bash
git clone https://github.com/VictorM21/victor-rag-document-qa.git
cd victor-rag-document-qa
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude 3.5 Sonnet (Anthropic) |
| Orchestration | LangChain 0.2+ |
| Vector Store | FAISS |
| API Layer | FastAPI |

## 👤 Author

**Olusegun (Victor) Makanju** — Data Scientist → AI Engineer  
GitHub: [@VictorM21](https://github.com/VictorM21)
