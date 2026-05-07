"""
Victor RAG Document Q&A System — FastAPI Entry Point
Author: Olusegun (Victor) Makanju
GitHub: https://github.com/VictorM21
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

from app.rag_pipeline import RAGPipeline
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Victor RAG Document Q&A",
    description="Upload PDFs and ask questions — powered by Claude + LangChain + FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag = RAGPipeline(settings=settings)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Victor RAG Document Q&A API is running",
        "docs": "/docs",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document and ingest it into the vector store.
    
    - Extracts text using PyMuPDF
    - Splits into chunks (configurable via CHUNK_SIZE in .env)
    - Embeds with Anthropic embeddings
    - Stores in FAISS vector index
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        num_chunks = rag.ingest_pdf(tmp_path, source_name=file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": num_chunks,
            "message": f"Document ingested. {num_chunks} chunks indexed. You can now ask questions.",
        }
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question against the indexed documents.
    
    Returns a Claude-generated answer grounded in retrieved document chunks.
    """
    if not rag.is_initialized():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload a PDF first via /upload",
        )

    result = rag.query(question=request.question, top_k=request.top_k)
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
    )


@app.delete("/index")
async def clear_index():
    """Clear the vector store index. Use with caution."""
    rag.clear()
    return {"status": "success", "message": "Vector index cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
