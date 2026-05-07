"""
RAG Pipeline — Core ingestion and query logic.

Flow:
  PDF → PyMuPDF → RecursiveTextSplitter → AnthropicEmbeddings → FAISS
  Query → AnthropicEmbeddings → FAISS similarity_search → Claude answer
"""

import os
from typing import Optional
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import AnthropicEmbeddings, ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import Settings


SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer the question, say:
"I couldn't find a clear answer in the document. Please try rephrasing your question."

Be concise, accurate, and cite the source filename when relevant.

Context:
{context}
"""


class RAGPipeline:
    """
    End-to-end RAG pipeline using LangChain + Anthropic + FAISS.
    
    Usage:
        pipeline = RAGPipeline(settings)
        pipeline.ingest_pdf("doc.pdf", source_name="doc.pdf")
        result = pipeline.query("What are the main findings?")
        print(result["answer"])
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._vectorstore: Optional[FAISS] = None
        self._initialized = False

        # Embedding model
        self.embeddings = AnthropicEmbeddings(
            model="voyage-3",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
        )

        # LLM
        self.llm = ChatAnthropic(
            model=settings.MODEL_NAME,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1,
            max_tokens=2048,
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Ensure vectorstore path exists
        Path(settings.VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.DATA_RAW_PATH).mkdir(parents=True, exist_ok=True)

        # Try to load existing vectorstore
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        """Load an existing FAISS index if one exists on disk."""
        index_path = self.settings.VECTORSTORE_PATH
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            try:
                self._vectorstore = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._initialized = True
                print(f"[RAG] Loaded existing FAISS index from {index_path}")
            except Exception as e:
                print(f"[RAG] Could not load existing index: {e}")

    def ingest_pdf(self, pdf_path: str, source_name: str = "") -> int:
        """
        Ingest a PDF file into the vector store.
        
        Args:
            pdf_path: Absolute path to the PDF file
            source_name: Human-readable name (filename) for metadata
            
        Returns:
            Number of chunks indexed
        """
        # 1. Extract text with PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if not full_text.strip():
            raise ValueError(f"Could not extract text from {source_name}. Is it a scanned PDF?")

        # 2. Split into chunks
        raw_chunks = self.splitter.split_text(full_text)
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": source_name, "chunk_index": i},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

        # 3. Embed and store
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self._vectorstore.add_documents(documents)

        # 4. Persist to disk
        self._vectorstore.save_local(self.settings.VECTORSTORE_PATH)
        self._initialized = True

        print(f"[RAG] Ingested '{source_name}': {len(documents)} chunks indexed")
        return len(documents)

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a question using the indexed documents.
        
        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve
            
        Returns:
            dict with 'answer' (str) and 'sources' (list[str])
        """
        if not self._initialized or self._vectorstore is None:
            raise RuntimeError("No documents indexed. Call ingest_pdf() first.")

        # 1. Retrieve relevant chunks
        docs = self._vectorstore.similarity_search(question, k=top_k)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        sources = list({d.metadata.get("source", "unknown") for d in docs})

        # 2. Build prompt + chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        return {"answer": answer, "sources": sources, "context_chunks": len(docs)}

    def is_initialized(self) -> bool:
        """Check if the pipeline has an active index."""
        return self._initialized

    def clear(self) -> None:
        """Clear the in-memory index. Does not delete files from disk."""
        self._vectorstore = None
        self._initialized = False
        print("[RAG] Index cleared from memory")
