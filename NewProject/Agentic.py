import hashlib
import math
import os
import re
from typing import Any, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


class SimpleEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "simple-text-embedding"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "SimpleEmbeddingFunction":
        return SimpleEmbeddingFunction()

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name()}

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str):
        tokens = re.findall(r"\b\w+\b", text.lower())
        vec = [0.0] * 128
        for token in tokens:
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 128
            vec[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0:
            return [0.0] * 128
        return [value / norm for value in vec]


class LocalRAGSystem:
    def __init__(self, db_dir=None, collection_name="scholarship_docs"):
        self.db_dir = db_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_chroma_db"))
        os.makedirs(self.db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=SimpleEmbeddingFunction()
        )
        self.splits = []

    def ingest_document(self, file_path):
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError("No PDF file found. Please provide a valid PDF path.")

        print(f"Loading document from {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        print("Splitting document")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        self.splits = splitter.split_documents(docs)
        print(f"Split into {len(self.splits)} chunks")

    def setup_database(self):
        if not self.splits:
            raise ValueError("No document splits available. Please ingest a document first.")

        print("Initializing embeddings and Chroma database")
        ids = [f"chunk_{i}" for i in range(len(self.splits))]
        texts = [chunk.page_content for chunk in self.splits]
        metadatas = [{"chunk_index": i} for i in range(len(self.splits))]

        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
        print(f"Stored {len(self.splits)} chunks in vector DB at {self.db_dir}")

    def query(self, question, k=2):
        results = self.collection.query(query_texts=[question], n_results=k)
        documents = results.get("documents", [])
        return documents[0] if documents else []

    def create_llm(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY or pass api_key to create_llm().")
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)
        return self.llm

    def answer_query(self, question, k=2):
        if not hasattr(self, "llm") or self.llm is None:
            raise ValueError("LLM not initialized. Call create_llm() first.")

        documents = self.query(question, k=k)
        context = "\n\n".join([doc for doc in documents if doc])
        prompt = [
            SystemMessage(content="You are a helpful assistant. Use only the provided context to answer the user's question."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @staticmethod
    def find_pdf(start_dir=None):
        start_dir = start_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for root, _, files in os.walk(start_dir):
            for file_name in files:
                if file_name.lower().endswith(".pdf"):
                    return os.path.join(root, file_name)
        return None


if __name__ == "__main__":
    rag = LocalRAGSystem()
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scholarship_info.pdf"))
    if not os.path.exists(pdf_path):
        pdf_path = LocalRAGSystem.find_pdf()

    rag.ingest_document(pdf_path)
    rag.setup_database()

    sample_question = "What is this document about?"
    answers = rag.query(sample_question, k=2)

    print("Sample vector search results:")
    for idx, answer in enumerate(answers, 1):
        print(f"{idx}. {answer[:220]}...")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        print("\nCreating Google GenAI LLM...")
        rag.create_llm(api_key=google_api_key)
        response = rag.answer_query(sample_question, k=2)
        print("\nLLM response:")
        print(response)
    else:
        print("\nGOOGLE_API_KEY not set. Skipping LLM creation. Set the environment variable to enable Google GenAI.")
