import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Load env variables
load_dotenv(dotenv_path="../.env")
load_dotenv()

def build_rag():
    print("Loading Knowledge Base...")
    # Load the comprehensive knowledge base
    loader = TextLoader("knowledge_base.md", encoding="utf-8")
    docs = loader.load()

    print("Splitting documents...")
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Generating FastEmbed embeddings and storing in Chroma DB...")
    # Initialize offline embeddings (no API key required)
    embeddings = FastEmbedEmbeddings()

    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./twin_chroma_db",
        collection_name="krishna_knowledge"
    )
    
    print("RAG database built successfully at './twin_chroma_db'!")

if __name__ == "__main__":
    build_rag()
