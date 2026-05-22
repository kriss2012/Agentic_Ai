import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def run_llm(
    prompt: str,
    model="llama-3.1-8b-instant",
    temperature=0.07,
    max_tokens=256,
):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = llm.invoke(prompt)
    print(response.content)

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
        )
        
    splits = splitter.split_documents(pages)

    print(f'document has been split into: {len(splits)}chunks')

    return splits

def generate_and_store_embeddings():
    splits = process_pdf("./scholarship_info.pdf")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory='rag_chroma_db',
        collection_name='scholarship_info'
    )

        vector_store.add_documents(splits)

    def create_retretriever():
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory='rag_chroma_db',
            collection_name='scholarship_info'
        )
        return vector_store.as_retriever(search_kwargs={'k': 2})
    
def get_llm(model="llama-3.1-8b-instant"):

    llm = ChatGroq(
        model=model,
        temperature=0.03,
        max_tokens=256
    )

return llm
#if __name__ == "__main__":
   # generate_and_store_embeddings()
