import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_classic.agents import create_agent
from langsmith import Client
import requests
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

# 1 & 2. FIXED: Moved outside the previous function and fixed the typo
def create_retriever():
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

def create_rag_chain(prompt:str):
    llm = get_llm()
    retriever = create_retriever()
    
    # 4. FIXED: Changed RetrieverQA to RetrievalQA
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    response = rag_chain.invoke({"query": prompt})
    return response 
    
#print(create_rag_chain("What is the future scope of expected investment in scholarships? "))

@tool
def get_weather(city:str) -> str:
    """Get the current weather information for a specific city."""
    api_key= os.getenv("WeatherStack_Api_Key")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    result = requests.get(url).json()
    return str(result)

print(get_weather.invoke({"city":"Pachora"}))


def run_agent():
    llm = get_llm()
    search_tool = DuckDuckGoSearchRun()

    tools = [get_weather, search_tool]