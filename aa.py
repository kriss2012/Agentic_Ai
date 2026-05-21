import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def run_llm(prompt: str,
             model="openai/gpt-oss-120b",
             temperature=0.7,
             max_tokens=256):

             llm = ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
             )

             response = llm.invoke(prompt)

             print(response.content)

run_llm("what is Gen AI?")

def process_pdf(file_path:str):
   loader = PyPDFLoader(file_path)
   pages =  loader.load()

   splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=100
   )

   splits = splitter.split_documents(pages)
   print(f'the document has been split into: {len(splits)}chunks')

   return splits

pages = process_pdf("./scholarship_info.pdf")