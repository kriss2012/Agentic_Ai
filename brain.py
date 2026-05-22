import os
print("Imported os")
from dotenv import load_dotenv
print("Imported load_dotenv")
from langchain_groq import ChatGroq
print("Imported ChatGroq")
from langchain_community.document_loaders import TextLoader
print("Imported TextLoader")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Imported RecursiveCharacterTextSplitter")
from langchain_google_genai import GoogleGenerativeAIEmbeddings
print("Imported GoogleGenerativeAIEmbeddings")
from langchain_chroma import Chroma
print("Imported Chroma")
from langchain.chains import RetrievalQA
print("Imported RetrievalQA")
from langchain_community.tools import tool, DuckDuckGoSearchRun
print("Imported tools")
from langchain.agents import create_react_agent, AgentExecutor
print("Imported agents")
from langchain_core.prompts import PromptTemplate
print("Imported PromptTemplate")
import requests
import warnings
warnings.filterwarnings("ignore")
print("Finished imports")

load_dotenv()
print("Loaded dotenv")

# --- 1. Setup RAG for Resume ---
def create_resume_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(base_dir, 'resume_chroma_db')
    resume_path = os.path.join(base_dir, 'twin_chat', 'resume_text.txt')
    
    if not os.path.exists(persist_dir):
        print("Initializing Knowledge Base from resume...")
        loader = TextLoader(resume_path, encoding="utf-8")
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = splitter.split_documents(pages)

        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name='resume_info'
        )
    else:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_name='resume_info'
        )
        
    return vector_store.as_retriever(search_kwargs={'k': 3})

global_retriever = None
try:
    global_retriever = create_resume_retriever()
except Exception as e:
    print(f"Warning: Could not initialize resume retriever: {e}")

@tool
def get_resume_info(query: str) -> str:
    """Use this tool to answer questions about Krishna Patil's background, education, skills, projects, and personal information."""
    if not global_retriever:
        return "Resume knowledge base is not available."
        
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.03, max_tokens=512)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=global_retriever,
    )
    response = rag_chain.invoke({"query": query})
    return response["result"]

# --- 2. Setup Additional Tools ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather information for a specific city."""
    api_key = os.getenv("WeatherStack_Api_Key")
    if not api_key:
        return "WeatherStack API key is not configured."
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    try:
        result = requests.get(url).json()
        if "current" in result:
            temp = result["current"]["temperature"]
            desc = result["current"]["weather_descriptions"][0]
            return f"The current weather in {city} is {desc} with a temperature of {temp} degrees Celsius."
        else:
            return "Could not fetch weather data."
    except Exception as e:
        return f"Error fetching weather: {e}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this tool for any math-related queries."""
    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# --- 3. Setup Agent ---
def run_agent():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=1024)
    search_tool = DuckDuckGoSearchRun()

    tools = [get_resume_info, get_weather, search_tool, calculator]

    template = """You are the digital twin of Krishna Patil. You talk like Krishna, share his skills and experiences from his resume, and act as a helpful personal AI assistant. 
Your goal is to impress job recruiters and help users with daily life tasks.
You are confident, enthusiastic, and highly knowledgeable about AI, ML, and Software Development.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If a user asks about your personal background, education, or projects, use the 'get_resume_info' tool to fetch accurate details from your resume, then answer in the first person as Krishna.

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    print("==================================================")
    print("Welcome to Krishna Patil's Digital Twin Assistant!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("==================================================\n")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Digital Twin: Goodbye! Feel free to reach out if you need anything else.")
                break
            if not user_input.strip():
                continue
            
            response = agent_executor.invoke({"input": user_input})
            print(f"\nDigital Twin: {response['output']}")
            
        except KeyboardInterrupt:
            print("\nDigital Twin: Goodbye! Feel free to reach out if you need anything else.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_agent()
