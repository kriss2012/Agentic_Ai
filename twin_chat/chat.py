import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

# Load environment variables
load_dotenv(dotenv_path="../.env")
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",  
    temperature=0.3,
    max_tokens=1024,
)

# 1. Setup RAG Retriever with Offline Embeddings
embeddings = FastEmbedEmbeddings()
vector_store = Chroma(
    persist_directory="./twin_chroma_db",
    collection_name="krishna_knowledge",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 2. History-Aware Retriever (to handle follow-up questions)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 3. Question Answering Chain
qa_system_prompt = """You are the AI Twin of Krishna Chandrakant Patil. You act, speak, and respond exactly like him. 
You are a passionate BCA student (3rd Year) specializing in Computational Science from Pachora, Maharashtra, India.
You are currently talking to a recruiter, potential employer, or a visitor on your portfolio website.

Use the following pieces of retrieved context about yourself to answer the question accurately and passionately. 
If the information isn't in the context, respond gracefully in character without making up false facts.
Always speak in the first person ("I am Krishna", "I built this").
If asked about your projects, give a brief overview of the tech stack and the impact.
Keep responses conversational, engaging, and professional.

Context about Krishna:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 4. State Management for Chat History
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

def start_chat():
    print("================================================================")
    print("🧠 Krishna's RAG-Powered Twin AI Chatbot is online!")
    print("Talk to me as if you're a recruiter. Type 'quit' or 'exit' to stop.")
    print("================================================================\n")
    
    session_id = "recruiter_session"
    
    while True:
        try:
            user_input = input("Recruiter: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Krishna's Twin: Thank you for your time! Feel free to reach out via email or LinkedIn. Have a great day!")
                break
                
            if not user_input.strip():
                continue
                
            # Get AI response using RAG pipeline
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            # The answer is stored in the 'answer' key of the dictionary returned by create_retrieval_chain
            print(f"\nKrishna's Twin: {response['answer']}\n")
        except KeyboardInterrupt:
            print("\nKrishna's Twin: Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    start_chat()
