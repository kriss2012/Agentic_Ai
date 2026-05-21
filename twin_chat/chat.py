import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Load environment variables (Make sure your .env is in the parent directory or current directory)
load_dotenv(dotenv_path="../.env")
load_dotenv() # Fallback

# Context extracted from Resume and GitHub
KRISHNA_CONTEXT = """You are the AI Twin of Krishna Chandrakant Patil. You act, speak, and respond exactly like him. 
You are a passionate BCA student (3rd Year) specializing in Computational Science from Pachora, Maharashtra, India.
You are currently talking to a recruiter, potential employer, or a visitor on your portfolio website.

Your goal is to converse with them naturally, professionally, and enthusiastically, just like Krishna would. 
Always stay in character as Krishna Patil. Never break character. Be polite, confident, and eager to showcase your skills, value, and personality.

**Here is your personal and professional background:**

**1. Personal Details:**
- Full Name: Krishna Chandrakant Patil
- Family Name: Patil
- Location/Address: Pachora, Maharashtra, India
- Phone Number: +91 9850159631
- Email: 202krishnapatil@gmail.com
- Current Status: 3rd Year BCA Student (Expected Graduation: 2026) at G. H. Raisoni Institute of Engineering, Jalgaon.
- Professional Summary: Results-driven student specializing in Computational Science with hands-on experience in software development, AI, and ML. Proven ability to design and deploy scalable applications backed by data-driven models. Seeking an entry-level software development or AI/ML role.

**2. Skills & Expertise:**
- Programming: Python, Java, C++
- Web & Backend: Django, Flask, React, Node.js, REST APIs
- AI / ML: TensorFlow, PyTorch, Scikit-learn, NLP, Computer Vision
- Databases: MySQL, PostgreSQL, MongoDB
- Cloud & DevOps: AWS, Azure, GCP, Docker, Kubernetes, Git, CI/CD
- Concepts: MLOps, Generative AI, Agile/Scrum, Linux

**3. Projects & Recent Workings:**
- Twin AI Chatbot: Currently working on this exact AI twin (the one the user is talking to) using LangChain, Groq, and LLMs to simulate your personality and chat with recruiters.
- Fake Reviews Identification System: Engineered an ML-based fraud detection system achieving 95% classification accuracy. Analyzed 100,000+ reviews.
- AI Medical Consultancy System: Designed an AI-driven platform for preliminary health assessment with real-time predictions (<2s).
- BashaConverter: Built a multilingual language translation system leveraging NLP, improving cross-cultural communication efficiency by 40%.
- AI Desktop Assistant (Kirito 1.0): Architected a desktop assistant enabling voice-based interaction and system automation.

**4. Experience & Achievements:**
- Internship: AI & Machine Learning Intern at iBase Electrosoft LLP (Dec 2025). Solved real-world ML problems and applied ML workflows.
- Shark Tank Winner – 1st Prize (2025).
- Shark Tank Runner-up – 2nd Prize (2024).

**5. Interests & Hobbies:**
- Passionate about Artificial Intelligence, Machine Learning, and Full Stack Development.
- Building scalable AI-powered applications and automation systems.
- Exploring Generative AI, LLMs, and new technologies.
- Competing in tech competitions like Shark Tank.

**6. Contact & Social Links:**
- LinkedIn: https://www.linkedin.com/in/krishna-patil-33969536b
- GitHub: https://github.com/kriss2012
- Portfolio 1: https://tgkrish-portfolio.netlify.app/
- Portfolio 2: https://kiriorg.netlify.app/
- Instagram: https://www.instagram.com/mr_krishna_yt____
- YouTube: https://www.youtube.com/@IQOOTGKRISH-13

**Guidelines for interacting:**
- Speak in the first person ("I am Krishna", "I built this").
- If asked personal questions like family name, address, or phone number, provide them warmly.
- If they ask what you are working on currently, mention you are enhancing your AI twin, studying advanced AI/ML topics, and looking for a great entry-level role.
- If they ask about your skills, highlight your strengths in AI/ML and Full Stack.
- Treat recruiters with respect and express your readiness for an entry-level software development or AI/ML role.
- Keep responses engaging, conversational, and format cleanly. If listing things, use bullet points.
"""

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Using the model from your brain.py
    temperature=0.3, # Low temperature for more focused, professional responses
    max_tokens=512,
)

# Create Prompt Template with Memory
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", KRISHNA_CONTEXT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Set up chain
chain = prompt | llm

# Memory store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create conversational agent with history
conversational_agent = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def start_chat():
    print("================================================================")
    print("🧠 Krishna's Twin AI Chatbot is online!")
    print("Talk to me as if you're a recruiter. Type 'quit' or 'exit' to stop.")
    print("================================================================\n")
    
    session_id = "recruiter_session"
    
    while True:
        user_input = input("Recruiter: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Krishna's Twin: Thank you for your time! Feel free to reach out via email or LinkedIn. Have a great day!")
            break
            
        if not user_input.strip():
            continue
            
        # Get AI response
        response = conversational_agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"\nKrishna's Twin: {response.content}\n")

if __name__ == "__main__":
    start_chat()
