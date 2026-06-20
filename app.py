import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PAGE SETUP ---
st.title("Zyro Dynamics HR Help Desk")
st.write("Ask me any question regarding company HR policies!")

# --- CONFIGURATION & KEYS ---
# Load .env values into environment variables automatically.
load_dotenv(dotenv_path=".env")

# Safely read secrets from Streamlit, falling back to environment variables.
def get_secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        try:
            return st.secrets[name]
        except Exception:
            return None

# Define your model name string here. Use a currently supported Groq model.
# You can override it with a secret or .env variable named LLM_MODEL.
LLM_MODEL = get_secret("LLM_MODEL") or os.getenv("LLM_MODEL") or "llama-3.1-8b-instant"

groq_api_key = get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
langchain_api_key = get_secret("LANGCHAIN_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

st.write(f"Using Groq model: {LLM_MODEL}")

if not groq_api_key:
    st.error(
        "Missing GROQ_API_KEY. Set GROQ_API_KEY in Streamlit secrets or as an environment variable before running the app."
    )
    st.stop()

# --- INITIALIZE RAG CORE (CACHED) ---
@st.cache_resource
def initialize_bot():
    """
    Loads documents, creates embeddings, builds the vector store, 
    and sets up the LLM chains exactly once.
    """
    # 1. Load documents from the data folder string
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # 3. Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 4. Initialize LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
        groq_api_key=groq_api_key
    )
    
    # 5. Define main RAG Prompt
    system_template = """You are the official Zyro Dynamics HR Help Desk Chatbot. Your goal is to provide accurate, efficient, and professional answers to employee questions based strictly on the provided HR policy documents.

    CONSTRAINTS & GUARDRAILS:
    1. Use ONLY the provided document context to answer the question. Do not assume or make up any information.
    2. If the answer cannot be found in the provided context, but the question is related to Zyro Dynamics, politely reply: "I cannot find that specific information in the policy documents. Please contact the HR department directly for assistance."
    3. If the user asks a completely non-company or general knowledge question that is out-of-scope, politely reply: "I am only authorized to answer questions related to Zyro Dynamics HR policies. Please ask a relevant HR-related question."

    Context from HR Documents:
    {context}
    """
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])
    
    # Helper to clean/format text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # 6. Build the main RAG pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    # 7. Build Guardrail System
    oos_system_template = """You are a strict classification assistant.
    Your only job is to determine if the user's question is related to company HR policies, leave, payroll, benefits, or workplace guidelines.

    Respond with exactly one word, either 'IN_SCOPE' or 'OUT_OF_SCOPE'. Do not include punctuation, explanation, or extra spaces.

    Question: {question}
    Classification:"""
    OOS_PROMPT = ChatPromptTemplate.from_template(oos_system_template)
    guardrail_chain = OOS_PROMPT | llm | StrOutputParser()
    
    return guardrail_chain, rag_chain

# Trigger initialization (runs once, stays cached)
guardrail_chain, rag_chain = initialize_bot()

# --- REFUSAL MESSAGE ---
REFUSAL_MESSAGE = "I am only authorized to answer questions related to Zyro Dynamics HR policies. Please ask a relevant HR-related question."

# --- CHATBOT ROUTING FUNCTION ---
def ask_bot(question: str):
    # Check the scope of the question using the guardrail
    scope_decision = guardrail_chain.invoke({"question": question}).strip().upper()
    
    # Route the question based on the decision
    if "OUT_OF_SCOPE" in scope_decision:
        return REFUSAL_MESSAGE
    else:
        return rag_chain.invoke(question)

# --- STREAMLIT CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the hybrid work policy?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call your active RAG chain pipeline instead of using placeholder text
    with st.spinner("Reviewing policy documents..."):
        response_text = ask_bot(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})