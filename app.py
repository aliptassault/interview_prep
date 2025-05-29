## RAG Q&A Conversation With PDF Including Chat History
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import ast
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import uvicorn
from typing import List
import os
import threading
import ast
import requests
import streamlit as st
from glob import glob



from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']="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"


# Create FastAPI app
app = FastAPI()

# Define a model for chat input
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# Store fine-prints globally for simplicity
fine_prints = []

def extract_fine_prints_from_documents(documents, llm):
    """Extracts fine-prints from the documents using prompt engineering and the LLM."""

    # Combine all document texts
    all_text = "\n\n".join([doc.page_content for doc in documents])
    # Define a prompt for extracting fine-prints
    fine_prints_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract all fine-prints, disclaimers, and important legal notes from the following document. Return each fine-print as a separate item in a list. If none are found, return an empty list."),
    ])
    # Run the LLM to extract fine-prints
    chain = create_stuff_documents_chain(llm, fine_prints_prompt)
    result = chain.invoke({"input": all_text})
    # Try to parse as list, fallback to splitting by newlines
    try:
        extracted = ast.literal_eval(result['answer'])
        if not isinstance(extracted, list):
            raise ValueError
        return extracted
    except Exception:
        return [line.strip() for line in result['answer'].split('\n') if line.strip()]

@app.get("/fine-prints")
async def get_fine_prints():
    """API endpoint to extract fine-prints from uploaded PDFs (expects files in ./data/ or similar)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key="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", model_name="Gemma2-9b-It")
    # For demo, load all PDFs in a folder (e.g., ./data/)
    documents = []
    for pdf_path in glob("./data/**/*.pdf", recursive=True):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    print(f"Loaded {len(documents)} documents from PDFs.")
    if not documents:
        return JSONResponse(content={"fine_prints": []})
    fine_prints_result = extract_fine_prints_from_documents(documents, llm)
    return JSONResponse(content={"fine_prints": fine_prints_result})

@app.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint for chat with PDF content (expects session_id and user_input)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key="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", model_name="Gemma2-9b-It")
    # For demo, load all PDFs in a folder (e.g., ./data/)
    documents = []
    for pdf_path in glob("./data/**/*.pdf", recursive=True):
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    if not documents:
        return JSONResponse(content={"answer": "No documents found.", "chat_history": []})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    # Minimal chat history for API
    session_histories = getattr(chat, "_session_histories", {})
    if request.session_id not in session_histories:
        session_histories[request.session_id] = ChatMessageHistory()
    session_history = session_histories[request.session_id]
    setattr(chat, "_session_histories", session_histories)
    # Prompts
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    def get_session_history(session: str) -> BaseChatMessageHistory:
        return session_histories[session]
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    response = conversational_rag_chain.invoke(
        {"input": request.user_input},
        config={"configurable": {"session_id": request.session_id}},
    )
    return JSONResponse(content={"answer": response['answer'], "chat_history": session_history.messages})

# --- Streamlit UI (optional, only runs if script is executed directly) ---
if __name__ == "__main__":
    st.title("Conversational RAG With PDF uploads and chat history (Frontend)")
    st.write("Upload PDFs and chat with their content. Uses FastAPI backend.")
    api_url = "http://localhost:8000"
    # Fine-prints
    if st.button("Extract Fine Prints from PDFs in ./data/"):
        resp = requests.get(f"{api_url}/fine-prints")
        try:
            json_data = resp.json()
            st.write("response:", json_data)
            if resp.ok:
                st.write("Fine Prints:", json_data.get("fine_prints", []))
            else:
                st.error("Failed to extract fine prints.")
        except Exception as e:
            st.error(f"Failed to parse response as JSON: {e}")
            st.write("Raw response:", resp.text)
    # Chat
    session_id = st.text_input("Session ID", value="default_session")
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        resp = requests.post(f"{api_url}/chat", json={"session_id": session_id, "user_input": user_input})
        if resp.ok:
            st.write("Assistant:", resp.json()["answer"])
            st.write("Chat History:", resp.json()["chat_history"])
        else:
            st.error("Failed to get answer from backend.")
