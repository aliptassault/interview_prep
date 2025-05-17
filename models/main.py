from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema.runnable import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import Tool
from langgraph.graph import StateGraph, END
import json
import os

# Load and split documents
doc_loader = DirectoryLoader("./enterprise_docs", loader_cls=TextLoader)
documents = doc_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
faiss_db = FAISS.from_documents(split_docs, embedding_model)
retriever = faiss_db.as_retriever()

# Chat Models
llm_models = {
    "gpt-4": ChatOpenAI(model_name="gpt-4", temperature=0.3),
    "claude-3-sonnet": HuggingFaceEndpoint(endpoint_url="https://api.anthropic.com/claude-3-sonnet", temperature=0.3),
    "gemma2-9b-it": HuggingFaceEndpoint(endpoint_url="https://api.huggingface.co/gemma2-9b-it", temperature=0.3),
    "mistral": HuggingFaceEndpoint(endpoint_url="https://api.huggingface.co/mistral", temperature=0.3),
    "deepseek": HuggingFaceEndpoint(endpoint_url="https://api.huggingface.co/deepseek", temperature=0.3),
}

# Select active model
def get_llm(model_name="gpt-4"):
    return llm_models.get(model_name, llm_models["gpt-4"])

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompts
rag_prompt = PromptTemplate.from_template(
    "Answer the question using the context below:\n\n{context}\n\nQuestion: {question}"
)

fallback_prompt = PromptTemplate.from_template(
    "Answer the question using your own knowledge:\n\nQuestion: {question}"
)

# Logging setup
LOG_FILE = "unanswered_queries.json"
unanswered_log = []

def log_unanswered_query(query):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)
    with open(LOG_FILE, "r+") as f:
        data = json.load(f)
        data.append({"query": query})
        f.seek(0)
        json.dump(data, f, indent=2)

# LangGraph nodes
def retrieve_context(state):
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    if not docs:
        unanswered_log.append(query)
        log_unanswered_query(query)
    return {
        "question": query,
        "docs": docs,
        "chat_history": state.get("chat_history", []),
        "model_name": state.get("model_name", "gpt-4")
    }

def route(state):
    return "rag_chain" if state["docs"] else "fallback_chain"

def run_rag(state):
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    model = get_llm(state["model_name"])
    chain = LLMChain(llm=model, prompt=rag_prompt)
    answer = chain.run({"context": context, "question": state["question"]})
    return {"answer": answer, "chat_history": state["chat_history"] + [state["question"], answer]}

def run_fallback(state):
    model = get_llm(state["model_name"])
    chain = LLMChain(llm=model, prompt=fallback_prompt)
    answer = chain.run({"question": state["question"]})
    return {"answer": answer, "chat_history": state["chat_history"] + [state["question"], answer]}

# Define and compile graph
graph = StateGraph()
graph.add_node("retrieve", retrieve_context)
graph.add_conditional_edges("retrieve", route)
graph.add_node("rag_chain", run_rag)
graph.add_node("fallback_chain", run_fallback)
graph.add_edge("rag_chain", END)
graph.add_edge("fallback_chain", END)
graph.set_entry_point("retrieve")
compiled_graph = graph.compile()

# Session-aware chatbot interface
chatbot_graph = RunnableWithMessageHistory(
    compiled_graph,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Sample run
if __name__ == "__main__":
    session_id = "user_123"
    user_question = "What are the compliance rules for remote employees?"
    result = chatbot_graph.invoke(
        {"question": user_question, "model_name": "claude-3-sonnet"},
        config={"configurable": {"session_id": session_id}}
    )
    print("Answer:", result["answer"])

    if unanswered_log:
        print("\n🛑 Unanswered Queries Logged:")
        for q in unanswered_log:
            print(" -", q)
