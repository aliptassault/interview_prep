# from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.memory import ConversationBufferMemory
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
# from langchain.agents.agent_toolkits import Tool
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json 
import os
from langchain.tools import Tool
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
num_workers = multiprocessing.cpu_count()

# Load and split documents
doc_loader = PyPDFDirectoryLoader("./enterprise_docs", glob="**/*.pdf")
# Print documents present in the directory
print("Documents in directory:", os.listdir("enterprise_docs"), doc_loader)
documents = doc_loader.load()

# Check if documents are loaded successfully
if not documents:
    raise ValueError("No documents found in the specified directory. Please check the directory path and ensure it contains valid files.")

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split_docs = splitter.split_documents(documents)
# Parallel document splitting
def split_doc_batch(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def parallel_split_documents(documents, batch_size=100):
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        split_docs = list(executor.map(split_doc_batch, batches))
    return [doc for batch in split_docs for doc in batch]

# Use parallel splitting
split_docs = parallel_split_documents(documents)
print("Number of split documents:", len(split_docs))
# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded successfully.")
# Vector DB
shutil.rmtree("./chroma_langchain_db", ignore_errors=True)  # Deletes existing persisted data
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    collection_name="example_collection",
    persist_directory="./chroma_langchain_db"  # Where to save data locally, remove if not necessary
)
print("Chroma DB created successfully.")
retriever = vector_store.as_retriever()
print("Vector DB created successfully.")

# Chat Models (Local only)
# llm_models = {
#     "gemma2-9b-it-local": LlamaCpp(
#         model_path="./models/gemma-2b-it.Q4_K_M.gguf",
#         temperature=0.3,
#         max_tokens=512,
#         n_ctx=2048
#     ),
#     "mistral-local": LlamaCpp(
#         model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#         temperature=0.3,
#         max_tokens=512,
#         n_ctx=2048
#     ),
#     "deepseek-local": LlamaCpp(
#         model_path="./models/deepseek-llm-7b-instruct.Q4_K_M.gguf",
#         temperature=0.3,
#         max_tokens=512,
#         n_ctx=2048
#     ),
#     "claude3-local": LlamaCpp(  # Only if weights are available
#         model_path="./models/claude3-sonnet.Q4_K_M.gguf",
#         temperature=0.3,
#         max_tokens=512,
#         n_ctx=2048
#     )
# }
llm_models = {
    "gemma-7b": Llama.from_pretrained(
        repo_id="google/gemma-7b",
        filename="gemma-7b.gguf",
        n_gpu_layers=40,  # RTX 4080 has ample VRAM, increase GPU layers
        n_batch=1024,     # Increase batch size for better performance
        n_ctx=4096,       # Keep larger context window
        f16_kv=True,      # Enable half-precision for efficiency
        verbose=False,    # Reduce logging overhead
        n_threads=16,     # Utilize more CPU cores for preprocessing
        gpu_layers_max_thread_use=16,  # Max threads for GPU layers
        temperature=0.3
    )
}


# Select active model
def get_llm(model_name="gemma-7b"):
    return llm_models.get(model_name, llm_models["gemma-7b"])

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompts
rag_prompt = PromptTemplate.from_template(
    "Answer the question using the context below:\n\n{context}\n\nQuestion: {question}"
)

fallback_prompt = PromptTemplate.from_template(
    "Answer the question using your own knowledge:\n\nQuestion: {question}"
)
HF_TOKEN="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"
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
        "model_name": state.get("model_name", "gemma-7b")
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
    user_question = "who is harry?"
    result = chatbot_graph.invoke(
        {"question": user_question, "model_name": "gemma-7b"},
        config={"configurable": {"session_id": session_id}}
    )
    print("Answer:", result["answer"])

    if unanswered_log:
        print("\n🛑 Unanswered Queries Logged:")
        for q in unanswered_log:
            print(" -", q)
