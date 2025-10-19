# main.py 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from qdrant_client import QdrantClient
import requests
import re
import logging
import os

try:
    from dspy.agent import FeedbackAgent
    feedback_agent = FeedbackAgent()
    DSPY_AVAILABLE = True
except ModuleNotFoundError:
    logging.warning("⚠️ DSPy not installed. Feedback endpoint disabled.")
    feedback_agent = None
    DSPY_AVAILABLE = False

app = FastAPI(title="Full Assignment Math Agent")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Console KB
# ----------------------------
console_kb = [
    {"question": "Solve x^2 + 5x + 6 = 0",
     "answer": "Step1: Factor (x+2)(x+3)=0; Step2: x=-2, x=-3"},
    {"question": "Integrate x^2 dx",
     "answer": "Step1: Increase power by 1 → x^3; Step2: Divide by new power → x^3/3 + C"}
]
kb_documents = [Document(text=f"Q: {item['question']}\nA: {item['answer']}") for item in console_kb]

# ----------------------------
# Qdrant Vector Store
# ----------------------------
client = QdrantClient(":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="math_agent")

# ----------------------------
# Embedding + LLM
# ----------------------------
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

os.makedirs("./offload", exist_ok=True)

llm = HuggingFaceLLM(
    model_name="EleutherAI/gpt-neo-125M",
    tokenizer_name="EleutherAI/gpt-neo-125M",
    max_new_tokens=150,
    generate_kwargs={"temperature": 0.7},
    device_map="auto",  
    model_kwargs={
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "use_safetensors": False,
        "offload_folder": "./offload"
    }
)

# ----------------------------
# VectorStoreIndex
# ----------------------------
index = VectorStoreIndex.from_documents(
    kb_documents,
    vector_store=vector_store,
    embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)

# ----------------------------
# Helper Functions
# ----------------------------
def is_math_question(question: str) -> bool:
    math_keywords = ["solve", "integrate", "derivative", "probability", "find", "equation", "calculate"]
    return any(k.lower() in question.lower() for k in math_keywords)

def validate_answer(answer: str) -> str:
    if answer and any(k in answer for k in ["Step", "=", "+", "-", "*", "/", "^"]):
        return answer
    return "Sorry, could not generate a valid math solution."

def serper_search(query: str) -> str:
    API_KEY = "ca58cb81f6d9676cde10d87468b4344b1b4006c1"
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": API_KEY}
    data = {"q": f"{query} step by step solution math"}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=5)
        r.raise_for_status()
        result = r.json()
        if "organic" in result and len(result["organic"]) > 0:
            snippet = result["organic"][0]["snippet"]
            return snippet
    except Exception as e:
        logging.warning(f"Serper search failed: {e}")
    return ""

def mcp_process(snippet: str) -> str:
    cleaned = re.sub(r"[\n]+", "\n", snippet).strip()
    math_lines = [line for line in cleaned.split("\n") if re.search(r"[0-9x+=\^*/]", line)]
    return "Context extracted from web: " + " ".join(math_lines)

# Feedback model 
class FeedbackModel(BaseModel):
    question: str
    corrected_answer: str

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "Math Agent API running. Use /solve?question=... to get answers."}

@app.get("/solve")
def solve(question: str):
    if not is_math_question(question):
        raise HTTPException(status_code=400, detail="Only math questions allowed.")

    answer = ""
    snippet = ""

    # Console KB
    for item in console_kb:
        if item["question"].strip().lower() == question.strip().lower():
            answer = item["answer"]

    # Vector KB
    if not answer:
        try:
            kb_response = query_engine.query(question)
            answer = str(kb_response).strip()
            if not answer or "No relevant documents" in answer.lower() or answer.lower() == "none":
                answer = ""
        except Exception as e:
            logging.warning(f"Vector KB query failed: {e}")
            answer = ""

    # Web Search + MCP
    if not answer:
        try:
            snippet = serper_search(question)
            if snippet:
                context = mcp_process(snippet)
                prompt = f"Use the following context to solve the math problem step by step:\n{context}\nProblem: {question}"
                answer_obj = llm.predict(prompt)
                answer = str(answer_obj).strip()
        except Exception as e:
            logging.warning(f"Web Search + MCP + LLM failed: {e}")
            answer = ""

    # Final LLM fallback
    if not answer:
        try:
            prompt = f"Solve this math problem step by step:\n{question}"
            answer_obj = llm.predict(prompt)
            answer = str(answer_obj).strip()
        except Exception as e:
            logging.warning(f"LLM final fallback failed: {e}")
            answer = "Sorry, could not generate a solution."

    if snippet:
        source = "Web"
    elif answer and any(item["question"].strip().lower() == question.strip().lower() for item in console_kb):
        source = "KB"
    else:
        source = "LLM"

    return {"answer": answer, "source": source}

@app.post("/feedback")
def feedback(feedback: FeedbackModel):
    if DSPY_AVAILABLE:
        feedback_agent.submit_feedback(
            question=feedback.question,
            proposed_answer="",
            human_corrected_answer=feedback.corrected_answer
        )

    new_doc = Document(text=f"Q: {feedback.question}\nA: {feedback.corrected_answer}")
    index.insert_documents([new_doc])

    return {"status": "Feedback recorded and KB updated"}

@app.post("/jee_bench_eval")
def jee_bench_eval(jee_questions: list):
    results = []
    for item in jee_questions:
        question = item.get("question")
        gold_answer = item.get("answer", "")
        agent_answer = solve(question)["answer"]
        results.append({
            "question": question,
            "agent_answer": agent_answer,
            "gold_answer": gold_answer,
            "kb_hit": "Step" in agent_answer
        })
    total = len(results)
    kb_hits = sum(1 for r in results if r["kb_hit"])
    metrics = {
        "total_questions": total,
        "kb_hits": kb_hits,
        "kb_hit_percentage": kb_hits / total * 100 if total > 0 else 0
    }
    return {"metrics": metrics, "results": results}






