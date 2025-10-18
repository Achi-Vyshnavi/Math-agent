# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from qdrant_client import QdrantClient
import requests
import re
from dspy.agent import FeedbackAgent

app = FastAPI(title="Full Assignment Math Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------
# 1️⃣ Initialize DSPy Feedback Agent
# ----------------------------
feedback_agent = FeedbackAgent()

# ----------------------------
# 2️⃣ Console KB
# ----------------------------
console_kb = [
    {"question": "Solve x^2 + 5x + 6 = 0",
     "answer": "Step1: Factor (x+2)(x+3)=0; Step2: x=-2, x=-3"},
    {"question": "Integrate x^2 dx",
     "answer": "Step1: Increase power by 1 → x^3; Step2: Divide by new power → x^3/3 + C"}
]

# Convert console KB to Document objects
kb_documents = [Document(text=f"Q: {item['question']}\nA: {item['answer']}") for item in console_kb]

# ----------------------------
# 3️⃣ Optional: Load local KB documents
# ----------------------------
# from llama_index.core import SimpleDirectoryReader
# documents = SimpleDirectoryReader("math_kb_docs").load_data()
documents = kb_documents  # for now using console KB

# ----------------------------
# 4️⃣ Setup Qdrant Vector Store
# ----------------------------
client = QdrantClient(":memory:")  # in-memory dev
vector_store = QdrantVectorStore(client=client, collection_name="math_agent")

# ----------------------------
# 5️⃣ Embedding model
# ----------------------------
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# 6️⃣ LLM
# ----------------------------
llm = HuggingFaceLLM(
    model_name="EleutherAI/gpt-neo-125M",
    tokenizer_name="EleutherAI/gpt-neo-125M",
    max_new_tokens=150,
    generate_kwargs={"temperature": 0.7},
    model_kwargs={
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "use_safetensors": False
    }
)

# ----------------------------
# 7️⃣ Build VectorStoreIndex
# ----------------------------
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)

# ----------------------------
# 8️⃣ Input guardrail
# ----------------------------
def is_math_question(question: str) -> bool:
    math_keywords = ["solve", "integrate", "derivative", "probability", "find", "equation", "calculate"]
    return any(k.lower() in question.lower() for k in math_keywords)

# ----------------------------
# 9️⃣ Serper Web Search
# ----------------------------
def serper_search(query: str) -> str:
    API_KEY = "44a8345d40d31ecf430a6ba705a042448290d526"  # Replace with actual key
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": API_KEY}
    data = {"q": f"{query} step by step solution math"}
    try:
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()
        result = r.json()
        if "organic" in result and len(result["organic"]) > 0:
            snippet = result["organic"][0]["snippet"]
            return snippet
        else:
            return ""
    except Exception as e:
        return ""

# ----------------------------
# 10️⃣ MCP Processing (placeholder)
# ----------------------------
def mcp_process(context_snippet: str) -> str:
    """
    Placeholder for MCP server integration.
    Cleans snippet and returns structured context.
    """
    cleaned_context = re.sub(r"\n+", "\n", context_snippet).strip()
    return cleaned_context

# ----------------------------
# 11️⃣ Feedback model
# ----------------------------
class FeedbackModel(BaseModel):
    question: str
    proposed_answer: str
    correct_answer: str

# ----------------------------
# 12️⃣ API Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "Math Agent API running. Use /solve?question=... to get answers."}

@app.get("/solve")
def solve(question: str):
    # Input guardrail
    if not is_math_question(question):
        raise HTTPException(status_code=400, detail="Only math questions allowed.")

    # 1️⃣ Query vector KB
    kb_response = query_engine.query(question)
    answer = str(kb_response).strip()

    # 2️⃣ If no KB answer, fallback to Serper + MCP
    if not answer or "No relevant documents" in answer or answer.lower() == "none":
        snippet = serper_search(question)
        if snippet:
            context = mcp_process(snippet)
            prompt = f"Using this context, solve step by step:\n{context}"
            answer_obj = llm.generate(prompt)
            answer = str(answer_obj).strip()
        else:
            answer = "Sorry, could not find solution via KB or web search."

    # Output guardrail
    if not answer:
        answer = "Sorry, could not find solution."

    return {"answer": answer}

@app.post("/feedback")
def feedback(feedback: FeedbackModel):
    """
    DSPy Human-in-the-loop feedback.
    Stores feedback and updates vector KB.
    """
    # 1️⃣ Submit structured feedback to DSPy
    feedback_agent.submit_feedback(
        question=feedback.question,
        proposed_answer=feedback.proposed_answer,
        human_corrected_answer=feedback.correct_answer
    )

    # 2️⃣ Update vector KB
    new_doc = Document(text=f"Q: {feedback.question}\nA: {feedback.correct_answer}")
    index.insert_documents([new_doc])

    return {"status": "feedback recorded via DSPy"}

# ----------------------------
# 13️⃣ JEE Bench Benchmark Script (bonus)
# ----------------------------
@app.post("/jee_bench_eval")
def jee_bench_eval(jee_questions: list):
    """
    Input: list of {"question": str, "answer": str}
    Returns: evaluation metrics and agent answers
    """
    results = []
    for item in jee_questions:
        question = item.get("question")
        gold_answer = item.get("answer", "")
        # Call solve internally
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
        "kb_hit_percentage": kb_hits/total*100 if total>0 else 0
    }

    return {"metrics": metrics, "results": results}
