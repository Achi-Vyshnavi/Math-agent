# Math Agent

AI-powered math solver with a dynamic knowledge base, LLM fallback, and web search integration. The system offers features:

Step-by-step solutions: Predefined KB for common math problems and semantic retrieval for similar questions.

Input/Output Guardrails: Validates questions and answers to ensure math-related queries and protects sensitive inputs.

Dynamic KB Updates: Human-in-the-loop feedback allows users to correct answers, updating the KB for future queries.

Web Search + MCP: For questions outside the KB, the agent uses Serper web search and a Math Context Processing module to extract relevant information.

LLM Fallback: If no KB or web info is available, the agent generates step-by-step solutions using HuggingFace LLM.

Interactive React Frontend: User-friendly interface to input questions, view answers, and submit feedback.

JEE Benchmark Support: Capability to evaluate performance on JEE-level math problems and track KB hits.

This repository includes all backend and frontend code, sample KB (`kb.json`), configuration files (`requirements.txt`), and React app files.  


Repository Contents:

Backend: main.py, math_agent.py, requirements.txt

Frontend: React app files (App.js, index.js, index.css, etc.)

Sample KB: kb.json

Tests & Config: setupTests.js, reportWebVitals.js

Other assets: logo.svg, integration.txt

# Backend
cd math_agent
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd math-agent-app
npm install
npm start
# Running the Project

## Backend
1. Navigate to backend folder:
   cd math_agent
Activate virtual environment and install requirements:
venv\Scripts\activate  # Windows
pip install -r requirements.txt
Start the FastAPI server:
uvicorn main:app --reload
Backend will run at http://localhost:8000.

Frontend
Navigate to frontend folder:
cd math-agent-app
Install dependencies:
npm install
Start the React app:
npm start
Frontend will run at http://localhost:3000.
