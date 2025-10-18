# Math Agent

AI-powered math solver with a dynamic knowledge base, LLM fallback, and web search integration. The system offers:

- **Step-by-step solutions**: Predefined KB for common math problems and semantic retrieval for similar questions.
- **Input/Output Guardrails**: Validates questions and answers to ensure math-related queries and protects sensitive inputs.
- **Dynamic KB Updates**: Human-in-the-loop feedback allows users to correct answers, which updates the KB for future queries.
- **Web Search + MCP**: For questions outside the KB, the agent uses Serper web search and a Math Context Processing module to extract relevant information for reasoning.
- **LLM Fallback**: If no KB or web info is available, the agent generates step-by-step solutions using HuggingFace LLM.
- **Interactive React Frontend**: User-friendly interface to input questions, view answers, and submit feedback.
- **JEE Benchmark Support**: Capability to evaluate performance on JEE-level math problems and track KB hits.

This repository includes all backend and frontend code, sample KB (`kb.json`), configuration files (`requirements.txt`), and React app files.  

