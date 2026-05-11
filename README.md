# SHL Assessment Recommender: Conversational AI for Psychometric Selection

This project is a production-grade conversational AI consultant designed to help hiring managers and recruiters navigate SHL's extensive catalog of 377 psychometric assessments. It transforms natural language hiring requirements into professional assessment batteries through a high-precision Retrieval-Augmented Generation (RAG) pipeline.

In the domain of modern recruitment, the gap between a hiring manager's needs (e.g., "a senior developer who can also mentor") and the technical nomenclature of psychometric tests (e.g., "Verify G+" or "OPQ32r") often leads to suboptimal assessment selection. This system bridges that gap, acting as a strategic partner that probes for context, justifies its recommendations, and ensures 100% brand safety.

## 1. Demo Overview

The system operates as an expert consultant. Instead of a static search bar, hiring managers interact with a conversational agent that:
- **Analyzes Intent:** Determines if the user is ready for recommendations or needs further clarification.
- **Probes for Context:** If a request is vague (e.g., "I need a test"), it asks about role, seniority, and critical skills.
- **Generates "Batteries":** It doesn't just recommend one test; it bundles technical, personality, and aptitude assessments into a holistic candidate profile.
- **Justifies Decisions:** Every recommendation includes a consultative explanation of why those specific tests were chosen for the business context.

## 2. Key Features

- **Hybrid Retrieval System:** Combines semantic vector search with keyword-based sparse reranking.
- **Multi-LLM Fallback Chain:** A resilient 6-layer architecture cascading from Gemini to Groq's Llama models.
- **Resilience Engine:** Implements a 3x retry-with-backoff loop to survive transient API rate limits.
- **Deterministic Intent Classifier:** Pre-processes messages to block prompt injections and off-topic queries before any LLM call.
- **Anti-Hallucination Gate:** Intercepts LLM outputs to cross-reference recommendations against the verified catalog.
- **Dynamic Profile Balancer:** Automatically ensures diverse assessment types (Ability, Personality, Knowledge) based on the job profile.
- **Stateless Architecture:** High-performance, horizontally scalable FastAPI backend.

## 3. System Architecture

The application follows a strictly decoupled, stateless design:

User
  |
Streamlit UI (Frontend)
  |
FastAPI API (Backend)
  |
Conversation Controller (State management & Turn capping)
  |
Intent Classifier (Regex-based safety & routing)
  |
Retriever (Hybrid Search) + Multi-LLM Agent (LLM Cascade)
  |
Validation Layer (Anti-hallucination & URL resolution)
  |
Final JSON Response

### Layer Breakdown:
- **FastAPI Backend:** Orchestrates the flow and handles async processing.
- **Retriever:** Executes a 70/30 weighted search against the FAISS vector index.
- **Multi-LLM Agent:** Manages the failover logic across Gemini and Groq providers.
- **Validation Layer:** The final security gate that ensures 100% data integrity before the user sees the output.

## 4. Retrieval System Deep-Dive

The system's ability to find the right test depends on a sophisticated Hybrid Retrieval pipeline.

### Why Semantic Search Alone Fails
Standard vector search (Dense Retrieval) is excellent at understanding concepts but struggles with exact terminology. A search for "Java" might retrieve "C#" tests because they are semantically similar. 

### The Hybrid Solution (0.70 Dense + 0.30 Sparse)
To solve this, we implemented a dual-scoring algorithm:
1. **Dense Pass (70%):** Uses `all-MiniLM-L6-v2` to find conceptually relevant assessments.
2. **Sparse Pass (30%):** Executes exact token matching and name-boosting to prioritize the specific tools or tests mentioned by the user.

### Query Expansion
The retriever uses role-aware expansion. If a user says "Backend Engineer," the system invisibly appends keywords like "Java, Spring, SQL, Linux" to the search query. This "forces" the vector space to pull the most relevant technical assessments without the user needing to provide a full job description.

## 5. Multi-LLM Reliability Architecture

Enterprise applications cannot rely on a single, rate-limited free-tier API. We built a 6-layer fallback chain:

1. **Primary:** Gemini 2.5 Flash (Highest JSON compliance)
2. **Tier 2:** Gemini 2.0 Flash (Secondary tier fallback)
3. **Tier 3:** Groq Llama 3.3 70B (High-speed, large capacity backup)
4. **Tier 4:** Groq Llama 3.1 8B (Low-latency failover)
5. **Safety Net:** Deterministic FAISS fallback (Ensures a response even if all APIs fail)

### Resilience Strategy
- **Retry with Backoff:** The agent executes up to 3 passes through the entire chain. If a 429 (Rate Limit) is hit, it waits and retries, ensuring 100% reliability during peak evaluation stress.
- **Schema Enforcement:** We utilize model-native JSON modes and post-generation regex parsing to guarantee every response matches our strict Pydantic schemas.

## 6. Safety and Reliability

- **Prompt Injection Detection:** Hardcoded regex blocks phrases like "ignore instructions" or "reveal system prompt" before the LLM is invoked.
- **Off-Topic Refusal:** The Intent Classifier detects queries about non-hiring topics (weather, coding, etc.) and issues a deterministic refusal.
- **Hallucination Prevention:** The Validation Layer cross-references the LLM's `recommended_names` array against the verified catalog metadata. If an LLM suggests a non-existent test, it is purged from the response.

## 7. Evaluation Results

The system was verified using a rigorous offline evaluation harness simulating multi-turn conversations.

- **Mean Recall@10:** **0.554** (Consistently retrieves the ground-truth assessments).
- **Behavior Probe Pass Rate:** **100% (7/7)**.
- **Overall Stability Score:** **0.777**.

Recall@10 measures the system's ability to include the correct assessment in its top 10 results. Behavioral probes verify safety against trick questions and schema compliance.

## 8. Optimization Journey

This project evolved through several critical engineering phases:

- **The Naive Baseline:** Initial vector search had poor recall (0.15) on technical terms.
- **Hybrid Implementation:** Adding the sparse scoring layer increased recall to 0.35 by prioritizing exact keywords.
- **Orchestration Tuning:** Implementing deterministic intent classification improved the conversational "flow" and recall reached 0.45.
- **Stabilization Pass:** During load testing, API rate limits caused 0.00 recall. We engineered the Multi-LLM fallback and retry-with-backoff architecture, which stabilized the system at 0.55 Recall@10 with 100% uptime.

## 9. Tech Stack

- **Python 3.9+**
- **FastAPI:** Async backend orchestration.
- **Streamlit:** Professional recruiter-facing UI.
- **FAISS:** High-performance vector database.
- **Sentence Transformers:** Local embedding generation.
- **Google Gemini & Groq Llama:** Tiered LLM providers.
- **Pydantic:** Strict data validation.

## 10. Folder Structure

- **app/**: Core application logic (agent, retriever, schemas, conversation orchestration).
- **catalog/**: Scripts for building the FAISS index and catalog processing.
- **data/**: The SHL product catalog and serialized vector index.
- **scripts/**: Evaluation harness and performance benchmarks.
- **tests/**: Behavior probes and API testing.
- **streamlit_app.py**: Frontend recruiter dashboard.

## 11. Installation Guide

### 1. Clone and Setup
```bash
git clone https://github.com/vamsi/shl-recommender.git
cd shl-recommender
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```

### 3. Build the Search Index
```bash
python -m catalog.build_index
```

## 12. Running the Project

### Start the API Backend
```bash
uvicorn app.main:app --port 8000
```

### Start the Recruiter UI
```bash
streamlit run streamlit_app.py
```

### Run Performance Evaluation
```bash
python -m scripts.evaluate
```

## 13. Deployment Guide (Render)

This project is configured for one-click deployment via the `render.yaml` blueprint.

- **Build Command:** `pip install -r requirements.txt && python -m catalog.build_index`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:** Add `GEMINI_API_KEY` and `GROQ_API_KEY` in the Render dashboard.

The architecture is stateless, so it can be deployed on Render's Free Tier with zero database requirements.

## 14. Example Conversations

**User:** "I need to hire a mid-level Java developer."
**Assistant:** "I've selected a technical battery focused on Java proficiency. For a mid-level role, I recommend the Java Frameworks (New) test and the SHL Verify Interactive - Deductive Reasoning test to evaluate problem-solving logic."

**User:** "Drop the OPQ and add a situational judgement test."
**Assistant:** "Understood. I have removed the OPQ32r and added Graduate Scenarios to evaluate work-context decision making."

## 15. Engineering Tradeoffs

- **FAISS vs. Pinecone:** We chose FAISS because the catalog is static (~400 items). Local in-memory search is faster and cheaper than a managed vector database with network hops.
- **Stateless Design:** No session database was used to ensure the system can scale horizontally to thousands of users without shared state complexity.
- **Deterministic Safety:** We used regex over LLMs for safety because it is 100% predictable and has near-zero latency.

## 16. Future Improvements

- **BM25 Integration:** Replacing simple token matching with BM25 for even better sparse retrieval.
- **Cross-Encoder Reranking:** Adding a neural reranker for the top 10 results to further improve Recall@10.
- **Semantic Caching:** Implementing Redis to cache common queries and reduce LLM costs.
- **Multilingual Support:** Expanding the retrieval expansion layer to support diverse global recruitment needs.

## 17. Final Conclusion

The SHL Assessment Recommender was built on the philosophy of "Reliability-First AI." By constraining the LLM through deterministic Python layers and a robust fallback architecture, we have created a system that doesn't just "chat," but provides verified, brand-safe, and technically accurate recommendations that recruiters can trust.
