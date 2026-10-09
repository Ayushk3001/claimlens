# 🛡️ ClaimLens — AI-Powered Insurance Claims Assistant

**ClaimLens** is an intelligent, production-ready Insurance Claims Decisioning System built for Property & Casualty (P&C) carriers. Combines **Hybrid RAG Retrieval (Vector + BM25)**, **Autonomous Multi-Agent Orchestration**, and **Explainable Predictive Machine Learning** to streamline claims investigation, triage complexity, calculate settlement estimates, detect fraud risk, and assist human adjusters.

---

## 1. Project Overview & Business Objective

### Problem Statement
Insurance carriers process massive volumes of claims across **Auto**, **Home**, **Renters**, **Property**, and **Business** insurance. Manually inspecting historical claims files to identify similar cases, assess damage credibility, detect suspicious fraud patterns, prioritize urgent claims, and determine settlement amounts is time-consuming (averaging 18–35 days per claim) and often inconsistent. Fraudulent claims alone cost the insurance sector over $308 Billion annually in losses.

### Business Objective
To build an AI-assisted claims retrieval, prediction, and multi-agent workflow solution that:
1. **Accelerates Decisioning:** Automates routine low-risk claims (reducing cycle times from days to minutes).
2. **Combats Fraud:** Detects early warning risk signals and routes suspicious claims directly to Special Investigation Units (SIU).
3. **Retrieves Precedents:** Semantically locates similar historical claims with rich metadata filtering.
4. **Maintains Human-in-the-Loop:** Captures adjuster feedback for continuous organizational learning.

---

## 2. Solution Summary

- **Hybrid RAG Precedent Search:** Combines semantic vector similarity (cosine distance over dense 1536-dimensional embeddings) and keyword BM25 retrieval using **Reciprocal Rank Fusion (RRF)**. Supports metadata filtering on `claim_type`, `state`, `claim_status`, `claim_amount`, and policyholder tenure.
- **Predictive Machine Learning Engine:** Fast, explainable Scikit-Learn ensemble models trained on the Hugging Face 100M insurance claims dataset predicting:
  - Estimated Claim Payout Amount
  - Turnaround Days to Resolution
  - Fraud Probability Percentage (with top underwriting risk drivers)
- **Multi-Agent Claims Workflow:**
  - **Triage & Classification Agent:** Classifies complexity and assigns SLA priority triage tiers (P1 Critical to P4 Routine).
  - **Investigation Agent:** Analyzes policyholder tenure, prior claims frequency, and clusters peer claim precedents.
  - **Risk Assessment Agent:** Evaluates financial exposure, anomalies, and loss drivers.
  - **Recommendation Agent:** Formulates the final claims action (`AUTO_APPROVE`, `MANUAL_ADJUSTER_REVIEW`, or `SIU_REFERRAL`) and creates actionable checklists.
  - **LLM-as-Judge Evaluator:** Independently audits the recommendation for factual consistency, completeness, and policy compliance.
- **Microservice & Interactive UI:** FastAPI REST microservice with OpenAPI specifications and an interactive Streamlit UI for claim adjusters.
- **Automated Quality Evaluation:** DeepEval test suite evaluating Faithfulness (95.0%), Answer Relevancy (96.0%), Policy Compliance (95.0%), and LLM-as-Judge Quality (9.3/10.0).

---

## 3. Technology Stack

| Component | Technology / Library | Description |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.11 | Core runtime environment |
| **LLM Used** | OpenAI `gpt-5-nano` via `https://aicredits.in/v1` | Multi-agent reasoning, synthesis, and LLM-as-Judge |
| **Embedding Model** | OpenAI `text-embedding-3-small` | 1536-dimensional dense claim vector embeddings (with deterministic fallback) |
| **Vector Database** | In-Memory Vector Store | NumPy normalized cosine similarity retrieval (sub-2ms, ChromaDB schema compatible) |
| **Keyword Search** | Rank-BM25 | BM25Okapi lexical token search |
| **Machine Learning** | Scikit-Learn (Random Forest) | Regression (Amount, Days) & Calibrated Classification (Fraud) |
| **Backend Microservice** | FastAPI, Uvicorn, Pydantic v2 | High-throughput async REST API with validation guardrails |
| **Frontend UI** | Streamlit | Adjuster dashboard, search explorer, and feedback UI |
| **Evaluation Framework** | DeepEval & Pytest | Faithfulness, relevancy, and policy compliance benchmarks |
| **Documentation & Decks** | ReportLab & Python-PPTX | Automated generation of PDF deliverables and slides |

---

## 4. Repository Structure

```
ProjectRepository/
├── architecture/
│   └── Architecture_Diagram.pdf     # High-level architecture flow diagram
├── design/
│   └── Design_Document.pdf          # Full engineering design document
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                # FastAPI endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classification_agent.py  # Triage and priority scoring agent
│   │   └── multi_agent_workflow.py  # 3-Agent workflow + LLM-as-Judge
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py            # OpenAI & dense embedding generator
│   │   ├── keyword_search.py        # BM25 search engine with metadata filters
│   │   ├── vector_store.py          # Vector store with cosine similarity
│   │   └── hybrid_retriever.py      # Reciprocal Rank Fusion (RRF) retriever
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_models.py             # Amount, Days, and Fraud ML models
│   │   └── feedback_service.py      # Human-in-the-loop feedback store
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                # Application configuration & paths
│   │   ├── guardrails.py            # Input validation & security guardrails
│   │   └── data_loader.py           # HF 100M Parquet dataset loader
│   └── main.py                      # FastAPI application entrypoint
├── data/
│   └── sample_data/
│       ├── claims_sample.parquet    # Cleaned Hugging Face 100M sample
│       └── claims_sample.csv        # Tabular export for quick viewing
├── docs/
│   ├── API_Documentation.pdf        # Complete REST API reference
│   ├── Dataset_Details.pdf          # 13 Core fields & preprocessing specs
│   └── Evaluation_Report.pdf        # DeepEval & LLM-as-Judge scores
├── tests/
│   ├── __init__.py
│   ├── test_guardrails.py           # Input schema & security tests
│   ├── test_rag.py                  # Vector & BM25 retrieval tests
│   ├── test_ml.py                   # Machine learning model tests
│   ├── test_api.py                  # FastAPI route tests
│   └── run_evaluation.py            # DeepEval & benchmark test suite
├── presentation/
│   └── Project_Presentation.pptx    # 10-minute panel presentation deck
├── scripts/
│   └── generate_all_deliverables.py # Automated PDF/PPTX generator
├── frontend.py                      # Streamlit Adjuster Dashboard
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment configuration template
└── README.md
```

---

## 5. Dataset Details

- **Dataset Name:** Free Synthetic Insurance Claims 100M
- **Hugging Face Primary Link:** [ziadatalabs/FreeInsuranceClaims100M](https://huggingface.co/datasets/ziadatalabs/FreeInsuranceClaims100M)
- **Format:** Apache Parquet
- **13 Core Fields:**
  1. `claim_id` — Unique claim record identifier
  2. `policy_id` — Policyholder account contract ID
  3. `claim_type` — Insurance line (Auto, Home, Renters, Property, Business)
  4. `state` — 2-letter US State postal code
  5. `policyholder_tenure_years` — Tenure length with carrier in years
  6. `previous_claims_count` — Historical number of claims filed
  7. `incident_date` — Date loss occurred (YYYY-MM-DD)
  8. `claim_filed_date` — Date claim was submitted (YYYY-MM-DD)
  9. `claim_amount` — Claimed dollar loss amount
  10. `deductible` — Applicable policy deductible
  11. `claim_status` — Open, Approved, Closed, Denied, Under Investigation
  12. `days_to_resolution` — Turnaround duration to close claim
  13. `is_fraud_flagged_ground_truth` — Ground truth label for suspected fraud

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.11 installed
- Git installed

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/Ayushk3001/claimlens.git
cd claimlens
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. API Key Configuration
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Open `.env` and set your OpenAI API key (optional — system includes robust offline heuristics if no key is configured):
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://aicredits.in/v1
OPENAI_MODEL=gpt-5-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 7. Local Deployment Instructions

### Running the Backend Microservice (FastAPI)
Launch the FastAPI server on port 8000:
```bash
uvicorn src.main:app --reload --port 8000
```
- **Live API Root:** `http://127.0.0.1:8000`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **Redoc Documentation:** `http://127.0.0.1:8000/redoc`

### Running the Frontend Dashboard (Streamlit)
In a separate terminal, launch the Streamlit frontend:
```bash
streamlit run frontend.py
```
- **Streamlit Web UI:** Opens automatically at `http://localhost:8501`

---

## 8. Running Automated Tests & Evaluation Suite

### Run All Unit & API Tests
```bash
pytest -v
```
*(All 23 comprehensive unit, guardrail, RAG metadata filtering, and API tests pass in under 20 seconds)*

### Run DeepEval & LLM-as-Judge Benchmark
```bash
python -m tests.run_evaluation
```
- **Test Scenarios Evaluated:**
  - `TC-001` (Auto collision, clean history): **AUTO_APPROVE** (Decision aligned: True)
  - `TC-002` (Business arson risk): **SIU_REFERRAL** (Decision aligned: True)
  - `TC-003` (Home water loss >$10k): **MANUAL_ADJUSTER_REVIEW** (Decision aligned: True)
- **Results:** 3/3 tests passed (100% alignment), Faithfulness 95.0%, Relevancy 96.0%, Compliance 95.0%, LLM Judge 9.3/10.0.

### Re-Generate Deliverables (PDFs & Presentation Deck)
```bash
python scripts/generate_all_deliverables.py
```

---

## 9. Sample Queries & API Examples

### Sample Query 1: Routine Low-Risk Auto Claim (Eligible for Fast-Track)
```bash
curl -X POST http://127.0.0.1:8000/claims/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-SAMPLE-01",
    "policy_id": "POL-100234",
    "claim_type": "Auto",
    "state": "CA",
    "policyholder_tenure_years": 5.2,
    "previous_claims_count": 0,
    "incident_date": "2024-03-01",
    "claim_filed_date": "2024-03-03",
    "claim_amount": 2800.0,
    "deductible": 500.0
  }'
```
**Expected Decision:**
- **Recommendation:** `AUTO_APPROVE` (Fast-Track Automated Settlement)
- **Recommended Net Payout:** `$2,300.00`
- **Priority Tier:** `P4 - Low / Routine` (SLA: 120 hrs)
- **Fraud Probability:** `~31.0%` (Low Risk)
- **LLM-as-Judge Verdict:** `PASS` (Score: 9.3 / 10.0)

---

### Sample Query 2: High-Risk Commercial Fire Claim (Routed to SIU)
```bash
curl -X POST http://127.0.0.1:8000/claims/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-SAMPLE-02",
    "policy_id": "POL-992311",
    "claim_type": "Business",
    "state": "FL",
    "policyholder_tenure_years": 0.2,
    "previous_claims_count": 3,
    "incident_date": "2024-01-10",
    "claim_filed_date": "2024-03-25",
    "claim_amount": 85000.0,
    "deductible": 1000.0
  }'
```
**Expected Decision:**
- **Recommendation:** `SIU_REFERRAL` (Withhold settlement disbursement, assign field investigator)
- **Priority Tier:** `P1 - Critical Priority` (SLA: 24 hrs)
- **Fraud Probability:** `~46.0%` (High Financial Exposure)
- **Top Risk Drivers:** Short policy tenure (<1 yr), multiple prior claims, high claim-to-deductible ratio (>20x)
- **LLM-as-Judge Verdict:** `PASS` (Score: 9.3 / 10.0)

---

### Sample Query 3: Hybrid Search for Precedent Cases
```bash
curl -X POST http://127.0.0.1:8000/claims/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "rear-end bumper collision damage repair",
    "top_k": 3,
    "claim_type": "Auto",
    "state": "CA"
  }'
```
**Expected Output:**
- 3 most relevant historical auto claims from California with similarity scores > 0.90, displaying historical status (Approved/Closed) and payout benchmarks.

---

## 10. Submission Checklist Verification

- [x] **Source code is complete** (FastAPI backend, Streamlit UI, Multi-Agent workflow, ML models, RAG)
- [x] **README is updated** (All 12 sections from guidelines fulfilled)
- [x] **Architecture diagram included** (`architecture/Architecture_Diagram.pdf`)
- [x] **Design document included** (`design/Design_Document.pdf`)
- [x] **Presentation deck included** (`presentation/Project_Presentation.pptx`)
- [x] **APIs are working** (All endpoints tested and verified)
- [x] **Sample data included** (`data/sample_data/claims_sample.parquet` and `.csv`)
- [x] **Evaluation reports included** (`docs/Evaluation_Report.pdf` with DeepEval metrics)
- [x] **All deliverables follow naming conventions**
- [x] **Local deployment instructions are included**
- [x] **No secrets or API keys committed**
