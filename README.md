# Agentic ML Model & Dataset Evaluation Platform

An intelligent, autonomous evaluation and diagnostic workspace for Machine Learning models and tabular datasets. Built with a FastAPI backend, an agentic LLM orchestration engine (Groq API), deterministic security workflows, and a relational SQLite architecture for authoritative state tracking.

---

## Key Features

- **Autonomous Agent Workflows**: Executes model inspection, baseline performance evaluations, data quality audits, ART-based adversarial robustness tests, targeted dataset transformations, and two multi-step security workflows (comparative sweep and cumulative hardening loop).
- **Dual-Memory System**: Active session context + persistent, searchable catalog of uploaded models/datasets and experiment logs via ChromaDB.
- **Versioned Dataset Lineage**: Transformations (SMOTE, imputation, outlier handling, scaling, noise injection) create immutable version directories (`v001..vN`) with full provenance.
- **Multi-Step Robustness**: Comparative attack/defense sweep and cumulative adversarial hardening loop with per-step/round metrics persisted as chartable series.
- **Deterministic Intent Routing**: Routes requests into targeted workflows or contextual/comparative analyses with fail-fast validation and compatibility checks.
- **Structured Evaluation Reports**: Generates detailed diagnostic reports comparing baseline and post-transformation/post-hardening performance with reproducible metadata.
- **Web UI with Live Charts**: Interactive dashboard with multi-step chart views (sweep: grouped bars; loop: points for mixed threats) backed by persisted series data.

---

## Project Structure

```text
AGENTIC-ML-EVALUATION-PLATFORM/
├── app/
│   ├── api/                  # FastAPI router endpoints
│   │   ├── chat.py           # Intent routing, chat, agent execution bridge
│   │   ├── datasets.py       # Dataset upload, selection, transformations
│   │   ├── models.py         # Model upload & selection
│   │   ├── presets.py        # Benchmark presets
│   │   ├── projects.py       # Project state management
│   │   ├── evaluations.py    # Metric calculation routes
│   │   ├── experiments.py    # Experiment tracking routes
│   │   └── reports.py        # Report generation routes
│   ├── agent/                # Agent planner, state machine, tools, security aliases
│   ├── core/                 # Storage manager, SQLite handlers, config
│   ├── datasets/             # Data loading, versioning, validation
│   ├── evaluation/           # Robustness (ART), metrics, noise benchmarking
│   ├── llm/                  # Qwen client integration
│   ├── memory/               # ChromaDB vector store
│   ├── models/               # Adapters, loader, adversarial trainer
│   ├── reports/              # Markdown report generator
│   ├── schemas/              # Pydantic request/response models
│   ├── services/             # Shared service helpers
│   ├── transformations/      # Data cleaning, balancing, scaling, noise
│   └── main.py               # FastAPI application entry point (python -m app.main)
├── frontend/                 # Web UI (index.html, style.css, app.js)
├── storage/                  # SQLite DB, models, dataset versions, reports, Chroma
├── tests/                    # Pytest suite (conftest.py isolates storage)
└── requirements.txt
```

---

## Prerequisites

- **Python**: `3.10` or higher
- **Groq API key** (optional, for LLM features): the platform routes prompts to an LLM through Groq's OpenAI-compatible endpoint (`app/llm/qwen_client.py`). The model is selected by `GROQ_MODEL`. Without a key the deterministic Python workflows still run; LLM-backed routing/synthesis falls back to heuristics.

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/AGENTIC-ML-EVALUATION-PLATFORM.git
   cd AGENTIC-ML-EVALUATION-PLATFORM
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration (optional)**
   ```env
   APP_NAME="Agentic ML Evaluation Platform"
   APP_ENV="development"
   DEBUG=true
   HOST="0.0.0.0"
   PORT=8000
   GROQ_API_KEY=your_groq_api_key
   GROQ_BASE_URL="https://api.groq.com/openai/v1"
   GROQ_MODEL="openai/gpt-oss-120b"
   MAX_AGENT_ITERATIONS=12
   STORAGE_DIR="storage"
   CHROMA_PERSIST_DIR="storage/chromadb"
   ```

---

## Running the Application

1. **Launch the FastAPI Server**
   ```bash
   python -m app.main
   ```

2. **Access the Web Dashboard**
   ```text
   http://localhost:8000
   ```
   LLM features activate automatically when `GROQ_API_KEY` is set in `.env`.

---

## Testing

```bash
python -m pytest tests/ -q
```

---

## Notes

- ART-based adversarial workflows (evasion attacks: FGSM/PGD/CW/boundary/HopSkipJump/ZOO; defenses/preprocessors) with framework-aware validation.
- Multi-step: comparative sweep and cumulative hardening loop (up to 5 steps/rounds), persisted with series for UI charts.
- Deterministic Python core for numeric correctness; LLM used only for intent/synthesis.