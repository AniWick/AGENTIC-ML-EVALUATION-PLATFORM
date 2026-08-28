# Agentic ML Model & Dataset Evaluation Platform

An autonomous, closed-loop machine learning evaluation and data optimization platform.

---

## 🌟 Key Features

1. **Autonomous Closed-Loop Agent**:
   - Understands user goals and inspects model architectures and datasets.
   - Diagnoses data quality issues (class imbalance, missing values, outliers, high cardinality).
   - Dynamically selects and executes deterministic tools.
   - Applies versioned, immutable dataset transformations (SMOTE, imputation, scaling, outlier clipping).
   - Benchmarks noise robustness with Gaussian perturbation tests.
   - Programmatically compares before/after metrics and compiles full diagnostic reports.

2. **Immutable Dataset Lineage**:
   - Original datasets are preserved and never overwritten.
   - Version history (`v001`, `v002`, ...) tracks exact operations, parameters, and parent lineage.

3. **Multi-Framework Model Support**:
   - **Scikit-learn** (`.pkl`, `.joblib`)
   - **PyTorch** (`.pt`, `.pth`, TorchScript)
   - **ONNX Runtime** (`.onnx`)

4. **Deterministic Evaluation Engine**:
   - Performance metrics: Accuracy, Macro F1, Minority Class F1, Precision, Recall, Confusion Matrix, ROC-AUC, RMSE, MAE, R².
   - All numerical computations are performed deterministically by Python/Scikit-learn; LLMs are used strictly for diagnostic reasoning and natural language synthesis.

5. **Semantic Project Memory (ChromaDB)**:
   - Persists experiment records, deltas, and diagnostic observations.
   - Allows users to query past experiments conversationally (*"Did balancing help?"*, *"What was the baseline accuracy?"*).

6. **Interactive Web Dashboard & REST API**:
   - Built with FastAPI, SQLite, ChromaDB, and interactive Bootstrap dashboard.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy or modify `.env`:
```env
OLLAMA_BASE_URL="http://localhost:11434"
LLM_MODEL="llama3.1"
FALLBACK_LLM_MODEL="phi3"
```

### 3. Run the Backend & Dashboard
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
👉 **`http://localhost:8000/`**

### 4. Run Automated Tests
```bash
python -m pytest -v
```

---

## 📂 Project Structure

```
Project1_CDAC/
├── app/
│   ├── core/           # Config, database, storage layout
│   ├── models/         # Model adapters (Sklearn, PyTorch, ONNX)
│   ├── datasets/       # Dataset inspection, profiling, versioning
│   ├── evaluation/     # Deterministic metrics & robustness
│   ├── transformations/# Data cleaners, scalers, SMOTE balancer, noise
│   ├── memory/         # ChromaDB semantic retrieval
│   ├── agent/          # AgentState, ToolRegistry, Planner, AgentRunner
│   ├── llm/            # Ollama client & prompt templates
│   ├── reports/        # Multi-format report generator
│   ├── schemas/        # Pydantic models
│   ├── api/            # FastAPI routers
│   └── main.py         # App entrypoint
├── frontend/           # Web dashboard (HTML/CSS/JS)
├── storage/            # Models, datasets (original + versions), reports
├── tests/              # Unit & E2E integration test suite
├── requirements.txt
├── .env
└── README.md
```

