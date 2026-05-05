# 🤖 Autonomous Data Analyst Agent

A **production-grade multi-agent AI system** that behaves like a mid-level data analyst. Give it a CSV file and a natural language question — it will autonomously plan, execute, validate, and explain a full data analysis.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                          │
│   Central control loop · manages state · prevents loops      │
└───────────┬───────────────────────────────────┬─────────────┘
            │                                   │
     ┌──────▼──────┐                   ┌────────▼────────┐
     │   PLANNER   │                   │  MEMORY SYSTEM  │
     │   Agent     │                   │  (session state)│
     │             │                   └─────────────────┘
     │ Reads meta- │
     │ data+query  │
     │ → produces  │
     │ step list   │
     └──────┬──────┘
            │ plan steps
            ▼
┌───────────────────────────────────────────────────────────┐
│                     ANALYSIS LOOP                          │
│                                                           │
│  ┌────────────┐    code    ┌──────────────┐              │
│  │  ANALYST   │──────────▶│  PYTHON      │              │
│  │  Agent     │           │  EXECUTOR    │              │
│  │            │◀──────────│  (sandboxed) │              │
│  └────────────┘   output  └──────┬───────┘              │
│         ▲                        │ output                │
│         │ improve               ▼                        │
│  ┌──────┴──────┐         ┌──────────────┐              │
│  │   CRITIC    │◀────────│  validates   │              │
│  │   Agent     │ rejects │  & scores    │              │
│  │             │──────── │  0.0 – 1.0   │              │
│  └─────────────┘ approves└──────────────┘              │
│                           → store in memory              │
└───────────────────────────────────────────────────────────┘
            │
            ▼
     ┌──────────────┐
     │ FINAL INSIGHT│
     │ GENERATION   │
     │ (Critic LLM) │
     └──────────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-agent** | Planner → Analyst → Critic with CrewAI roles |
| **Real code execution** | Sandboxed Python with pandas/numpy/matplotlib |
| **Retry logic** | Up to 3 retries per step with critic feedback |
| **Critic scoring** | 0.0–1.0 quality gate per step |
| **Memory system** | Full session state, prevents repetition |
| **Auto-visualization** | Charts auto-saved as PNG files |
| **Large dataset handling** | Auto-samples datasets > 10,000 rows |
| **Business insights** | Markdown report with recommendations |
| **REST API** | FastAPI with async background jobs |
| **CLI** | Rich terminal output with progress tracking |

---

## 📁 Project Structure

```
autonomous-data-analyst/
│
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py      # Strategic analysis planning
│   ├── analyst_agent.py      # Python code generation
│   └── critic_agent.py       # Output validation & scoring
│
├── tools/
│   ├── __init__.py
│   └── python_executor.py    # Sandboxed Python execution engine
│
├── core/
│   ├── __init__.py
│   ├── memory.py             # Session state & step tracking
│   ├── metadata_extractor.py # Dataset structural analysis
│   └── orchestrator.py       # Central control loop
│
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI REST endpoints
│
├── data/
│   └── generate_sample.py    # Creates sample_sales.csv for testing
│
├── outputs/                  # Auto-created: PNGs + session JSONs
├── logs/                     # Auto-created: log files
│
├── run.py                    # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python 3.10 or 3.11 (recommended)
- An [Anthropic API key](https://console.anthropic.com/)
- VS Code (optional but recommended)

### 2. Clone / Download the Project

```bash
# If using git
git clone <your-repo-url>
cd autonomous-data-analyst

# Or just navigate to the project folder
cd autonomous-data-analyst
```

### 3. Create a Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment

```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` and set your Anthropic API key:

```env
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# Optional overrides (defaults shown)
LLM_MODEL=claude-sonnet-4-20250514
LLM_TEMPERATURE=0.2
MAX_RETRIES=3
CRITIC_SCORE_THRESHOLD=0.65
MAX_PLAN_STEPS=8
OUTPUT_DIR=outputs
LOG_DIR=logs
API_HOST=0.0.0.0
API_PORT=8000
```

### 6. Generate Sample Dataset

```bash
python data/generate_sample.py
```

This creates `data/sample_sales.csv` — a 1,200-row e-commerce sales dataset with realistic columns: revenue, profit, region, category, channel, date, customer satisfaction, etc.

---

## 🚀 Running the Project

### Option A: CLI (Recommended for Development)

```bash
# Basic usage
python run.py --csv data/sample_sales.csv --query "Which product category drives the most profit?"

# More queries to try
python run.py --csv data/sample_sales.csv --query "Identify seasonal revenue trends over the year"
python run.py --csv data/sample_sales.csv --query "Which sales channel has the highest customer satisfaction?"
python run.py --csv data/sample_sales.csv --query "Analyze return rates by category and their business impact"
python run.py --csv data/sample_sales.csv --query "What discount strategy maximizes profit margin?"

# Custom settings
python run.py --csv data/sample_sales.csv --query "..." --max-retries 2 --threshold 0.7 --max-steps 6
```

**What you'll see:**
```
────────────────────────────────────────────────────
  ▶  PIPELINE START
     Query: Which category drives the most profit? | Dataset: data/sample_sales.csv
────────────────────────────────────────────────────

  ▶  PLANNING
  ▶  STEP 1/7: Assess data quality...
  Critic score: 0.82 | Approved: True
  ✅ Step 1 completed | score=0.82 | retries=0

  ... (continues for all steps)

  ┌─ Final Insights & Recommendations ─────────────┐
  │ ## Executive Summary                            │
  │ Electronics leads profit at 34% share...        │
  └────────────────────────────────────────────────┘
```

---

### Option B: FastAPI Server

```bash
# Start the server
python run.py --server

# Or directly with uvicorn
uvicorn api.main:app --reload --port 8000
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Upload CSV + query → returns `job_id` |
| `GET` | `/jobs/{job_id}` | Poll for status and results |
| `GET` | `/sessions` | List all saved sessions |
| `GET` | `/sessions/{id}` | Get a specific session |
| `GET` | `/visualizations/{file}` | Download a chart PNG |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

**Example API Call (curl):**

```bash
# Submit analysis job
curl -X POST http://localhost:8000/analyze \
  -F "file=@data/sample_sales.csv" \
  -F "query=Which product category drives the most profit?" \
  -F "max_retries=3" \
  -F "score_threshold=0.65"

# Response: { "job_id": "abc-123...", "status": "queued" }

# Poll for results
curl http://localhost:8000/jobs/abc-123...
```

**Example API Call (Python):**

```python
import requests, time

# Submit
with open("data/sample_sales.csv", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        data={"query": "Which category drives the most profit?"}
    )
job_id = resp.json()["job_id"]

# Poll until done
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    if status["status"] in ("completed", "failed"):
        break
    print(f"Status: {status['status']} ...")
    time.sleep(5)

print(status["final_insights"])
```

---

### Option C: VS Code (Development)

1. Open the project folder in VS Code
2. Install the Python extension
3. Select the `venv` interpreter (`Ctrl+Shift+P` → "Python: Select Interpreter")
4. Open the integrated terminal and run:
   ```bash
   python run.py --csv data/sample_sales.csv --query "Which category is most profitable?"
   ```
5. For API development, install the REST Client extension and use the Swagger UI at `http://localhost:8000/docs`

**Recommended VS Code extensions:**
- Python (Microsoft)
- Pylance
- REST Client
- Thunder Client (API testing)

---

## 🧠 Agent Roles

### 🗓️ Planner Agent
- **Role:** Senior Data Analysis Strategist
- **Input:** Dataset metadata + user query
- **Output:** Ordered list of 4–8 specific analysis steps
- **Behavior:** Considers domain, data types, missing values, correlations
- **LLM calls:** 1 per session

### 🔬 Analyst Agent
- **Role:** Senior Data Analyst & Python Engineer
- **Input:** Step description + prior context + critic feedback (on retry)
- **Output:** Clean, executable Python code only
- **Behavior:** Handles NaN, type errors, uses matplotlib/seaborn with proper labels
- **LLM calls:** 1–4 per step (initial + retries)

### ⚖️ Critic Agent
- **Role:** Chief Data Science Critic & QA Lead
- **Input:** Step description + generated code + execution output
- **Output:** JSON with score (0–1), approval, feedback, improvement suggestions
- **Behavior:** Rejects empty output, execution errors, vague conclusions
- **Threshold:** Score ≥ 0.65 to approve (configurable)
- **LLM calls:** 1 per step attempt

### 🎵 Orchestrator
- **Role:** Pipeline controller
- **Behavior:** Loops through plan, manages retries (max 3), stores state in memory
- **Safeguards:** Max steps cap, retry limit, graceful failure handling

---

## 📊 Output Files

After each run, the `outputs/` directory contains:

```
outputs/
├── session_20250101_120000.json    # Full session data (all steps, scores, outputs)
├── 20250101_120000_step_plot_000.png
├── 20250101_120000_step_plot_001.png
└── ...
```

The JSON session file has this structure:
```json
{
  "session_id": "20250101_120000",
  "user_query": "Which category drives the most profit?",
  "analysis_plan": ["Step 1...", "Step 2..."],
  "steps": [
    {
      "step_id": 1,
      "description": "...",
      "code": "...",
      "output": "...",
      "critic_score": 0.82,
      "status": "completed",
      "visualizations": ["outputs/...png"]
    }
  ],
  "final_insights": "## Executive Summary\n...",
  "summary": {
    "total_steps": 7,
    "completed": 6,
    "avg_critic_score": 0.78
  }
}
```

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic code |
| `MAX_RETRIES` | `3` | Max retries per step if critic rejects |
| `CRITIC_SCORE_THRESHOLD` | `0.65` | Min score for step approval |
| `MAX_PLAN_STEPS` | `8` | Max analysis steps per session |
| `MIN_PLAN_STEPS` | `4` | Min analysis steps per session |
| `OUTPUT_DIR` | `outputs` | Where PNGs and JSON are saved |
| `LOG_DIR` | `logs` | Log file location |
| `MAX_DATASET_ROWS` | `10000` | Rows before sampling kicks in |
| `SAMPLE_SIZE` | `5000` | Rows to sample for large datasets |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |

---

## 🐛 Troubleshooting

**`ANTHROPIC_API_KEY not found`**
→ Ensure `.env` exists and has your key. Run `cp .env.example .env` first.

**`ModuleNotFoundError: No module named 'crewai'`**
→ Ensure your venv is activated and run `pip install -r requirements.txt`

**`FileNotFoundError: data/sample_sales.csv`**
→ Run `python data/generate_sample.py` to create the sample dataset first.

**Analysis takes too long**
→ Reduce `MAX_PLAN_STEPS=4` and `MAX_RETRIES=1` in `.env`.

**Critic always rejects**
→ Lower `CRITIC_SCORE_THRESHOLD=0.5` or raise `LLM_TEMPERATURE=0.4`.

**Charts not saved**
→ Ensure `outputs/` directory is writable. Check `LOG_DIR/cli.log` for errors.

---

## 🏛️ Technology Stack

| Layer | Technology |
|---|---|
| **Agents** | [CrewAI](https://github.com/crewAIInc/crewAI) |
| **LLM** | [Claude (Anthropic)](https://anthropic.com) via [LangChain](https://langchain.com) |
| **API** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| **Data** | pandas, numpy, scipy |
| **Visualization** | matplotlib, seaborn |
| **CLI** | [Rich](https://github.com/Textualize/rich) |
| **Config** | python-dotenv |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with Claude Sonnet · CrewAI · FastAPI · pandas*
