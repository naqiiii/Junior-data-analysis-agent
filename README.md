# 🤖 Autonomous Data Analyst Agent

A **production-grade multi-agent AI system** that behaves like a mid-level data analyst. Give it a CSV file and a natural language question — it will autonomously plan, execute, validate, and explain a full data analysis.

---

##  Architecture Overview

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

## Features

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

##  Output Files

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

##  Technology Stack

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
