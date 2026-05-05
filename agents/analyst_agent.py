
"""
Analyst Agent — converts analysis plan steps into executable Python code.

Responsibilities
----------------
- Write correct, self-contained Python code for each analysis step
- Use pandas, numpy, matplotlib (seaborn) idiomatically
- Handle missing data, type casting, and edge cases
- Generate clean visualizations with titles and labels
- Return ONLY Python code — no explanations, no markdown prose
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from crewai import Agent
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_analyst_agent(llm: Any) -> Agent:
    """Return a CrewAI Agent configured as the Analyst."""
    return Agent(
        role="Senior Data Analyst & Python Engineer",
        goal=(
            "Write production-quality Python code that accurately executes the given analysis "
            "step, handles edge cases, and produces clear, labeled visualizations."
        ),
        backstory="""You are a senior data analyst with deep expertise in Python, pandas, numpy,
matplotlib, and seaborn. You have written thousands of analysis scripts and know every
common pitfall: mixed dtypes, NaN propagation, off-by-one errors in groupbys, silent
silent type coercions. You write clean, readable code with meaningful print statements
so that results are easily interpreted. You NEVER return explanations — only code.
CRITICAL INSTRUCTION: You MUST NOT generate any plots unless explicitly instructed by the plan. Avoid creating multiple plots or looping over columns to create plots unless necessary.""",
        verbose=True,
        allow_delegation=False,
        llm=f"groq/{getattr(llm, 'model_name', 'llama-3.3-70b-versatile')}",
        max_iter=1,
    )


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Python data analyst. Your output must be ONLY valid Python
code — no markdown, no explanation, no commentary outside of # comments inside the code.
The dataset is already loaded as `df` (a pandas DataFrame). Do NOT re-read the CSV."""


def generate_analysis_code(
    llm: Any,
    step_description: str,
    context: str,
    metadata: Dict[str, Any],
    critic_feedback: Optional[str] = None,
    retry_count: int = 0,
) -> str:
    """
    Ask the LLM to write Python code for one analysis step.

    Parameters
    ----------
    step_description : The current analysis step (from the plan).
    context          : Compact context built by MemorySystem.
    metadata         : Dataset metadata dict.
    critic_feedback  : If retrying, the critic's improvement suggestions.
    retry_count      : How many times this step has been retried.

    Returns
    -------
    A string of Python code ready for execution.
    """
    dtypes_str = _format_dtypes(metadata)
    numeric_cols = [c for c, t in metadata.get("dtypes", {}).items() if "float" in t or "int" in t]
    cat_cols = [c for c, t in metadata.get("dtypes", {}).items() if "object" in t or "category" in t]
    datetime_cols = metadata.get("datetime_columns", [])
    missing = {k: v for k, v in metadata.get("missing_values", {}).items() if v > 0}

    retry_note = ""
    if retry_count > 0 and critic_feedback:
        retry_note = f"""
        RETRY #{retry_count} — Previous attempt was rejected by the Critic.
Critic Feedback:
{critic_feedback}

Address ALL feedback above in this revised implementation.
"""

    user_prompt = f"""
{retry_note}
ANALYSIS CONTEXT: 
{context}

CURRENT TASK:
{step_description}

DATASET DETAILS:
- Columns & dtypes: {dtypes_str}
- Numeric columns: {', '.join(numeric_cols[:15])}
- Categorical columns: {', '.join(cat_cols[:10])}
- Datetime columns: {', '.join(datetime_cols) if datetime_cols else 'none'}
- Columns with missing values: {missing if missing else 'none'}
- Large dataset (sampled): {metadata.get('is_large_dataset', False)}
- User query: {metadata.get('query', '')}

CODING REQUIREMENTS:
1. `df` is already loaded — do NOT call pd.read_csv().
2. Handle NaN values explicitly (dropna, fillna, or skip) — never let NaN silently corrupt results.
3. Check that columns exist before using them: use `if 'col' in df.columns`.
4. Cast dtypes if needed (pd.to_numeric with errors='coerce', pd.to_datetime with errors='coerce').
5. Every plot MUST have: title, axis labels, and call plt.tight_layout() before plt.show().
6. Use plt.show() to trigger auto-save — do NOT call plt.savefig() yourself.
7. Print all numeric results with clear labels, e.g.: print(f"Mean Sales: {{df['sales'].mean():.2f}}")
8. Use seaborn (imported as sns if available) for prettier charts when appropriate.
9. Keep code concise and readable — no dead code or redundant loops.
10. Use try/except around risky operations and print a helpful message on failure.

Return ONLY the Python code. No markdown fences, no explanations.
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        code = _clean_code(response.content)
        if not code.strip():
            raise ValueError("Empty code returned")
        return code
    except Exception as e:
        print(f"[Analyst] LLM call failed: {e}")
        return _fallback_code(step_description, metadata)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_dtypes(metadata: Dict[str, Any]) -> str:
    dtypes = metadata.get("dtypes", {})
    return ", ".join(f"{col}({dtype})" for col, dtype in list(dtypes.items())[:20])


def _clean_code(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    raw = raw.strip()
    # Remove ```python ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:python|py)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def _fallback_code(step_description: str, metadata: Dict[str, Any]) -> str:
    """Generic fallback code when LLM is unavailable."""
    numeric_cols = [c for c, t in metadata.get("dtypes", {}).items() if "float" in t or "int" in t]
    first_num = numeric_cols[0] if numeric_cols else None

    lines = [
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        f"# Fallback code for: {step_description}",
        "print('=== Data Overview ===')",
        "print(df.shape)",
        "print(df.dtypes)",
        "print(df.isnull().sum())",
        "",
    ]

    if first_num:
        lines += [
            f"print('\\n=== {first_num} Statistics ===')",
            f"print(df['{first_num}'].describe())",
            "",
            "import matplotlib.pyplot as plt",
            f"plt.figure(figsize=(10, 4))",
            f"df['{first_num}'].dropna().hist(bins=30, edgecolor='black')",
            f"plt.title('Distribution of {first_num}')",
            f"plt.xlabel('{first_num}')",
            "plt.ylabel('Frequency')",
            "plt.tight_layout()",
            "plt.show()",
        ]

    return "\n".join(lines)
