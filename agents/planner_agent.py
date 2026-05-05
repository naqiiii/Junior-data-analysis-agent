"""
Planner Agent — creates a strategic, step-by-step analysis plan.

Responsibilities
----------------
- Understand the user query and dataset structure
- Devise a logical, ordered sequence of analysis steps
- Tailor the plan to the inferred business domain
- Return a clean, parseable list of steps
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from crewai import Agent
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_planner_agent(llm: Any) -> Agent:
    """Return a CrewAI Agent configured as the Planner."""
    return Agent(
        role="Senior Data Analysis Strategist",
        goal=(
            "Design a comprehensive, logically ordered analysis plan that directly addresses "
            "the user's query, exploits the dataset's structure, and surfaces actionable insights."
        ),
        backstory="""You are a principal data scientist with 15+ years of experience across
e-commerce, finance, healthcare, and SaaS. You are known for your ability to decompose
vague business questions into precise, executable analysis steps.  You think like a
consultant: you always ask 'so what?' — your plans are not just exploratory, they drive
real decisions. You strictly limit visualization steps: ONLY include a visualization step if it is absolutely critical to the core question. Do not generate extraneous exploratory plots.""",
        verbose=True,
        allow_delegation=False,
        llm=f"groq/{getattr(llm, 'model_name', 'llama-3.3-70b-versatile')}",
        max_iter=1,
    )


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior data science strategist. Your ONLY output must be a
valid JSON array of analysis step strings. Do NOT include any explanation, preamble,
or markdown formatting outside the JSON array."""

def generate_analysis_plan(
    llm: Any,
    metadata: Dict[str, Any],
    query: str,
    max_steps: int = 8,
    min_steps: int = 4,
) -> List[str]:
    """
    Call the LLM to produce a structured analysis plan.

    Returns a list of step description strings (4–8 items).
    Falls back to a sensible default plan on parse failure.
    """

    meta_summary = _build_metadata_summary(metadata)

    user_prompt = f"""
Analyze this dataset to answer the user's query.

USER QUERY:
"{query}"

DATASET OVERVIEW:
{meta_summary}

Generate a step-by-step analysis plan with {min_steps}–{max_steps} steps.

Rules:
1. Each step must be a single, specific, actionable task (one Python operation or chart).
2. Always start with a data quality / cleaning check.
3. Include at least one visualization step.
4. Include at least one statistical analysis step.
5. End with a "Generate business insights and recommendations" step.
6. Steps must build logically on each other.
7. Tailor steps to the inferred domain: {metadata.get('inferred_domain', 'general')}.

Return ONLY a JSON array of strings, like:
[
  "Check and handle missing values, duplicates, and data type corrections",
  "Compute descriptive statistics for all numeric columns",
  ...
]
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        content = response.content.strip()
        steps = _parse_plan(content)
        if len(steps) >= min_steps:
            return steps[:max_steps]
    except Exception as e:
        print(f"[Planner] LLM call failed: {e}")

    # Fallback plan
    return _default_plan(metadata, query)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_metadata_summary(metadata: Dict[str, Any]) -> str:
    shape = metadata.get("shape", {})
    dtypes = metadata.get("dtypes", {})
    missing = metadata.get("missing_values", {})
    numeric_cols = [c for c, t in dtypes.items() if "float" in t or "int" in t]
    cat_cols = [c for c, t in dtypes.items() if "object" in t or "category" in t]

    missing_info = ", ".join(
        f"{col}: {cnt}" for col, cnt in missing.items() if cnt > 0
    ) or "none"

    top_corr = metadata.get("top_correlations", [])
    corr_text = ""
    if top_corr:
        corr_text = "Top correlations: " + ", ".join(
            f"{c['col1']}↔{c['col2']} ({c['correlation']:.2f})"
            for c in top_corr[:3]
        )

    return f"""
- Rows: {shape.get('rows', '?')} | Columns: {shape.get('columns', '?')}
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}
- Columns with missing values: {missing_info}
- Large dataset: {metadata.get('is_large_dataset', False)} (sampled to {metadata.get('sample_size', '?')} rows if large)
- Has datetime columns: {metadata.get('has_datetime_columns', False)}
- Inferred domain: {metadata.get('inferred_domain', 'general')}
- {corr_text}
""".strip()


def _parse_plan(content: str) -> List[str]:
    """Try to extract a JSON array from the LLM response."""
    # Direct parse
    try:
        steps = json.loads(content)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
    except json.JSONDecodeError:
        pass

    # Find JSON array within text
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        try:
            steps = json.loads(match.group())
            if isinstance(steps, list):
                return [str(s) for s in steps]
        except json.JSONDecodeError:
            pass

    # Last resort: extract numbered/bulleted lines
    lines = []
    for line in content.split("\n"):
        line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
        if len(line) > 15:
            lines.append(line)
    if lines:
        return lines

    return []


def _default_plan(metadata: Dict[str, Any], query: str) -> List[str]:
    """Fallback plan when LLM is unavailable or output unparseable."""
    has_datetime = metadata.get("has_datetime_columns", False)
    is_large = metadata.get("is_large_dataset", False)

    steps = [
        "Assess data quality: identify missing values, duplicate rows, and incorrect data types; apply necessary corrections",
        "Compute descriptive statistics (mean, median, std, min, max) for all numeric columns and review distributions",
        "Visualize distributions of key numeric variables using histograms and box plots to detect outliers",
    ]

    if has_datetime:
        steps.append("Parse datetime columns and analyze trends over time with line charts")

    steps += [
        "Examine relationships between variables using a correlation heatmap and scatter plots",
        "Perform group-by aggregations relevant to the user query and visualize comparisons",
        "Identify top-performing and bottom-performing segments based on key metrics",
        "Generate business insights and actionable recommendations based on all findings",
    ]

    return steps
