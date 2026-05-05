"""
Critic Agent — validates analysis outputs and challenges weak conclusions.

Responsibilities
----------------
- Score result quality on a 0–1 scale
- Detect errors, empty outputs, statistical fallacies, weak logic
- Identify whether the output addresses the original query
- Provide specific, actionable improvement suggestions
- Approve (score ≥ threshold) or reject (score < threshold) the step
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from crewai import Agent
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriticResult:
    score: float                    # 0.0 – 1.0
    approved: bool
    feedback: str
    improvements: str               # actionable suggestions for the analyst
    reasoning: str                  # internal critic reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "approved": self.approved,
            "feedback": self.feedback,
            "improvements": self.improvements,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_critic_agent(llm: Any, score_threshold: float = 0.65) -> Agent:
    """Return a CrewAI Agent configured as the Critic."""
    return Agent(
        role="Chief Data Science Critic & Quality Assurance Lead",
        goal=(
            f"Ensure every analysis step is statistically sound, addresses the user query, "
            f"and meets quality standards (minimum score: {score_threshold:.0%}). "
            f"Reject weak or erroneous outputs and demand improvements."
        ),
        backstory="""You are the toughest reviewer in the data science team — a PhD statistician
who has seen every form of analysis mistake: cherry-picked metrics, overlooked outliers,
correlation-causation confusion, misleading charts, and missing error bars. You do not
accept vague or generic outputs. You demand specificity, statistical rigour, and outputs
that directly answer the business question. Your feedback is blunt but fair and always
constructive.""",
        verbose=True,
        allow_delegation=False,
        llm=f"groq/{getattr(llm, 'model_name', 'llama-3.3-70b-versatile')}",
        max_iter=1,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a rigorous data science critic. Evaluate the analysis step and
return ONLY a valid JSON object — no explanation outside the JSON."""


def validate_step_output(
    llm: Any,
    context: str,
    code: str,
    output: str,
    step_description: str,
    user_query: str,
    score_threshold: float = 0.65,
    execution_error: Optional[str] = None,
) -> CriticResult:
    """
    Validate the output of an analysis step.

    Returns a CriticResult with a score, approval status, and improvement notes.
    """

    # Auto-fail on execution errors
    if execution_error and "ERROR" in (output + execution_error):
        return CriticResult(
            score=0.1,
            approved=False,
            feedback="The code raised a Python exception during execution.",
            improvements=_extract_error_suggestions(execution_error or output),
            reasoning="Execution failed; automatic rejection.",
        )

    # Auto-fail on empty output
    if not output.strip() or output.strip() in {
        "Code executed successfully with no text output.",
        "Code executed with no output.",
    }:
        return CriticResult(
            score=0.2,
            approved=False,
            feedback="The code produced no visible output or results.",
            improvements=(
                "Add print() statements to display all computed values. "
                "Ensure the analysis produces interpretable numeric or textual results."
            ),
            reasoning="Empty output; automatic rejection.",
        )

    user_prompt = f"""
Evaluate this data analysis step with strict quality standards.

USER QUERY (the ultimate goal): "{user_query}"

ANALYSIS STEP BEING EVALUATED: "{step_description}"

CODE EXECUTED:
```python
{code[:3000]}
```

EXECUTION OUTPUT:
{output[:3000]}

Score this step from 0.0 to 1.0 across these dimensions:
- correctness      : Is the output factually/statistically correct? (0–1)
- relevance        : Does it address the user query? (0–1)
- completeness     : Are all important aspects covered? (0–1)
- clarity          : Are results clearly printed/labeled? (0–1)
- statistical_depth: Is appropriate statistical reasoning applied? (0–1)

Return ONLY this JSON (no other text):
{{
  "overall_score": <float 0.0–1.0>,
  "correctness": <float>,
  "relevance": <float>,
  "completeness": <float>,
  "clarity": <float>,
  "statistical_depth": <float>,
  "approved": <true|false>,
  "feedback": "<1–2 sentence summary of the evaluation>",
  "improvements": "<specific, actionable improvements the analyst must make>",
  "reasoning": "<internal critic reasoning, 2–3 sentences>"
}}

Approve (approved: true) if overall_score >= {score_threshold}.
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        result = _parse_critic_response(response.content, score_threshold)
        return result
    except Exception as e:
        print(f"[Critic] LLM call failed: {e}")
        # Conservative fallback — partial approval
        return CriticResult(
            score=0.5,
            approved=False,
            feedback="Critic evaluation failed; conservative rejection.",
            improvements="Re-run the analysis with more explicit print statements.",
            reasoning=f"LLM error: {str(e)[:200]}",
        )


def generate_final_insights(
    llm: Any,
    full_context: str,
    metadata: Dict[str, Any],
) -> str:
    """
    Generate the final business-level insights and recommendations
    based on the complete analysis context.
    """
    domain = metadata.get("inferred_domain", "general")

    prompt = f"""
You are a senior data science consultant producing the final deliverable for a client.

Based on the complete analysis below, write a structured report with:
1. **Executive Summary** (2–3 sentences)
2. **Key Findings** (bullet points, each with a supporting statistic)
3. **Business Insights** (what do these findings mean for the business?)
4. **Recommendations** (3–5 specific, prioritized, actionable recommendations)
5. **Caveats & Limitations** (data quality issues, sample size, assumptions)

Domain context: {domain}

FULL ANALYSIS CONTEXT:
{full_context[:6000]}

Write in clear, professional business language. Use bold headers. Be specific — cite
actual numbers from the analysis wherever possible.
""".strip()

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"Final insight generation failed: {e}\n\nPlease review the individual step outputs above."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_critic_response(content: str, threshold: float) -> CriticResult:
    """Parse the LLM JSON response into a CriticResult."""
    content = content.strip()

    # Try direct parse
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Find JSON object in text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError("Could not parse critic JSON")
        else:
            raise ValueError("No JSON found in critic response")

    score = float(data.get("overall_score", 0.5))
    approved = score >= threshold

    return CriticResult(
        score=score,
        approved=approved,
        feedback=str(data.get("feedback", "")),
        improvements=str(data.get("improvements", "")),
        reasoning=str(data.get("reasoning", "")),
    )


def _extract_error_suggestions(error_text: str) -> str:
    """Generate improvement hints from a Python traceback."""
    suggestions = ["Fix the Python error before re-running."]

    if "KeyError" in error_text:
        suggestions.append("A column name doesn't exist — check df.columns before accessing.")
    if "ValueError" in error_text:
        suggestions.append("A value conversion failed — use pd.to_numeric(errors='coerce') or pd.to_datetime(errors='coerce').")
    if "AttributeError" in error_text:
        suggestions.append("An attribute doesn't exist on the object — check the dtype first.")
    if "MemoryError" in error_text:
        suggestions.append("Dataset too large — sample with df.sample(1000).")
    if "NameError" in error_text:
        suggestions.append("A variable is undefined — ensure imports and variable names are correct.")
    if "TypeError" in error_text:
        suggestions.append("A type mismatch — cast columns to correct types before operations.")

    return " ".join(suggestions) + f"\n\nError detail:\n{error_text[:800]}"
