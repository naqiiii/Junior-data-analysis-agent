"""
Memory System — tracks all analysis steps, results, and session state.
Prevents repetition, enables context-aware agent decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AnalysisStep:
    step_id: int
    description: str
    code: str = ""
    output: str = ""
    critic_score: float = 0.0
    critic_feedback: str = ""
    status: str = "pending"          # pending | running | completed | failed | skipped
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retry_count: int = 0
    visualizations: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "code": self.code,
            "output": self.output[:2000],          # truncate for summary
            "critic_score": self.critic_score,
            "critic_feedback": self.critic_feedback,
            "status": self.status,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "visualizations": self.visualizations,
            "error": self.error,
        }

    def short_summary(self) -> str:
        """Return a compact summary suitable for agent context windows."""
        status_emoji = {"completed": "✅", "failed": "❌", "skipped": "⏭️", "pending": "⏳"}.get(
            self.status, "❓"
        )
        return (
            f"{status_emoji} Step {self.step_id}: {self.description}\n"
            f"   Score: {self.critic_score:.2f} | Retries: {self.retry_count}\n"
            f"   Output preview: {self.output[:300].strip()}\n"
        )


# ---------------------------------------------------------------------------
# Memory System
# ---------------------------------------------------------------------------

class MemorySystem:
    """
    Central memory store for a single analysis session.
    Agents read from and write to this object via the orchestrator.
    """

    def __init__(self) -> None:
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_metadata: Dict[str, Any] = {}
        self.user_query: str = ""
        self.dataset_path: str = ""
        self.analysis_plan: List[str] = []
        self.steps: List[AnalysisStep] = []
        self.final_insights: str = ""
        self.all_visualizations: List[str] = []
        self.created_at: str = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, step: AnalysisStep) -> None:
        self.steps.append(step)
        if step.visualizations:
            self.all_visualizations.extend(step.visualizations)

    def update_step(self, step_id: int, **kwargs: Any) -> None:
        for step in self.steps:
            if step.step_id == step_id:
                for key, value in kwargs.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                if "visualizations" in kwargs:
                    self.all_visualizations.extend(kwargs["visualizations"])
                return

    def get_step(self, step_id: int) -> Optional[AnalysisStep]:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def get_completed_steps(self) -> List[AnalysisStep]:
        return [s for s in self.steps if s.status == "completed"]

    def get_failed_steps(self) -> List[AnalysisStep]:
        return [s for s in self.steps if s.status == "failed"]

    # ------------------------------------------------------------------
    # Context builders (used to inject context into agent prompts)
    # ------------------------------------------------------------------

    def build_context_for_analyst(self, current_step_description: str) -> str:
        """Compact context fed to the Analyst Agent before code generation."""
        completed = self.get_completed_steps()
        prior_outputs = ""
        if completed:
            prior_outputs = "\n".join(
                f"  - Step {s.step_id} ({s.description}): {s.output[:400].strip()}"
                for s in completed[-3:]   # last 3 steps max
            )
        else:
            prior_outputs = "  (No prior steps completed yet)"

        meta = self.dataset_metadata
        return f"""
Dataset: {self.dataset_path}
Shape: {meta.get('shape', {}).get('rows', '?')} rows × {meta.get('shape', {}).get('columns', '?')} columns
Columns: {', '.join(meta.get('dtypes', {}).keys())}
User Query: {self.user_query}

Current Task: {current_step_description}

Prior Completed Analysis (last 3 steps):
{prior_outputs}
""".strip()

    def build_context_for_critic(self, step: AnalysisStep) -> str:
        """Context fed to the Critic Agent when validating a step."""
        return f"""
User Query: {self.user_query}
Analysis Step: {step.description}
Code Executed:
```python
{step.code}
```
Execution Output:
{step.output[:3000]}
""".strip()

    def build_final_context(self) -> str:
        """Full context for the final insight generation."""
        step_summaries = "\n\n".join(s.short_summary() for s in self.steps)
        all_outputs = "\n\n".join(
            f"--- Step {s.step_id}: {s.description} ---\n{s.output[:1500]}"
            for s in self.get_completed_steps()
        )
        return f"""
User Query: {self.user_query}
Dataset: {self.dataset_path}
Shape: {self.dataset_metadata.get('shape', {})}

Analysis Plan:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(self.analysis_plan))}

Step Execution Summary:
{step_summaries}

Detailed Outputs:
{all_outputs}
""".strip()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "user_query": self.user_query,
            "dataset_path": self.dataset_path,
            "dataset_metadata": self.dataset_metadata,
            "analysis_plan": self.analysis_plan,
            "steps": [s.to_dict() for s in self.steps],
            "final_insights": self.final_insights,
            "all_visualizations": self.all_visualizations,
            "summary": {
                "total_steps": len(self.steps),
                "completed": len(self.get_completed_steps()),
                "failed": len(self.get_failed_steps()),
                "avg_critic_score": (
                    sum(s.critic_score for s in self.get_completed_steps())
                    / max(len(self.get_completed_steps()), 1)
                ),
            },
        }

    def save_to_file(self, output_dir: str = "outputs") -> str:
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"session_{self.session_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path
