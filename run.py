"""
CLI Entry Point — run analysis directly from the command line.

Usage
-----
  python run.py --csv data/sample_sales.csv --query "Which product category drives the most profit?"
  python run.py --csv data/sample_sales.csv --query "Identify seasonal trends in revenue"
  python run.py --server   # start the FastAPI server instead
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "cli.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI analysis runner
# ---------------------------------------------------------------------------

def run_cli_analysis(csv_path: str, query: str, **kwargs) -> None:
    """Run the full agent pipeline and print a formatted report."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[red]GROQ_API_KEY not found in .env[/red]\n"
                "Copy .env.example → .env and add your key.",
                title="Configuration Error",
            )
        )
        sys.exit(1)

    if not os.path.exists(csv_path):
        console.print(f"[red]Dataset not found:[/red] {csv_path}")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold cyan]Autonomous Data Analyst Agent[/bold cyan]\n\n"
            f"[yellow]Dataset:[/yellow] {csv_path}\n"
            f"[yellow]Query:[/yellow]   {query}",
            title="🤖 Starting Analysis",
            border_style="cyan",
        )
    )

    from core.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        groq_api_key=api_key,
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_retries=int(kwargs.get("max_retries") if kwargs.get("max_retries") is not None else os.getenv("MAX_RETRIES", "3")),
        score_threshold=float(kwargs.get("score_threshold") if kwargs.get("score_threshold") is not None else os.getenv("CRITIC_SCORE_THRESHOLD", "0.65")),
        max_plan_steps=int(kwargs.get("max_plan_steps") if kwargs.get("max_plan_steps") is not None else os.getenv("MAX_PLAN_STEPS", "8")),
        output_dir=os.getenv("OUTPUT_DIR", "outputs"),
        log_dir=LOG_DIR,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Running multi-agent analysis pipeline...", total=None)
        result = orchestrator.run(dataset_path=csv_path, query=query)
        progress.update(task, completed=True, description="Analysis complete")

    _print_report(result)


def _print_report(result: dict) -> None:
    """Render the analysis result as a rich terminal report."""
    console.print()

    # ── Dataset summary ────────────────────────────────────────────────────
    meta = result.get("dataset_metadata", {})
    shape = meta.get("shape", {})
    console.print(
        Panel(
            f"Rows: [bold]{shape.get('rows', '?')}[/bold] | "
            f"Columns: [bold]{shape.get('columns', '?')}[/bold] | "
            f"Domain: [bold]{meta.get('inferred_domain', '?')}[/bold]",
            title="Dataset Info",
            border_style="blue",
        )
    )

    # ── Analysis plan ──────────────────────────────────────────────────────
    plan = result.get("analysis_plan", [])
    if plan:
        plan_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
        console.print(Panel(plan_text, title="Analysis Plan", border_style="green"))

    # ── Step results table ─────────────────────────────────────────────────
    steps = result.get("steps", [])
    if steps:
        table = Table(title="Execution Summary", show_lines=True)
        table.add_column("Step", style="cyan", width=6)
        table.add_column("Description", style="white", width=40)
        table.add_column("Status", width=10)
        table.add_column("Score", width=8)
        table.add_column("Retries", width=8)
        table.add_column("Plots", width=6)

        for s in steps:
            status = s.get("status", "?")
            score = s.get("critic_score", 0)
            score_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            table.add_row(
                str(s.get("step_id", "?")),
                s.get("description", "")[:40] + ("..." if len(s.get("description", "")) > 40 else ""),
                status,
                f"[{score_color}]{score:.2f}[/{score_color}]",
                str(s.get("retry_count", 0)),
                str(len(s.get("visualizations", []))),
            )
        console.print(table)

    # ── Visualizations ─────────────────────────────────────────────────────
    plots = result.get("all_visualizations", [])
    if plots:
        console.print(
            Panel(
                "\n".join(f"  {p}" for p in plots),
                title=f" Saved Visualizations ({len(plots)})",
                border_style="magenta",
            )
        )

    # ── Final insights ─────────────────────────────────────────────────────
    insights = result.get("final_insights", "")
    if insights:
        console.print(Panel(Markdown(insights), title="💡 Final Insights & Recommendations", border_style="yellow"))

    # ── Session summary ────────────────────────────────────────────────────
    summary = result.get("summary", {})
    elapsed = result.get("elapsed_seconds", 0)
    session_file = result.get("session_file", "")
    console.print(
        Panel(
            f"Session ID:  [bold]{result.get('session_id', '?')}[/bold]\n"
            f"Elapsed:     [bold]{elapsed}s[/bold]\n"
            f"Completed:   [bold]{summary.get('completed', 0)}/{summary.get('total_steps', 0)}[/bold] steps\n"
            f"Avg Score:   [bold]{summary.get('avg_critic_score', 0):.2f}[/bold]\n"
            f"Session File:[bold] {session_file}[/bold]",
            title="📁 Session Summary",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def run_server() -> None:
    """Start the FastAPI server via uvicorn."""
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    console.print(
        Panel(
            f"Starting API server at [bold]http://{host}:{port}[/bold]\n"
            f"Docs: [bold]http://localhost:{port}/docs[/bold]",
            title="FastAPI Server",
            border_style="cyan",
        )
    )
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Data Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --file data/sample.csv --query "Which category drives the most profit?"
  python run.py --file data/sample.csv --query "Show revenue trends over time"
  python run.py --server
        """,
    )

    parser.add_argument("--file", type=str, help="Path to CSV/TSV/DOCX dataset")
    parser.add_argument("--query", type=str, help="Natural language analysis query")
    parser.add_argument("--server", action="store_true", help="Start the FastAPI server")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries per step")
    parser.add_argument("--threshold", type=float, default=None, help="Critic score threshold (0–1)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max plan steps")

    args = parser.parse_args()

    if args.server:
        run_server()
        return

    if not args.file or not args.query:
        console.print(
            "[red]Error:[/red] Both --file and --query are required for analysis.\n"
            "Use --server to start the API instead."
        )
        parser.print_help()
        sys.exit(1)

    run_cli_analysis(
        csv_path=args.file,
        query=args.query,
        max_retries=args.max_retries,
        score_threshold=args.threshold,
        max_plan_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
