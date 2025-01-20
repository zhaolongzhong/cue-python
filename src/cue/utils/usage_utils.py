import os
import json
from typing import Any, Optional
from logging import getLogger
from pathlib import Path
from datetime import datetime, timedelta

from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from ..types.completion_response import CompletionUsage

logger = getLogger(__name__)


def record_usage(
    response: Optional[CompletionUsage],
    subfolder: Optional[str] = "usage",
    filename: Optional[str] = None,
) -> Optional[dict]:
    """
    Record API usage statistics to a JSONL file.

    Args:
        response: CompletionResponse object containing usage information
    """
    usage = response.get_usage()
    if not usage:
        return

    logger.debug(f"completion response usage: {usage.model_dump(exclude_none=True, exclude_defaults=True)}")

    # Get base directory and create full path
    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    _filename = filename if filename else "usage"

    if not subfolder:
        subfolder = "usage"
    usage_path = base_dir / "logs" / subfolder / f"{_filename}.jsonl"

    # Create directory if it doesn't exist
    usage_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare usage entry
    usage_entry = {
        "id": response.get_id(),
        "model": response.model,
        "usage": usage.model_dump() if usage else None,
        "timestamp": datetime.now().isoformat(),
    }

    # Append to JSONL file
    with open(usage_path, "a", encoding="utf-8") as f:
        json.dump(usage_entry, f)
        f.write("\n")
    return usage_entry


def record_usage_details(
    usage_details: dict,
    subfolder: Optional[str] = "usage",
    filename: Optional[str] = "usage_detail",
) -> Optional[dict]:
    """
    Record API usage statistics to a JSONL file.
    """

    # Get base directory and create full path
    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    _filename = filename if filename else "usage"

    if not subfolder:
        subfolder = "usage"
    usage_path = base_dir / "logs" / subfolder / f"{_filename}.jsonl"

    # Create directory if it doesn't exist
    usage_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to JSONL file
    with open(usage_path, "a", encoding="utf-8") as f:
        json.dump(usage_details, f)
        f.write("\n")
    return usage_details


def analyze_usage(
    file_path: str,
    model: Optional[str] = None,
    min_cached_tokens: Optional[int] = None,
    max_cached_tokens: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list[dict]:
    """
    Filter and analyze usage data from JSONL file based on various criteria.
    """
    filtered_entries = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Apply filters
                if model and entry.get("model") != model:
                    continue

                cached_tokens = entry.get("usage", {}).get("cached_tokens", 0)
                if min_cached_tokens is not None and cached_tokens < min_cached_tokens:
                    continue
                if max_cached_tokens is not None and cached_tokens > max_cached_tokens:
                    continue

                if start_date or end_date:
                    entry_date = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_date < start_date:
                        continue
                    if end_date and entry_date > end_date:
                        continue

                filtered_entries.append(entry)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON lines
            except KeyError:
                continue  # Skip entries with missing required fields

    return filtered_entries


def get_usage_statistics(model: str, entries: list[dict]) -> dict:
    """
    Calculate statistics for the filtered usage entries.

    Args:
        model: Model name to determine statistics calculation method
        entries: List of filtered usage entries

    Returns:
        Dictionary containing usage statistics
    """
    if not entries:
        return {}

    def format_ratio(numerator: int, denominator: int) -> str:
        """Helper function to format ratios as '1/N' string"""
        if denominator == 0:
            return "N/A"
        if numerator == 0:
            return "0/1"
        # Normalize ratio to form 1/N
        ratio = denominator / numerator if numerator > 0 else float("inf")
        return f"1/{int(round(ratio))}" if ratio != float("inf") else "N/A"

    try:
        if "claude" in model.lower():
            total_input_tokens = sum(entry["usage"]["input_tokens"] for entry in entries)
            total_output_tokens = sum(entry["usage"]["output_tokens"] for entry in entries)
            total_cache_creation_input_tokens = sum(entry["usage"]["cache_creation_input_tokens"] for entry in entries)
            total_cache_read_input_tokens = sum(entry["usage"]["cache_read_input_tokens"] for entry in entries)

            return {
                "count": len(entries),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_input_tokens": round(total_input_tokens / len(entries), 2),
                "avg_output_tokens": round(total_output_tokens / len(entries), 2),
                "total_cache_creation_input_tokens": total_cache_creation_input_tokens,
                "total_cache_read_input_tokens": total_cache_read_input_tokens,
                "cache_create_read_diff": total_cache_read_input_tokens - total_cache_creation_input_tokens,
                "cache_create_read_ratio": format_ratio(
                    total_cache_creation_input_tokens, total_cache_read_input_tokens
                ),
                "cache_read_input_ratio": format_ratio(total_cache_read_input_tokens, total_input_tokens),
            }
        else:
            total_input_tokens = sum(entry["usage"]["input_tokens"] for entry in entries)
            total_output_tokens = sum(entry["usage"]["output_tokens"] for entry in entries)
            total_tokens = sum(entry["usage"]["total_tokens"] for entry in entries)
            total_cached_tokens = sum(entry["usage"]["cached_tokens"] for entry in entries)
            total_reasoning_tokens = sum(entry["usage"]["reasoning_tokens"] for entry in entries)

            return {
                "count": len(entries),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_input_tokens": round(total_input_tokens / len(entries), 2),
                "avg_output_tokens": round(total_output_tokens / len(entries), 2),
                "total_tokens": total_tokens,
                "total_cached_tokens": total_cached_tokens,
                "cached_input_ratio": format_ratio(total_cached_tokens, total_input_tokens),
                "total_reasoning_tokens": total_reasoning_tokens,
            }
    except (KeyError, TypeError) as e:
        print(f"Error processing entries: {e}")
        return {}


def create_stats_table(stats: dict[str, Any], model_name: str) -> Table:
    """Create a formatted table for usage statistics."""
    table = Table(title=f"Usage Statistics for {model_name}", show_header=True, header_style="bold magenta")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    # Common metrics for all models
    common_metrics = [
        ("Total Requests", f"{stats['count']:,}"),
        ("Total Input Tokens", f"{stats['total_input_tokens']:,}"),
        ("Total Output Tokens", f"{stats['total_output_tokens']:,}"),
        ("Average Input Tokens", f"{stats['avg_input_tokens']:f}"),
        ("Average Output Tokens", f"{stats['avg_output_tokens']:f}"),
    ]

    for metric, value in common_metrics:
        table.add_row(metric, value)

    # Model-specific metrics
    if "claude" in model_name.lower():
        claude_metrics = [
            ("Cache Creation Input Tokens", f"{stats['total_cache_creation_input_tokens']:,}"),
            ("Cache Read Input Tokens", f"{stats['total_cache_read_input_tokens']:,}"),
            ("Cache Create/Read Ratio", stats["cache_create_read_ratio"]),
        ]
        for metric, value in claude_metrics:
            table.add_row(metric, value)
    else:
        gpt_metrics = [
            ("Total Tokens", f"{stats['total_tokens']:,}"),
            ("Total Cached Tokens", f"{stats['total_cached_tokens']:,}"),
            ("Cached/Input Ratio", stats["cached_input_ratio"]),
            ("Total Reasoning Tokens", f"{stats['total_reasoning_tokens']:,}"),
        ]
        for metric, value in gpt_metrics:
            table.add_row(metric, value)

    return table


def analyze_model_usage(usage_file: str, model_name: str, console: Console):
    """
    Analyze usage for a specific model with enhanced visual output.
    """
    # Get entries for last 24 hours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    # Get all entries for the model
    entries = analyze_usage(usage_file, model=model_name, start_date=start_date, end_date=end_date)

    # Calculate statistics
    all_stats = get_usage_statistics(model_name, entries)

    if not all_stats:
        console.print(f"[red]No valid statistics found for model {model_name}[/red]")
        return

    # Create and display the detailed stats table
    stats_table = create_stats_table(all_stats, model_name)
    console.print(stats_table)

    # Display raw stats in a collapsible panel for debugging
    raw_stats = Text(json.dumps(all_stats, indent=2), style="dim")
    console.print(Panel(raw_stats, title="Raw Statistics", subtitle="(Detailed Debug Information)", expand=False))


def main():
    """Main function to run the usage analysis."""
    console = Console()
    usage_file = "logs/usage/usage.jsonl"

    console.print("[bold blue]Running usage analysis...[/bold blue]")
    console.print("=" * 80)

    # Analyze for each model
    for model in ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview", "claude-3-5-sonnet-20241022"]:
        analyze_model_usage(usage_file, model, console)
        console.print("-" * 80)  # Separator between models


if __name__ == "__main__":
    main()
