"""Daily summary generator for agent activity.

Queries the last 24 hours of logs and produces a formatted text report
suitable for sending via Telegram. The report includes:

- Total pipeline runs and their success/failure breakdown
- Average pipeline duration
- Error and warning counts
- Top errors grouped by component
- Activity breakdown by component
"""

from collections import Counter
from datetime import timedelta
from typing import Any, Dict, List, Optional

from src.logging.agent_logger import AgentLogger, get_logger
from src.logging.models import LogComponent, LogEntry, LogLevel
from src.utils import utc_now


async def generate_daily_digest(
    logger: Optional[AgentLogger] = None,
) -> str:
    """Generate a human-readable daily digest of agent activity.

    Queries the last 24 hours of logs from the provided (or global) logger
    and returns a formatted summary string.

    Args:
        logger: AgentLogger instance. Falls back to ``get_logger()`` if None.

    Returns:
        Formatted text summary suitable for Telegram.
    """
    if logger is None:
        logger = get_logger()

    yesterday = utc_now() - timedelta(days=1)

    # Fetch all logs from the last 24 hours
    all_logs: List[LogEntry] = await logger.query_logs(
        start_time=yesterday,
        limit=10000,
    )

    # ------------------------------------------------------------------
    # Aggregate counts
    # ------------------------------------------------------------------

    level_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    for log_entry in all_logs:
        level_counts[log_entry.level.name_str] += 1
        component_counts[log_entry.component.value] += 1

    # ------------------------------------------------------------------
    # Pipeline run statistics
    # ------------------------------------------------------------------

    run_ids = {entry.run_id for entry in all_logs if entry.run_id}
    total_runs = len(run_ids)

    # Identify pipeline completion entries to compute success/failure
    completion_logs = [
        entry
        for entry in all_logs
        if entry.component == LogComponent.ORCHESTRATOR
        and "Pipeline run completed" in entry.message
    ]

    successful_runs = sum(
        1 for entry in completion_logs if "success" in entry.message.lower()
    )
    failed_runs = sum(
        1 for entry in completion_logs if "failed" in entry.message.lower()
    )

    # Average duration from completion entries that carry duration_ms
    durations = [
        entry.duration_ms
        for entry in completion_logs
        if entry.duration_ms is not None
    ]
    avg_duration_ms = (
        int(sum(durations) / len(durations)) if durations else 0
    )

    # ------------------------------------------------------------------
    # Errors grouped by component
    # ------------------------------------------------------------------

    errors: List[LogEntry] = [
        entry
        for entry in all_logs
        if entry.level in (LogLevel.ERROR, LogLevel.CRITICAL)
    ]

    errors_by_component: Counter[str] = Counter()
    for err in errors:
        errors_by_component[err.component.value] += 1

    # ------------------------------------------------------------------
    # Build the digest text
    # ------------------------------------------------------------------

    date_str = yesterday.strftime("%d.%m.%Y")
    lines: List[str] = [
        f"Daily Digest ({date_str})",
        "",
        "Activity:",
        f"  Total log entries: {len(all_logs)}",
        f"  Pipeline runs: {total_runs}",
        f"  Successful: {successful_runs}",
        f"  Failed: {failed_runs}",
    ]

    if avg_duration_ms:
        lines.append(f"  Avg duration: {avg_duration_ms}ms")

    lines.append(f"  Errors: {level_counts.get('error', 0)}")
    lines.append(f"  Warnings: {level_counts.get('warning', 0)}")

    # Top 5 components by activity
    lines.append("")
    lines.append("By Component (top 5):")
    for comp, count in component_counts.most_common(5):
        lines.append(f"  {comp}: {count}")

    # Error details
    if errors:
        lines.append("")
        lines.append(f"Errors ({len(errors)}):")
        for err_comp, err_count in errors_by_component.most_common(5):
            lines.append(f"  [{err_comp}]: {err_count}")

        lines.append("")
        lines.append("Recent errors:")
        for err_entry in errors[:5]:
            truncated = (
                err_entry.message[:80] + "..."
                if len(err_entry.message) > 80
                else err_entry.message
            )
            lines.append(f"  [{err_entry.component.value}] {truncated}")
    else:
        lines.append("")
        lines.append("No errors!")

    return "\n".join(lines)
