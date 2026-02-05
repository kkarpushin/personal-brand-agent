"""Pipeline run tracking with structured stage-level timing.

``PipelineRunLogger`` wraps the global ``AgentLogger`` and provides a
higher-level API for tracking the stages of a single pipeline execution:

1. Instantiate with a ``run_id`` -- this sets the logger context.
2. Call ``start_stage()`` / ``end_stage()`` around each pipeline step.
3. Call ``finish()`` when the pipeline is done -- returns a summary dict.
4. Call ``get_summary_text()`` for a human-readable summary.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.logging.agent_logger import get_logger
from src.logging.models import LogComponent


class PipelineRunLogger:
    """Track an entire pipeline run with per-stage timing and status.

    Parameters:
        run_id: Unique identifier for this pipeline execution.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.logger = get_logger()
        self.logger.set_context(run_id=run_id)

        self.start_time = datetime.now()
        self.stages: List[Dict[str, Any]] = []
        self.current_stage: Optional[str] = None

    async def start_stage(self, stage: str) -> None:
        """Mark the beginning of a pipeline stage.

        Args:
            stage: Human-readable name of the stage (e.g. ``"trend_scout"``).
        """
        self.current_stage = stage
        self.stages.append(
            {
                "stage": stage,
                "start": datetime.now(),
                "end": None,
                "status": "running",
                "duration_ms": None,
                "data": None,
            }
        )

        await self.logger.info(
            LogComponent.ORCHESTRATOR,
            f"Stage started: {stage}",
        )

    async def end_stage(
        self,
        status: str = "success",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark the end of the current pipeline stage.

        Args:
            status: Outcome string (``"success"``, ``"failed"``, etc.).
            data: Optional payload with stage-specific metrics.
        """
        if not self.stages:
            return

        stage = self.stages[-1]
        stage["end"] = datetime.now()
        stage["status"] = status
        stage["duration_ms"] = int(
            (stage["end"] - stage["start"]).total_seconds() * 1000
        )
        stage["data"] = data

        await self.logger.info(
            LogComponent.ORCHESTRATOR,
            f"Stage completed: {self.current_stage} ({status})",
            duration_ms=stage["duration_ms"],
        )

    async def finish(self, status: str = "success") -> Dict[str, Any]:
        """Finish the pipeline run and return a summary dict.

        Also clears the run context on the global logger.

        Args:
            status: Overall run outcome (``"success"``, ``"failed"``, etc.).

        Returns:
            Dictionary containing run_id, status, timings, and per-stage data.
        """
        end_time = datetime.now()
        total_duration_ms = int(
            (end_time - self.start_time).total_seconds() * 1000
        )

        # Serialise stage data for JSON compatibility
        serialisable_stages = []
        for s in self.stages:
            serialisable_stages.append(
                {
                    "stage": s["stage"],
                    "start": s["start"].isoformat() if s["start"] else None,
                    "end": s["end"].isoformat() if s["end"] else None,
                    "status": s["status"],
                    "duration_ms": s["duration_ms"],
                    "data": s["data"],
                }
            )

        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_ms": total_duration_ms,
            "stages": serialisable_stages,
        }

        await self.logger.info(
            LogComponent.ORCHESTRATOR,
            f"Pipeline run completed: {status}",
            data=summary,
            duration_ms=total_duration_ms,
        )

        self.logger.clear_context()

        return summary

    def get_summary_text(self) -> str:
        """Return a human-readable summary of the pipeline run.

        Suitable for Telegram messages or console output.
        """
        lines: List[str] = [
            f"Pipeline Run: {self.run_id}",
            "",
        ]

        for stage in self.stages:
            status_marker = (
                "[OK]" if stage["status"] == "success" else "[FAIL]"
            )
            duration = stage.get("duration_ms") or 0
            lines.append(f"{status_marker} {stage['stage']}: {duration}ms")

        total = sum(s.get("duration_ms") or 0 for s in self.stages)
        lines.append(f"\nTotal: {total}ms")

        return "\n".join(lines)
