"""Central logger with multiple outputs: file, Supabase, Telegram.

Provides the ``AgentLogger`` class that dispatches structured log entries
to local JSON files (via ``aiofiles``), an optional Supabase table, and
an optional Telegram notifier.  A lightweight in-memory ring buffer
allows fast ``get_recent()`` queries without hitting the database.

Global helpers:
    - ``init_logger()``  -- create and register a singleton ``AgentLogger``
    - ``get_logger()``   -- retrieve the singleton (raises if not initialised)
"""

import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import aiofiles

from src.logging.models import LogComponent, LogEntry, LogLevel
from src.utils import utc_now


class AgentLogger:
    """Central logging system for all agents.

    Supports multiple outputs: file, Supabase, Telegram.

    Parameters:
        log_dir: Directory for log files (created if missing).
        supabase_client: Optional Supabase DB client with ``insert()`` method.
        telegram_notifier: Optional notifier with ``send_log()`` method.
        min_level: Minimum level for Supabase writes.
        telegram_min_level: Minimum level for Telegram notifications.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        supabase_client: Any = None,
        telegram_notifier: Any = None,
        min_level: LogLevel = LogLevel.INFO,
        telegram_min_level: LogLevel = LogLevel.WARNING,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.supabase = supabase_client
        self.telegram = telegram_notifier
        self.min_level = min_level
        self.telegram_min_level = telegram_min_level

        # Current context (set per pipeline run)
        self._run_id: Optional[str] = None
        self._post_id: Optional[str] = None

        # Log file paths
        self._main_log = self.log_dir / "agent.log"
        self._error_log = self.log_dir / "errors.log"
        self._debug_log = self.log_dir / "debug.log"

        # In-memory ring buffer for quick access
        self._recent_logs: List[LogEntry] = []
        self._max_recent: int = 1000

        # Custom handlers registered via add_handler()
        self._handlers: List[Callable[[LogEntry], None]] = []

        # Track pending async tasks to prevent garbage collection
        self._pending_tasks: Set["asyncio.Task[None]"] = set()

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(
        self, run_id: Optional[str] = None, post_id: Optional[str] = None
    ) -> None:
        """Set context for subsequent log entries."""
        if run_id is not None:
            self._run_id = run_id
        if post_id is not None:
            self._post_id = post_id

    def clear_context(self) -> None:
        """Clear logging context."""
        self._run_id = None
        self._post_id = None

    # ------------------------------------------------------------------
    # Custom handler registration
    # ------------------------------------------------------------------

    def add_handler(self, handler: Callable[[LogEntry], None]) -> None:
        """Register a custom synchronous log handler."""
        self._handlers.append(handler)

    # ------------------------------------------------------------------
    # Core log method
    # ------------------------------------------------------------------

    async def log(
        self,
        level: LogLevel,
        component: LogComponent,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Log a structured message.

        Writes to all configured outputs (file always, Supabase and Telegram
        when connected and the severity threshold is met).
        """
        entry = LogEntry(
            timestamp=utc_now(),
            level=level,
            component=component,
            message=message,
            run_id=self._run_id,
            post_id=self._post_id,
            data=data or {},
            duration_ms=duration_ms,
        )

        if error is not None:
            entry.error_type = type(error).__name__
            entry.error_traceback = traceback.format_exc()

        # Append to ring buffer
        self._recent_logs.append(entry)
        if len(self._recent_logs) > self._max_recent:
            self._recent_logs.pop(0)

        # Write to file (always, awaited so file I/O completes before return)
        await self._write_to_file(entry)

        # Write to Supabase (fire-and-forget but tracked)
        if self.supabase and level.value >= self.min_level.value:
            task = asyncio.create_task(self._write_to_supabase(entry))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        # Send to Telegram for important logs (fire-and-forget but tracked)
        if self.telegram and level.value >= self.telegram_min_level.value:
            task = asyncio.create_task(self._send_to_telegram(entry))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        # Invoke custom handlers (synchronous, errors swallowed)
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass  # Never let a handler break logging

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def debug(
        self, component: LogComponent, message: str, **kwargs: Any
    ) -> None:
        """Log at DEBUG level."""
        await self.log(LogLevel.DEBUG, component, message, **kwargs)

    async def info(
        self, component: LogComponent, message: str, **kwargs: Any
    ) -> None:
        """Log at INFO level."""
        await self.log(LogLevel.INFO, component, message, **kwargs)

    async def warning(
        self, component: LogComponent, message: str, **kwargs: Any
    ) -> None:
        """Log at WARNING level."""
        await self.log(LogLevel.WARNING, component, message, **kwargs)

    async def error(
        self, component: LogComponent, message: str, **kwargs: Any
    ) -> None:
        """Log at ERROR level."""
        await self.log(LogLevel.ERROR, component, message, **kwargs)

    async def critical(
        self, component: LogComponent, message: str, **kwargs: Any
    ) -> None:
        """Log at CRITICAL level."""
        await self.log(LogLevel.CRITICAL, component, message, **kwargs)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_recent(
        self,
        limit: int = 20,
        level: Optional[LogLevel] = None,
        component: Optional[LogComponent] = None,
        run_id: Optional[str] = None,
    ) -> List[LogEntry]:
        """Return recent logs from the in-memory ring buffer.

        Filters are applied in-memory (fast, no I/O).
        """
        logs = self._recent_logs.copy()

        if level is not None:
            logs = [entry for entry in logs if entry.level == level]
        if component is not None:
            logs = [entry for entry in logs if entry.component == component]
        if run_id is not None:
            logs = [entry for entry in logs if entry.run_id == run_id]

        return logs[-limit:]

    async def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        component: Optional[LogComponent] = None,
        run_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Query logs from Supabase.

        Falls back to the in-memory buffer when Supabase is not configured.
        """
        if not self.supabase:
            return self.get_recent(limit, level, component, run_id)

        query = self.supabase.from_("agent_logs").select("*")

        if start_time:
            query = query.gte("timestamp", start_time.isoformat())
        if end_time:
            query = query.lte("timestamp", end_time.isoformat())
        if level:
            query = query.eq("level", level.value)
        if component:
            query = query.eq("component", component.value)
        if run_id:
            query = query.eq("run_id", run_id)
        if search:
            query = query.ilike("message", f"%{search}%")

        query = query.order("timestamp", desc=True).limit(limit)

        result = await query.execute()

        return [
            LogEntry(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                level=LogLevel(row["level"]),
                component=LogComponent(row["component"]),
                message=row["message"],
                run_id=row.get("run_id"),
                post_id=row.get("post_id"),
                data=row.get("data") or {},
                error_type=row.get("error_type"),
                error_traceback=row.get("error_traceback"),
                duration_ms=row.get("duration_ms"),
            )
            for row in result.data
        ]

    # ------------------------------------------------------------------
    # Flush (call before shutdown)
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Wait for all pending async log tasks (Supabase, Telegram).

        Call this before application shutdown to ensure every log entry
        has been written.
        """
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

    # ------------------------------------------------------------------
    # Private output methods
    # ------------------------------------------------------------------

    async def _write_to_file(self, entry: LogEntry) -> None:
        """Write log entry to JSON log files using async I/O.

        - ``agent.log``  -- all entries
        - ``errors.log`` -- ERROR and CRITICAL only
        - ``debug.log``  -- DEBUG only
        """
        json_line = entry.to_json() + "\n"

        # Always write to main log
        async with aiofiles.open(self._main_log, "a", encoding="utf-8") as f:
            await f.write(json_line)

        # Write errors to separate file
        if entry.level.value >= LogLevel.ERROR.value:
            async with aiofiles.open(self._error_log, "a", encoding="utf-8") as f:
                await f.write(json_line)

        # Write debug to separate file
        if entry.level == LogLevel.DEBUG:
            async with aiofiles.open(self._debug_log, "a", encoding="utf-8") as f:
                await f.write(json_line)

    async def _write_to_supabase(self, entry: LogEntry) -> None:
        """Write log entry to the ``agent_logs`` Supabase table."""
        try:
            await self.supabase.insert("agent_logs", entry.to_dict())
        except Exception as exc:
            # Last-resort fallback: print to stderr so we never lose
            # visibility into a Supabase write failure.
            print(
                f"[LOGGING] Failed to write to Supabase: {exc}",
                file=sys.stderr,
            )

    async def _send_to_telegram(self, entry: LogEntry) -> None:
        """Send an important log entry to Telegram."""
        try:
            await self.telegram.send_log(entry.to_readable())
        except Exception:
            pass  # Never fail on Telegram errors


# ======================================================================
# GLOBAL LOGGER SINGLETON
# ======================================================================

_logger: Optional[AgentLogger] = None


def init_logger(
    log_dir: str = "logs",
    supabase_client: Any = None,
    telegram_notifier: Any = None,
    min_level: LogLevel = LogLevel.INFO,
    telegram_min_level: LogLevel = LogLevel.WARNING,
) -> AgentLogger:
    """Initialise and register the global ``AgentLogger`` singleton.

    Returns the newly created logger instance.
    """
    global _logger
    _logger = AgentLogger(
        log_dir=log_dir,
        supabase_client=supabase_client,
        telegram_notifier=telegram_notifier,
        min_level=min_level,
        telegram_min_level=telegram_min_level,
    )
    return _logger


def get_logger() -> AgentLogger:
    """Retrieve the global ``AgentLogger`` singleton.

    Raises:
        RuntimeError: If ``init_logger()`` has not been called yet.
    """
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger() first.")
    return _logger
