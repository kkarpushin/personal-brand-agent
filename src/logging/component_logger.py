"""Per-component logger wrapper and timed-operation context manager.

``ComponentLogger`` provides a thin convenience wrapper around the global
``AgentLogger`` so that every agent or subsystem can log without
repeatedly specifying its ``LogComponent``.

``TimedOperation`` is an async context manager returned by
``ComponentLogger.timed()`` that automatically logs the start time,
elapsed duration, and success/failure of a block of code.
"""

from datetime import datetime
from typing import Any, Optional

from src.logging.agent_logger import get_logger
from src.logging.models import LogComponent


class ComponentLogger:
    """Wrapper that binds a fixed ``LogComponent`` to the global logger.

    Each agent creates its own ``ComponentLogger`` at ``__init__`` time::

        self.log = ComponentLogger(LogComponent.WRITER)
        await self.log.info("Draft generated", data={"chars": 1200})
    """

    def __init__(self, component: LogComponent) -> None:
        self.component = component

    async def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level for this component."""
        await get_logger().debug(self.component, message, **kwargs)

    async def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level for this component."""
        await get_logger().info(self.component, message, **kwargs)

    async def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level for this component."""
        await get_logger().warning(self.component, message, **kwargs)

    async def error(
        self, message: str, error: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Log at ERROR level for this component."""
        await get_logger().error(self.component, message, error=error, **kwargs)

    async def critical(
        self, message: str, error: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Log at CRITICAL level for this component."""
        await get_logger().critical(self.component, message, error=error, **kwargs)

    def timed(self, message: str) -> "TimedOperation":
        """Return an async context manager that logs start/end with duration.

        Usage::

            async with self.log.timed("Generating post"):
                draft = await self._generate(brief)
        """
        return TimedOperation(self, message)


class TimedOperation:
    """Async context manager that measures and logs operation duration.

    On entry, logs a DEBUG message (``"Starting: <message>"``).
    On successful exit, logs an INFO message with ``duration_ms``.
    On exception, logs an ERROR message with ``duration_ms`` and the error,
    then re-raises the exception (does **not** suppress it).
    """

    def __init__(self, logger: ComponentLogger, message: str) -> None:
        self.logger = logger
        self.message = message
        self.start_time: Optional[datetime] = None

    async def __aenter__(self) -> "TimedOperation":
        self.start_time = datetime.now()
        await self.logger.debug(f"Starting: {self.message}")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        assert self.start_time is not None
        duration_ms = int(
            (datetime.now() - self.start_time).total_seconds() * 1000
        )

        if exc_type is not None:
            await self.logger.error(
                f"Failed: {self.message}",
                error=exc_val if isinstance(exc_val, Exception) else None,
                duration_ms=duration_ms,
            )
        else:
            await self.logger.info(
                f"Completed: {self.message}",
                duration_ms=duration_ms,
            )
        # Return None (falsy) so exceptions propagate
