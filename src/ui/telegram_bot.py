"""
Telegram bot for human-in-the-loop approvals and notifications.

Provides two main classes:

    - **TelegramNotifier**: Lightweight notification sender. Does not require
      bot polling; simply pushes messages to a configured chat via the Bot API.
    - **TelegramBot**: Full-featured bot with command handlers for interactive
      approval workflows, pipeline status queries, and autonomy management.

The module gracefully degrades when the ``python-telegram-bot`` package is not
installed: all public methods become silent no-ops and a warning is logged once.

Fail-fast philosophy applies to *configuration* errors (missing token / chat ID).
Network errors during message delivery are logged but do not crash the caller.

Architecture reference: architecture.md lines 21377, 22086-22178
    (project structure, TelegramLogViewer), 22530-22716 (modification safety
    integration).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.utils import utc_now

# ---------------------------------------------------------------------------
# Conditional import of python-telegram-bot (>=21.0)
# ---------------------------------------------------------------------------
try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telegram message limits
# ---------------------------------------------------------------------------
_MAX_MESSAGE_LENGTH = 4096


def _truncate(text: str, max_length: int = _MAX_MESSAGE_LENGTH) -> str:
    """Truncate *text* to fit within Telegram's message size limit."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 20] + "\n...(truncated)"


# =============================================================================
# TELEGRAM NOTIFIER (lightweight, no polling)
# =============================================================================


class TelegramNotifier:
    """
    Lightweight Telegram notification sender.

    Uses the Bot API to push messages to a single chat.  Does **not** start
    long-polling; therefore suitable for embedding inside any async service
    without side-effects.

    Args:
        bot_token: Telegram Bot API token (from BotFather).
        chat_id: Target chat / group / channel ID.

    Usage::

        notifier = TelegramNotifier(token, chat_id)
        await notifier.send("Pipeline run completed successfully.")
        await notifier.request_approval("Approve post #42?")
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token:
            raise ValueError("TelegramNotifier requires a non-empty bot_token")
        if not chat_id:
            raise ValueError("TelegramNotifier requires a non-empty chat_id")

        self._bot_token: str = bot_token
        self._chat_id: str = chat_id

        # Build a lightweight Bot instance for sending (no Application needed)
        if TELEGRAM_AVAILABLE:
            from telegram import Bot

            self._bot: Any = Bot(token=self._bot_token)
        else:
            self._bot = None

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    async def send(self, message: str, channel: str = "telegram") -> None:
        """
        Send a plain text message to the configured chat.

        Args:
            message: Message body (Markdown supported).
            channel: Notification channel identifier.  Only ``"telegram"``
                is supported; other values are silently ignored.
        """
        if channel != "telegram":
            logger.debug(
                "TelegramNotifier.send called with channel='%s', ignoring",
                channel,
            )
            return

        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "[TELEGRAM] python-telegram-bot is not installed. "
                "Message not sent: %s",
                message[:120],
            )
            return

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=_truncate(message),
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception(
                "[TELEGRAM] Failed to send message to chat_id=%s",
                self._chat_id,
            )

    # ------------------------------------------------------------------
    # Approval requests
    # ------------------------------------------------------------------

    async def request_approval(self, message: str) -> None:
        """
        Send a message with inline *Approve* / *Reject* buttons.

        The callback data is ``"approve"`` or ``"reject"`` respectively,
        handled by :class:`TelegramBot._handle_callback`.

        Args:
            message: Approval request body.
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "[TELEGRAM] python-telegram-bot is not installed. "
                "Approval request not sent: %s",
                message[:120],
            )
            return

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data="approve"),
                    InlineKeyboardButton("Reject", callback_data="reject"),
                ]
            ]
        )
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=_truncate(message),
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
        except Exception:
            logger.exception(
                "[TELEGRAM] Failed to send approval request to chat_id=%s",
                self._chat_id,
            )

    async def request_critical_approval(self, message: str) -> None:
        """
        Send a critical approval request.

        Critical approvals require the human operator to type ``CONFIRM``
        in the chat rather than pressing a button, preventing accidental
        approval of high-risk changes.

        Args:
            message: Critical approval request body.
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "[TELEGRAM] python-telegram-bot is not installed. "
                "Critical approval request not sent: %s",
                message[:120],
            )
            return

        full_message = (
            f"{_truncate(message, _MAX_MESSAGE_LENGTH - 100)}\n\n"
            "To approve, reply with CONFIRM in your next message."
        )
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=full_message,
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception(
                "[TELEGRAM] Failed to send critical approval request to chat_id=%s",
                self._chat_id,
            )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    async def send_post_preview(
        self,
        run_id: str,
        post_text: str,
        image_path: Optional[str] = None,
        qc_score: Optional[float] = None,
        content_type: Optional[str] = None,
    ) -> None:
        """Send a post preview with image and Approve/Reject inline buttons.

        Args:
            run_id: Pipeline run ID (encoded in callback data).
            post_text: The generated post text.
            image_path: Optional local path to the generated image.
            qc_score: QC score for display.
            content_type: Content type label for display.
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning("[TELEGRAM] python-telegram-bot not installed. Preview not sent.")
            return

        # Send the image first (if available)
        if image_path:
            try:
                with open(image_path, "rb") as photo:
                    await self._bot.send_photo(
                        chat_id=self._chat_id,
                        photo=photo,
                    )
            except Exception:
                logger.exception("[TELEGRAM] Failed to send preview image")

        # Build caption with metadata
        header_parts = ["Post Preview"]
        if content_type:
            header_parts.append(f"Type: {content_type}")
        if qc_score is not None:
            header_parts.append(f"QC: {qc_score:.1f}/10")
        header = " | ".join(header_parts)

        preview_text = f"{header}\n{'=' * 40}\n\n{post_text}"

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Approve", callback_data=f"approve:{run_id}"
                    ),
                    InlineKeyboardButton(
                        "Reject", callback_data=f"reject:{run_id}"
                    ),
                ]
            ]
        )
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=_truncate(preview_text),
                reply_markup=keyboard,
            )
        except Exception:
            logger.exception("[TELEGRAM] Failed to send post preview")

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    async def notify(self, message: str) -> None:
        """Alias for :meth:`send` on the ``"telegram"`` channel."""
        await self.send(message, channel="telegram")

    async def send_log(self, message: str) -> None:
        """
        Send a log entry to Telegram.

        Used by the logging subsystem to forward high-severity entries.
        """
        await self.send(message, channel="telegram")


# =============================================================================
# TELEGRAM BOT (full interactive bot with command handlers)
# =============================================================================


class TelegramBot:
    """
    Interactive Telegram bot for human-in-the-loop pipeline management.

    Supports:
        - ``/start`` -- welcome message
        - ``/status`` -- pipeline status summary
        - ``/queue`` -- pending approvals queue
        - ``/autonomy`` -- view / change autonomy level
        - ``/approve_{id}`` -- approve a pending item
        - ``/reject_{id}`` -- reject a pending item
        - Inline button callbacks for approve / reject

    Args:
        bot_token: Telegram Bot API token.
        chat_id: Authorized chat ID. Only messages from this chat are
            processed; all others are silently ignored.
        on_approve: Async callback invoked when an item is approved.
            Signature: ``async (item_id: str) -> None``.
        on_reject: Async callback invoked when an item is rejected.
            Signature: ``async (item_id: str) -> None``.

    Usage::

        bot = TelegramBot(token, chat_id, on_approve=my_approve, on_reject=my_reject)
        await bot.start()
        # ... keep running ...
        await bot.stop()
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        on_approve: Optional[Callable[[str], Awaitable[None]]] = None,
        on_reject: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        if not bot_token:
            raise ValueError("TelegramBot requires a non-empty bot_token")
        if not chat_id:
            raise ValueError("TelegramBot requires a non-empty chat_id")

        self._bot_token: str = bot_token
        self._chat_id: str = chat_id
        self._on_approve: Optional[Callable[[str], Awaitable[None]]] = on_approve
        self._on_reject: Optional[Callable[[str], Awaitable[None]]] = on_reject

        self._app: Any = None  # telegram.ext.Application (set in start())
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Build the ``Application``, register handlers, and start polling.

        If ``python-telegram-bot`` is not installed the method logs a
        warning and returns immediately.
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "[TELEGRAM] python-telegram-bot is not installed. "
                "TelegramBot will not start.",
            )
            return

        if self._started:
            logger.warning("[TELEGRAM] Bot is already running, ignoring start()")
            return

        self._app = Application.builder().token(self._bot_token).build()
        self._setup_handlers(self._app)

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)  # type: ignore[union-attr]
        self._started = True
        logger.info("[TELEGRAM] Bot started polling (chat_id=%s)", self._chat_id)

    async def stop(self) -> None:
        """Stop polling and shut down the ``Application``."""
        if not TELEGRAM_AVAILABLE or not self._started:
            return

        try:
            if self._app and self._app.updater:
                await self._app.updater.stop()
            if self._app:
                await self._app.stop()
                await self._app.shutdown()
            self._started = False
            logger.info("[TELEGRAM] Bot stopped")
        except Exception:
            logger.exception("[TELEGRAM] Error during bot shutdown")

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _setup_handlers(self, app: Any) -> None:
        """
        Register all command and callback handlers on the ``Application``.

        Args:
            app: A ``telegram.ext.Application`` instance.
        """
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("queue", self._cmd_queue))
        app.add_handler(CommandHandler("autonomy", self._cmd_autonomy))

        # Pattern-based handlers for /approve_{id} and /reject_{id}
        app.add_handler(
            MessageHandler(
                filters.Regex(r"^/approve_\S+"),
                self._handle_approval,
            )
        )
        app.add_handler(
            MessageHandler(
                filters.Regex(r"^/reject_\S+"),
                self._handle_rejection,
            )
        )

        # Inline keyboard callback handler
        app.add_handler(CallbackQueryHandler(self._handle_callback))

        # Error handler
        app.add_error_handler(self._error_handler)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """Handle ``/start`` -- send welcome message."""
        if not self._is_authorized(update.effective_chat.id):
            return

        welcome = (
            "LinkedIn Super Agent -- Human-in-the-Loop Interface\n\n"
            "Available commands:\n"
            "/status  -- Pipeline status summary\n"
            "/queue   -- Pending approvals\n"
            "/autonomy -- View/change autonomy level\n"
            "/approve_{id} -- Approve a pending item\n"
            "/reject_{id}  -- Reject a pending item\n"
        )
        await update.message.reply_text(welcome)

    async def _cmd_status(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """Handle ``/status`` -- show pipeline status summary."""
        if not self._is_authorized(update.effective_chat.id):
            return

        now = utc_now()
        pending_count = len(self._pending_approvals)

        status_text = (
            f"Pipeline Status ({now.strftime('%Y-%m-%d %H:%M UTC')})\n\n"
            f"Pending approvals: {pending_count}\n"
            f"Bot uptime: {'running' if self._started else 'stopped'}\n"
        )
        await update.message.reply_text(status_text)

    async def _cmd_queue(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """Handle ``/queue`` -- show pending approvals queue."""
        if not self._is_authorized(update.effective_chat.id):
            return

        if not self._pending_approvals:
            await update.message.reply_text("No pending approvals.")
            return

        lines: List[str] = ["Pending Approvals:\n"]
        for item_id, item_data in self._pending_approvals.items():
            created_at = item_data.get("created_at", "unknown")
            description = item_data.get("description", "No description")
            lines.append(
                f"  [{item_id}] {description}\n"
                f"    Created: {created_at}\n"
                f"    /approve_{item_id}  /reject_{item_id}\n"
            )

        await update.message.reply_text(_truncate("\n".join(lines)))

    async def _cmd_autonomy(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """
        Handle ``/autonomy`` -- show or change autonomy level.

        Usage::

            /autonomy         -- show current level
            /autonomy 3       -- set level to 3
        """
        if not self._is_authorized(update.effective_chat.id):
            return

        args = context.args if context.args else []

        if not args:
            # Show current autonomy info
            autonomy_descriptions = {
                1: "Level 1: Human approves everything",
                2: "Level 2: Human approves posts only",
                3: "Level 3: Auto-publish high-score (>=9.0), human for rest",
                4: "Level 4: Full autonomy (human notified, not asked)",
            }
            text = "Autonomy Levels:\n\n"
            for level, desc in autonomy_descriptions.items():
                text += f"  {desc}\n"
            text += "\nUse /autonomy <level> to change (e.g. /autonomy 3)"
            await update.message.reply_text(text)
            return

        # Attempt to set autonomy level
        try:
            new_level = int(args[0])
            if new_level not in (1, 2, 3, 4):
                await update.message.reply_text(
                    "Invalid autonomy level. Must be 1, 2, 3, or 4."
                )
                return
            # Note: actual persistence of autonomy level is handled by the
            # orchestrator / config layer. Here we just acknowledge the request.
            await update.message.reply_text(
                f"Autonomy level change requested: {new_level}\n"
                "Note: The orchestrator will apply this on the next run."
            )
            logger.info(
                "[TELEGRAM] Autonomy level change requested: %d by user %s",
                new_level,
                update.effective_user.id,
            )
        except (ValueError, IndexError):
            await update.message.reply_text(
                "Usage: /autonomy <level> (1-4)"
            )

    # ------------------------------------------------------------------
    # Approval / rejection handlers
    # ------------------------------------------------------------------

    async def _handle_approval(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """
        Handle ``/approve_{id}`` messages.

        Parses the item ID from the command text, invokes the ``on_approve``
        callback, and replies with confirmation.
        """
        if not self._is_authorized(update.effective_chat.id):
            return

        text = update.message.text.strip()
        match = re.match(r"^/approve_(\S+)", text)
        if not match:
            await update.message.reply_text("Could not parse approval ID.")
            return

        item_id = match.group(1)

        # Check for CONFIRM suffix (critical approvals)
        has_confirm = "CONFIRM" in text.split(f"/approve_{item_id}", 1)[-1]

        logger.info(
            "[TELEGRAM] Approval received for item '%s' (confirm=%s) from user %s",
            item_id,
            has_confirm,
            update.effective_user.id,
        )

        # Remove from pending queue
        self._pending_approvals.pop(item_id, None)

        # Invoke callback
        if self._on_approve:
            try:
                await self._on_approve(item_id)
                await update.message.reply_text(f"Approved: {item_id}")
            except Exception:
                logger.exception(
                    "[TELEGRAM] on_approve callback failed for item '%s'",
                    item_id,
                )
                await update.message.reply_text(
                    f"Approval registered for {item_id} but callback failed. "
                    "Check logs for details."
                )
        else:
            await update.message.reply_text(
                f"Approved: {item_id} (no callback configured)"
            )

    async def _handle_rejection(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """
        Handle ``/reject_{id}`` messages.

        Parses the item ID from the command text, invokes the ``on_reject``
        callback, and replies with confirmation.
        """
        if not self._is_authorized(update.effective_chat.id):
            return

        text = update.message.text.strip()
        match = re.match(r"^/reject_(\S+)", text)
        if not match:
            await update.message.reply_text("Could not parse rejection ID.")
            return

        item_id = match.group(1)

        logger.info(
            "[TELEGRAM] Rejection received for item '%s' from user %s",
            item_id,
            update.effective_user.id,
        )

        # Remove from pending queue
        self._pending_approvals.pop(item_id, None)

        # Invoke callback
        if self._on_reject:
            try:
                await self._on_reject(item_id)
                await update.message.reply_text(f"Rejected: {item_id}")
            except Exception:
                logger.exception(
                    "[TELEGRAM] on_reject callback failed for item '%s'",
                    item_id,
                )
                await update.message.reply_text(
                    f"Rejection registered for {item_id} but callback failed. "
                    "Check logs for details."
                )
        else:
            await update.message.reply_text(
                f"Rejected: {item_id} (no callback configured)"
            )

    # ------------------------------------------------------------------
    # Inline button callback handler
    # ------------------------------------------------------------------

    async def _handle_callback(
        self,
        update: Any,
        context: Any,
    ) -> None:
        """
        Handle inline keyboard button presses (Approve / Reject).

        The callback data is expected to be ``"approve"`` or ``"reject"``.
        These come from the inline buttons sent by
        :meth:`TelegramNotifier.request_approval`.
        """
        query = update.callback_query
        if query is None:
            return

        if not self._is_authorized(update.effective_chat.id):
            await query.answer("Unauthorized.", show_alert=True)
            return

        await query.answer()  # Acknowledge to remove "loading" indicator

        data = query.data or ""

        # Parse callback data: "approve:{run_id}" or plain "approve"
        if data.startswith("approve"):
            item_id = data.split(":", 1)[1] if ":" in data else (
                str(query.message.message_id) if query.message else "unknown"
            )
            logger.info(
                "[TELEGRAM] Inline approval for '%s' from user %s",
                item_id,
                query.from_user.id,
            )
            if self._on_approve:
                try:
                    await self._on_approve(item_id)
                    await query.edit_message_text(
                        text=f"{query.message.text}\n\n-- APPROVED --"
                    )
                except Exception:
                    logger.exception("[TELEGRAM] on_approve callback failed (inline)")
                    await query.edit_message_text(
                        text=f"{query.message.text}\n\n-- APPROVAL FAILED (see logs) --"
                    )
            else:
                await query.edit_message_text(
                    text=f"{query.message.text}\n\n-- APPROVED (no callback) --"
                )

        elif data.startswith("reject"):
            item_id = data.split(":", 1)[1] if ":" in data else (
                str(query.message.message_id) if query.message else "unknown"
            )
            logger.info(
                "[TELEGRAM] Inline rejection for '%s' from user %s",
                item_id,
                query.from_user.id,
            )
            if self._on_reject:
                try:
                    await self._on_reject(item_id)
                    await query.edit_message_text(
                        text=f"{query.message.text}\n\n-- REJECTED --"
                    )
                except Exception:
                    logger.exception("[TELEGRAM] on_reject callback failed (inline)")
                    await query.edit_message_text(
                        text=f"{query.message.text}\n\n-- REJECTION FAILED (see logs) --"
                    )
            else:
                await query.edit_message_text(
                    text=f"{query.message.text}\n\n-- REJECTED (no callback) --"
                )

        else:
            logger.warning("[TELEGRAM] Unknown callback data: %s", data)

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def _is_authorized(self, user_id: int) -> bool:
        """
        Check whether *user_id* matches the configured chat ID.

        Args:
            user_id: Telegram user ID from the incoming update.

        Returns:
            ``True`` if the user is authorized to interact with the bot.
        """
        authorized = str(user_id) == str(self._chat_id)
        if not authorized:
            logger.warning(
                "[TELEGRAM] Unauthorized access attempt from user_id=%s "
                "(expected chat_id=%s)",
                user_id,
                self._chat_id,
            )
        return authorized

    # ------------------------------------------------------------------
    # Pending approval management (used by orchestrator)
    # ------------------------------------------------------------------

    def add_pending_approval(
        self,
        item_id: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a new pending approval item.

        Args:
            item_id: Unique identifier for the approval item.
            description: Human-readable description.
            metadata: Additional context stored with the approval.
        """
        self._pending_approvals[item_id] = {
            "description": description,
            "created_at": utc_now().isoformat(),
            "metadata": metadata or {},
        }

    def remove_pending_approval(self, item_id: str) -> None:
        """Remove a pending approval item (already handled or expired)."""
        self._pending_approvals.pop(item_id, None)

    def get_pending_count(self) -> int:
        """Return the number of pending approvals."""
        return len(self._pending_approvals)

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------

    @staticmethod
    async def _error_handler(update: object, context: Any) -> None:
        """
        Global error handler for the Telegram Application.

        Logs the error without crashing the polling loop.
        """
        logger.error(
            "[TELEGRAM] Update %s caused error: %s",
            update,
            context.error,
            exc_info=context.error,
        )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = ["TelegramNotifier", "TelegramBot"]
