"""
Modification Safety System for the LinkedIn Super Agent.

Controls all system modifications with a risk-based approval flow.
Critical changes require human approval via Telegram, low-risk changes
are auto-applied with automatic rollback monitoring.

Provides:
    - ModificationSafetySystem: Risk-based modification approval and rollback

Architecture reference: architecture.md lines 22438-22936
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import timedelta
from typing import Any, Dict, List, Optional

from src.meta_agent.models import (
    ModificationRiskLevel,
    ModificationRequest,
    RollbackTrigger,
    modification_risk_classification,
    CONFIG_PATH_MAPPING,
    get_config_path,
)
from src.exceptions import (
    ValidationError,
    SecurityError,
    ConfigurationBackupError,
    ConfigurationCorruptedError,
    ConfigurationAccessError,
    ConfigurationWriteError,
)
from src.utils import utc_now, generate_id

logger = logging.getLogger(__name__)


# ===========================================================================
# VALID COMPONENT WHITELIST
# Components that are allowed to be modified through the safety system.
# ===========================================================================

VALID_COMPONENTS = frozenset({
    "writer",
    "trend_scout",
    "visual_creator",
    "scheduler",
    "qc",
    "humanizer",
    "analyzer",
})


# ===========================================================================
# MODIFICATION SAFETY SYSTEM
# ===========================================================================


class ModificationSafetySystem:
    """
    Controls all system modifications with risk-based approval flow.

    Risk levels and their handling:
    - **LOW**: Auto-apply, monitor for 5 posts, rollback if performance drops >15%
    - **MEDIUM**: Auto-apply with stricter rollback (3 posts, >10% drop), notify via Telegram
    - **HIGH**: Store pending, request human approval via Telegram
    - **CRITICAL**: Store pending, request critical approval with explicit confirmation

    All modifications are validated, logged, and tracked in the database.
    Expired pending modifications are auto-rejected after 24 hours.

    Args:
        db: Database client (Supabase) for storing/retrieving modifications.
        telegram_notifier: Telegram notification client for human-in-the-loop.

    Usage::

        safety = ModificationSafetySystem(db=supabase_client, telegram_notifier=tg)
        result = await safety.request_modification(mod)
        # result is "applied" or "pending_approval"

        # Periodic checks (from scheduler)
        await safety.check_rollback_triggers()
        await safety.check_expired_modifications()
        await safety.send_pending_reminders()
    """

    def __init__(self, db: Any, telegram_notifier: Any) -> None:
        self.db = db
        self.telegram = telegram_notifier

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_modification_request(self, mod: ModificationRequest) -> None:
        """
        Validate a ModificationRequest before processing.

        Checks:
        1. Required fields are present and non-empty
        2. Component is in the allowed whitelist
        3. before_state and after_state are not None

        Args:
            mod: The modification request to validate.

        Raises:
            ValidationError: If required fields are missing or invalid.
            SecurityError: If the component is not in the whitelist.
        """
        if not mod.component:
            raise ValidationError("ModificationRequest.component is required")
        if not mod.modification_type:
            raise ValidationError(
                "ModificationRequest.modification_type is required"
            )
        if mod.before_state is None:
            raise ValidationError(
                "ModificationRequest.before_state is required"
            )
        if mod.after_state is None:
            raise ValidationError(
                "ModificationRequest.after_state is required"
            )
        if not mod.reasoning:
            raise ValidationError(
                "ModificationRequest.reasoning is required"
            )

        # Validate component is in whitelist (security check)
        if mod.component not in VALID_COMPONENTS:
            raise SecurityError(f"Unknown component: {mod.component}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def request_modification(self, mod: ModificationRequest) -> str:
        """
        Process a modification request based on its risk level.

        Steps:
        1. Validate all required fields
        2. Verify risk level matches the classification (prevent spoofing)
        3. Route to appropriate handler based on risk level

        Args:
            mod: The modification request to process.

        Returns:
            ``"applied"`` if the modification was auto-applied, or
            ``"pending_approval"`` if human approval is required.

        Raises:
            ValidationError: If required fields are missing.
            SecurityError: If the component is unknown or risk level is spoofed.
        """
        self._validate_modification_request(mod)

        logger.info(
            "[MOD_SAFETY] Processing modification request: type=%s, "
            "component=%s, risk=%s",
            mod.modification_type,
            mod.component,
            mod.risk_level.value,
        )

        # Verify risk level matches classification (prevent spoofing)
        expected_risk = modification_risk_classification.get(
            mod.modification_type
        )
        if expected_risk and expected_risk != mod.risk_level:
            raise SecurityError(
                f"Risk level mismatch: claimed {mod.risk_level.value}, "
                f"expected {expected_risk.value} for {mod.modification_type}"
            )

        # Route by risk level
        if mod.risk_level == ModificationRiskLevel.LOW:
            # Auto-apply, set up monitoring for 5 posts
            await self._apply_modification(mod)
            await self._setup_rollback_trigger(
                mod, window_posts=5, threshold=0.85
            )
            logger.info(
                "[MOD_SAFETY] LOW risk modification auto-applied: %s",
                mod.modification_type,
            )
            return "applied"

        elif mod.risk_level == ModificationRiskLevel.MEDIUM:
            # Auto-apply with stricter rollback
            await self._apply_modification(mod)
            await self._setup_rollback_trigger(
                mod, window_posts=3, threshold=0.90
            )
            await self.telegram.notify(
                f"Auto-applied: {mod.modification_type}\n"
                f"Reason: {mod.reasoning}\n"
                f"Rollback if performance drops >10%"
            )
            logger.info(
                "[MOD_SAFETY] MEDIUM risk modification auto-applied with "
                "notification: %s",
                mod.modification_type,
            )
            return "applied"

        elif mod.risk_level == ModificationRiskLevel.HIGH:
            # Request human approval
            await self._store_pending_modification(mod)
            await self.telegram.request_approval(
                f"Approval needed: {mod.modification_type}\n\n"
                f"Component: {mod.component}\n"
                f"Change: {mod.reasoning}\n\n"
                f"Before:\n```{mod.before_state}```\n\n"
                f"After:\n```{mod.after_state}```\n\n"
                f"Reply /approve_{mod.id} or /reject_{mod.id}"
            )
            logger.info(
                "[MOD_SAFETY] HIGH risk modification pending approval: %s",
                mod.modification_type,
            )
            return "pending_approval"

        elif mod.risk_level == ModificationRiskLevel.CRITICAL:
            # Request approval with explicit confirmation
            await self._store_pending_modification(mod)
            await self.telegram.request_critical_approval(
                f"CRITICAL change requires approval:\n\n"
                f"{mod.modification_type}\n\n"
                f"This will modify: {mod.component}\n"
                f"Reasoning: {mod.reasoning}\n\n"
                f"Review carefully before approving.\n"
                f"Reply /approve_{mod.id} CONFIRM or /reject_{mod.id}"
            )
            logger.info(
                "[MOD_SAFETY] CRITICAL modification pending approval: %s",
                mod.modification_type,
            )
            return "pending_approval"

        # Should not reach here, but fail-fast if unknown risk level
        raise ValidationError(
            f"Unknown risk level: {mod.risk_level}"
        )

    # ------------------------------------------------------------------
    # Rollback monitoring
    # ------------------------------------------------------------------

    async def check_rollback_triggers(self) -> None:
        """
        Check if any active modification should be rolled back.

        Called after each post is published. Iterates over all modifications
        with active rollback triggers and compares recent metrics against
        the baseline. If the performance ratio drops below the trigger
        threshold, the modification is automatically rolled back.
        """
        logger.debug("[ROLLBACK_CHECK] Checking rollback triggers")

        active_mods = await self.db.get_modifications_with_active_rollback()

        for mod in active_mods:
            trigger: RollbackTrigger = mod.rollback_trigger

            recent_metrics = await self.db.get_recent_metrics(
                metric=trigger.metric,
                limit=trigger.window_posts,
            )

            if not recent_metrics:
                logger.warning(
                    "[ROLLBACK_CHECK] No recent metrics for %s, skipping",
                    mod.modification_type,
                )
                continue

            if trigger.baseline_value == 0:
                logger.warning(
                    "[ROLLBACK_CHECK] Zero baseline for %s, skipping",
                    mod.modification_type,
                )
                continue

            avg_recent = sum(recent_metrics) / len(recent_metrics)
            performance_ratio = avg_recent / trigger.baseline_value

            if performance_ratio < trigger.threshold:
                logger.warning(
                    "[ROLLBACK_CHECK] Rollback triggered for %s: "
                    "performance_ratio=%.2f < threshold=%.2f",
                    mod.modification_type,
                    performance_ratio,
                    trigger.threshold,
                )
                await self._rollback_modification(mod)
                await self.telegram.notify(
                    f"Auto-rollback triggered!\n\n"
                    f"Modification: {mod.modification_type}\n"
                    f"Reason: {trigger.metric} dropped to "
                    f"{performance_ratio:.0%} of baseline\n"
                    f"Reverted to previous state."
                )

    # ------------------------------------------------------------------
    # Expiration and reminders
    # ------------------------------------------------------------------

    async def check_expired_modifications(self) -> None:
        """
        Auto-reject expired pending modifications.

        Called from the scheduler. Any pending modification that has
        exceeded its ``expires_at`` timestamp is automatically rejected
        with a reason of "Expired - no response within 24 hours".

        This prevents modifications from hanging indefinitely.
        """
        logger.debug("[MOD_SAFETY] Checking for expired modifications")

        expired = await self.db.get_expired_pending_modifications()
        for mod in expired:
            mod.status = "auto_rejected"
            mod.rejected_reason = "Expired - no response within 24 hours"
            await self.db.update_modification(mod)
            await self.telegram.notify(
                f"Modification auto-rejected (expired):\n"
                f"Type: {mod.modification_type}\n"
                f"Component: {mod.component}"
            )
            logger.info(
                "[MOD_SAFETY] Auto-rejected expired modification: %s",
                mod.modification_type,
            )

    async def send_pending_reminders(self) -> None:
        """
        Send reminders for pending modifications that need attention.

        Called from the scheduler. For each pending modification where
        ``reminder_at`` has passed, sends a Telegram reminder and
        reschedules the next reminder for 4 hours later.
        """
        logger.debug("[MOD_SAFETY] Checking for pending reminders")

        pending = await self.db.get_pending_modifications_needing_reminder()
        now = utc_now()

        for mod in pending:
            if mod.reminder_at and now >= mod.reminder_at:
                await self.telegram.notify(
                    f"Reminder: Pending approval for "
                    f"{mod.modification_type}\n"
                    f"Component: {mod.component}\n"
                    f"Expires: {mod.expires_at}\n"
                    f"Reply /approve_{mod.id} or /reject_{mod.id}"
                )
                # Reschedule next reminder to avoid spam
                mod.reminder_at = now + timedelta(hours=4)
                await self.db.update_modification(mod)
                logger.info(
                    "[MOD_SAFETY] Sent reminder for %s, next reminder at %s",
                    mod.modification_type,
                    mod.reminder_at,
                )

    # ------------------------------------------------------------------
    # Internal: apply, rollback, store
    # ------------------------------------------------------------------

    async def _rollback_modification(self, mod: ModificationRequest) -> None:
        """
        Restore the previous state of a modification.

        Applies the ``before_state`` and updates the modification status
        to ``"rolled_back"`` in the database.

        Args:
            mod: The modification to roll back.
        """
        logger.info(
            "[MOD_SAFETY] Rolling back modification: %s on component %s",
            mod.modification_type,
            mod.component,
        )
        await self._apply_state(mod.component, mod.before_state)
        mod.status = "rolled_back"
        await self.db.update_modification(mod)

    async def _apply_modification(self, mod: ModificationRequest) -> None:
        """
        Apply a modification to the system.

        Applies the ``after_state`` and saves the modification with
        status ``"auto_applied"`` to the database.

        Args:
            mod: The modification to apply.
        """
        logger.info(
            "[MOD_SAFETY] Applying modification: %s on component %s",
            mod.modification_type,
            mod.component,
        )
        await self._apply_state(mod.component, mod.after_state)
        mod.status = "auto_applied"
        await self.db.save_modification(mod)

    async def _store_pending_modification(
        self, mod: ModificationRequest
    ) -> None:
        """
        Store a modification awaiting human approval.

        Sets the status to ``"pending"``, the expiration to 24 hours from
        now, and the first reminder to 4 hours from now.

        Args:
            mod: The modification to store as pending.
        """
        mod.status = "pending"
        mod.expires_at = utc_now() + timedelta(hours=24)
        mod.reminder_at = utc_now() + timedelta(hours=4)
        await self.db.save_modification(mod)
        logger.info(
            "[MOD_SAFETY] Stored pending modification: %s, expires_at=%s",
            mod.modification_type,
            mod.expires_at,
        )

    async def _setup_rollback_trigger(
        self,
        mod: ModificationRequest,
        window_posts: int,
        threshold: float,
    ) -> None:
        """
        Set up automatic rollback monitoring for a modification.

        Retrieves the baseline engagement rate from the last 10 posts
        and creates a ``RollbackTrigger`` that will fire if the average
        of the next ``window_posts`` drops below ``threshold`` of the
        baseline.

        Args:
            mod: The modification to monitor.
            window_posts: Number of future posts to evaluate before deciding.
            threshold: Minimum acceptable ratio of new performance to baseline
                (e.g., 0.85 means rollback if performance drops >15%).
        """
        baseline = await self.db.get_average_metric(
            metric="engagement_rate",
            last_n_posts=10,
        )

        trigger = RollbackTrigger(
            metric="engagement_rate",
            threshold=threshold,
            window_posts=window_posts,
            baseline_value=baseline,
        )

        await self.db.save_rollback_trigger(mod.id, trigger)
        logger.info(
            "[MOD_SAFETY] Rollback trigger set for %s: "
            "baseline=%.4f, threshold=%.2f, window=%d",
            mod.modification_type,
            baseline,
            threshold,
            window_posts,
        )

    # ------------------------------------------------------------------
    # State application (config file management)
    # ------------------------------------------------------------------

    async def _apply_state(self, component: str, state: dict) -> None:
        """
        Apply state changes to a component's configuration file.

        Implements an atomic write pattern:
        1. Create a backup of the current config
        2. Read current config (or start with empty dict)
        3. Merge new state keys into current config
        4. Write to a temp file, then atomic rename

        If the write fails, the backup is restored.

        Args:
            component: Component name (e.g., ``"writer"``, ``"qc"``).
            state: Dictionary of configuration keys/values to merge.

        Raises:
            ConfigurationBackupError: If backup creation fails.
            ConfigurationCorruptedError: If the existing config is invalid JSON.
            ConfigurationAccessError: If the config file cannot be read.
            ConfigurationWriteError: If the config file cannot be written.
        """
        config_path = self._get_config_path(component)
        backup_path = f"{config_path}.backup"

        logger.info(
            "[STATE_APPLY] Applying state to %s: %s",
            component,
            list(state.keys()),
        )

        # Create backup before modification
        try:
            if os.path.exists(config_path):
                shutil.copy2(config_path, backup_path)
                logger.debug(
                    "[STATE_APPLY] Created backup: %s", backup_path
                )
        except (IOError, PermissionError) as e:
            logger.error(
                "[STATE_APPLY] Failed to create backup for %s: %s",
                component,
                e,
            )
            raise ConfigurationBackupError(
                f"Cannot backup {config_path}: {e}"
            )

        # Read current config
        current_config: Dict[str, Any]
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                current_config = json.load(f)
        except FileNotFoundError:
            logger.warning(
                "[STATE_APPLY] Config not found, creating new: %s",
                config_path,
            )
            current_config = {}
        except json.JSONDecodeError as e:
            logger.error(
                "[STATE_APPLY] Corrupted config file %s: %s",
                config_path,
                e,
            )
            raise ConfigurationCorruptedError(
                f"Cannot parse {config_path}: {e}"
            )
        except PermissionError as e:
            logger.error(
                "[STATE_APPLY] Permission denied reading %s: %s",
                config_path,
                e,
            )
            raise ConfigurationAccessError(
                f"Cannot read {config_path}: {e}"
            )

        # Track changes for logging
        changes_made: List[str] = []
        for key, value in state.items():
            old_value = current_config.get(key, "<not set>")
            current_config[key] = value
            changes_made.append(f"{key}: {old_value} -> {value}")

        # Write with atomic write pattern (temp file + rename)
        try:
            temp_path = f"{config_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(current_config, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(temp_path, config_path)

            logger.info(
                "[STATE_APPLY] Successfully applied %d changes to %s",
                len(changes_made),
                component,
            )
            for change in changes_made:
                logger.debug("[STATE_APPLY]   %s", change)

        except (IOError, PermissionError, OSError) as e:
            logger.error(
                "[STATE_APPLY] Failed to write config %s: %s",
                config_path,
                e,
            )
            # Attempt to restore from backup
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, config_path)
                    logger.info(
                        "[STATE_APPLY] Restored from backup after write "
                        "failure"
                    )
                except Exception as restore_error:
                    logger.critical(
                        "[STATE_APPLY] CRITICAL: Cannot restore backup: %s",
                        restore_error,
                    )
            raise ConfigurationWriteError(
                f"Cannot write {config_path}: {e}"
            )

    def _get_config_path(self, component: str) -> str:
        """
        Get the absolute configuration file path for a component.

        Delegates to the unified ``get_config_path()`` function from
        ``src.meta_agent.models``, then resolves to an absolute path
        using the ``PROJECT_ROOT`` environment variable or ``os.getcwd()``.

        Args:
            component: Component name (e.g., ``"writer"``, ``"qc"``).

        Returns:
            Absolute path to the component's configuration file.

        Raises:
            ValidationError: If the component is not recognized.
        """
        try:
            relative_path = get_config_path(component)
        except ValueError as e:
            raise ValidationError(str(e))

        # Get project root from environment or use current working directory
        project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        return os.path.join(project_root, relative_path)


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    "ModificationSafetySystem",
    "VALID_COMPONENTS",
]
