"""
Async LinkedIn API client wrapper.

Wraps the ``tomquirk/linkedin-api`` library (Voyager API) with
``asyncio.to_thread`` to provide an async interface.  Supports 2FA via
TOTP secrets.

The orchestrator and post-analytics agents use this client to publish
posts, retrieve engagement metrics, and collect comments/reactions.

Fail-fast philosophy: ``LinkedInAPIError``, ``LinkedInRateLimitError``,
and ``LinkedInSessionExpiredError`` are raised for corresponding failure
modes.  Transient errors are retried with exponential backoff.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from src.exceptions import (
    LinkedInAPIError,
    LinkedInRateLimitError,
    LinkedInSessionExpiredError,
)
from src.utils import with_retry

logger = logging.getLogger(__name__)


class LinkedInClient:
    """Async wrapper around the ``tomquirk/linkedin-api`` Voyager client.

    Lazy-initialises the underlying synchronous ``Linkedin`` client on
    first use so that import-time side effects are avoided.

    Args:
        email: LinkedIn account email.  Falls back to ``LINKEDIN_EMAIL``.
        password: LinkedIn account password.  Falls back to
            ``LINKEDIN_PASSWORD``.
        totp_secret: TOTP secret for 2FA.  Falls back to
            ``LINKEDIN_TOTP_SECRET``.  If empty, 2FA is skipped.

    Usage::

        client = LinkedInClient()
        result = await client.publish_post("Hello LinkedIn!")
        metrics = await client.get_post_metrics(result["post_id"])
    """

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ) -> None:
        self.email: str = email or os.environ.get("LINKEDIN_EMAIL", "")
        self.password: str = password or os.environ.get("LINKEDIN_PASSWORD", "")
        self.totp_secret: str = totp_secret or os.environ.get(
            "LINKEDIN_TOTP_SECRET", ""
        )
        self._api: Any = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    async def _get_api(self) -> Any:
        """Lazy-init the ``linkedin-api`` client with optional 2FA.

        Returns:
            An authenticated ``Linkedin`` client instance.

        Raises:
            LinkedInSessionExpiredError: If login fails.
        """
        if self._api is not None:
            return self._api

        from linkedin_api import Linkedin  # type: ignore[import-untyped]

        # Generate TOTP code if secret is configured
        totp_code: Optional[str] = None
        if self.totp_secret:
            import pyotp  # type: ignore[import-untyped]

            totp_code = pyotp.TOTP(self.totp_secret).now()

        def _login() -> Any:
            try:
                kwargs: Dict[str, Any] = {}
                if totp_code:
                    kwargs["two_factor_code"] = totp_code
                return Linkedin(self.email, self.password, **kwargs)
            except Exception as exc:
                raise LinkedInSessionExpiredError(
                    f"LinkedIn login failed: {exc}"
                ) from exc

        self._api = await asyncio.to_thread(_login)
        logger.info("LinkedIn client authenticated for %s", self.email)
        return self._api

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def publish_post(
        self,
        text: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a post to LinkedIn.

        Args:
            text: Post body text.
            image_path: Optional local path to an image to attach.

        Returns:
            Dict with ``post_id`` and ``status`` keys.

        Raises:
            LinkedInRateLimitError: If the API rate-limits the request.
            LinkedInAPIError: On any other API failure.
        """
        api = await self._get_api()

        def _post() -> Any:
            try:
                if image_path:
                    return api.post(text, media_path=image_path)
                return api.post(text)
            except Exception as exc:
                error_str = str(exc).lower()
                if "rate" in error_str or "429" in error_str:
                    raise LinkedInRateLimitError(
                        f"LinkedIn rate limit hit: {exc}"
                    ) from exc
                raise LinkedInAPIError(
                    f"LinkedIn post failed: {exc}"
                ) from exc

        result = await asyncio.to_thread(_post)

        logger.info(
            "LinkedIn post published: text_len=%d, has_image=%s",
            len(text),
            image_path is not None,
        )

        return {"post_id": str(result), "status": "published"}

    # ------------------------------------------------------------------
    # Metrics retrieval
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def get_post_metrics(self, post_urn: str) -> Dict[str, int]:
        """Get engagement metrics for a published post.

        Args:
            post_urn: LinkedIn post URN or ID.

        Returns:
            Dict with engagement counts: ``likes``, ``comments``,
            ``shares``, ``impressions``.

        Raises:
            LinkedInAPIError: On API failure.
        """
        api = await self._get_api()

        def _get_metrics() -> Dict[str, int]:
            try:
                # The linkedin-api library exposes social actions
                reactions = api.get_post_reactions(post_urn)
                comments = api.get_post_comments(post_urn)
                return {
                    "likes": len(reactions) if reactions else 0,
                    "comments": len(comments) if comments else 0,
                    "shares": 0,  # Not directly available via Voyager
                    "impressions": 0,  # Not directly available via Voyager
                }
            except Exception as exc:
                raise LinkedInAPIError(
                    f"Failed to get metrics for {post_urn}: {exc}"
                ) from exc

        metrics = await asyncio.to_thread(_get_metrics)
        logger.debug("LinkedIn metrics for %s: %s", post_urn, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def get_post_comments(
        self,
        post_urn: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get comments on a LinkedIn post.

        Args:
            post_urn: LinkedIn post URN or ID.
            limit: Maximum number of comments to retrieve.

        Returns:
            List of comment dicts with at minimum ``text`` and ``author``
            keys.

        Raises:
            LinkedInAPIError: On API failure.
        """
        api = await self._get_api()

        def _get_comments() -> List[Dict[str, Any]]:
            try:
                raw_comments = api.get_post_comments(post_urn, comment_count=limit)
                if not raw_comments:
                    return []
                comments: List[Dict[str, Any]] = []
                for c in raw_comments[:limit]:
                    comments.append({
                        "text": c.get("comment", {}).get("values", [{}])[0].get(
                            "value", ""
                        )
                        if isinstance(c.get("comment"), dict)
                        else str(c.get("comment", "")),
                        "author": c.get("commenter", "unknown"),
                        "created_at": c.get("createdAt", ""),
                        "raw": c,
                    })
                return comments
            except Exception as exc:
                raise LinkedInAPIError(
                    f"Failed to get comments for {post_urn}: {exc}"
                ) from exc

        return await asyncio.to_thread(_get_comments)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def get_post_reactions(
        self,
        post_urn: str,
    ) -> List[Dict[str, Any]]:
        """Get reactions (likes, celebrates, etc.) on a LinkedIn post.

        Args:
            post_urn: LinkedIn post URN or ID.

        Returns:
            List of reaction dicts with ``type`` and ``actor`` keys.

        Raises:
            LinkedInAPIError: On API failure.
        """
        api = await self._get_api()

        def _get_reactions() -> List[Dict[str, Any]]:
            try:
                raw_reactions = api.get_post_reactions(post_urn)
                if not raw_reactions:
                    return []
                reactions: List[Dict[str, Any]] = []
                for r in raw_reactions:
                    reactions.append({
                        "type": r.get("reactionType", "LIKE"),
                        "actor": r.get("actor", {}).get("name", "unknown"),
                        "raw": r,
                    })
                return reactions
            except Exception as exc:
                raise LinkedInAPIError(
                    f"Failed to get reactions for {post_urn}: {exc}"
                ) from exc

        return await asyncio.to_thread(_get_reactions)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def refresh_session(self) -> None:
        """Force re-authentication by clearing the cached API client."""
        self._api = None
        logger.info("LinkedIn session cleared; will re-authenticate on next call")
