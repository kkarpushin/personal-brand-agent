"""
Async LinkedIn API client wrapper.

Wraps the ``tomquirk/linkedin-api`` library (Voyager API) with
``asyncio.to_thread`` to provide an async interface.

Authentication strategy (in order):
1. Reuse cached cookies from ``data/linkedin_cookies.json`` (instant).
2. If no cookies or expired, launch headless Playwright browser to
   log in with email + password, handle 2FA via TOTP, extract cookies,
   and cache them to disk.
3. Pass cookies to ``linkedin-api`` for all subsequent Voyager API calls.

Fail-fast philosophy: ``LinkedInAPIError``, ``LinkedInRateLimitError``,
and ``LinkedInSessionExpiredError`` are raised for corresponding failure
modes.  Transient errors are retried with exponential backoff.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exceptions import (
    LinkedInAPIError,
    LinkedInRateLimitError,
    LinkedInSessionExpiredError,
)
from src.utils import with_retry

logger = logging.getLogger(__name__)

COOKIES_PATH = Path("data/linkedin_cookies.json")
COOKIE_MAX_AGE = 3600 * 24 * 7  # 7 days


# ======================================================================
# COOKIE MANAGEMENT
# ======================================================================


def _save_cookies(cookies: List[Dict[str, Any]]) -> None:
    """Save browser cookies to disk."""
    COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cookies": cookies, "saved_at": time.time()}
    COOKIES_PATH.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("LinkedIn cookies saved to %s", COOKIES_PATH)


def _load_cookies() -> Optional[List[Dict[str, Any]]]:
    """Load cached cookies from disk. Returns None if missing or expired."""
    if not COOKIES_PATH.exists():
        return None
    try:
        payload = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
        saved_at = payload.get("saved_at", 0)
        if time.time() - saved_at > COOKIE_MAX_AGE:
            logger.info("LinkedIn cookies expired (age %.0fh)", (time.time() - saved_at) / 3600)
            return None
        cookies = payload.get("cookies", [])
        # Check that li_at is present (authenticated session)
        if not any(c.get("name") == "li_at" for c in cookies):
            return None
        return cookies
    except (json.JSONDecodeError, KeyError):
        return None


def _cookies_to_dict(cookies: List[Dict[str, Any]]) -> Dict[str, str]:
    """Convert list of cookie dicts to {name: value} mapping."""
    return {c["name"]: c["value"] for c in cookies if "name" in c and "value" in c}


# ======================================================================
# PLAYWRIGHT LOGIN WITH 2FA
# ======================================================================


def _browser_login(email: str, password: str, totp_secret: str) -> List[Dict[str, Any]]:
    """Log in to LinkedIn via headless browser, handling 2FA with TOTP.

    Returns:
        List of cookie dicts from the authenticated session.

    Raises:
        LinkedInSessionExpiredError: If login fails at any step.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    logger.info("Starting Playwright browser login for %s", email)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        try:
            # --- Step 1: Go to login page ---
            page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            page.wait_for_selector("#username", timeout=15000)

            # --- Step 2: Fill credentials ---
            page.fill("#username", email)
            page.fill("#password", password)
            page.click("button[type='submit']")

            # --- Step 3: Wait for navigation (may land on feed or 2FA) ---
            page.wait_for_load_state("domcontentloaded", timeout=15000)
            _wait_for_stable_url(page, timeout=10000)

            current_url = page.url

            # --- Step 4: Handle 2FA challenge if present ---
            if "checkpoint" in current_url or "challenge" in current_url:
                logger.info("2FA challenge detected at %s", current_url)

                if not totp_secret:
                    raise LinkedInSessionExpiredError(
                        "2FA challenge but no TOTP_SECRET configured"
                    )

                import pyotp
                code = pyotp.TOTP(totp_secret).now()
                logger.info("Generated TOTP code: %s***", code[:3])

                # Find and fill the PIN input
                pin_selector = _find_pin_input(page)
                if not pin_selector:
                    raise LinkedInSessionExpiredError(
                        "Could not find PIN input on challenge page"
                    )

                page.fill(pin_selector, code)

                # Submit the code
                _click_submit(page)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
                _wait_for_stable_url(page, timeout=10000)

                current_url = page.url
                logger.info("Post-2FA URL: %s", current_url)

            # --- Step 5: Verify we're authenticated ---
            if "feed" in current_url or "mynetwork" in current_url or "in/" in current_url:
                logger.info("LinkedIn login successful (landed on %s)", current_url)
            else:
                # One more check: see if li_at cookie exists
                cookies = context.cookies()
                if not any(c["name"] == "li_at" for c in cookies):
                    raise LinkedInSessionExpiredError(
                        f"Login did not complete. Final URL: {current_url}"
                    )

            # --- Step 6: Extract and return cookies ---
            cookies = context.cookies()
            logger.info(
                "Extracted %d cookies (li_at=%s)",
                len(cookies),
                "present" if any(c["name"] == "li_at" for c in cookies) else "MISSING",
            )
            return cookies

        except PwTimeout as exc:
            raise LinkedInSessionExpiredError(
                f"Browser login timed out: {exc}"
            ) from exc
        except LinkedInSessionExpiredError:
            raise
        except Exception as exc:
            raise LinkedInSessionExpiredError(
                f"Browser login failed: {exc}"
            ) from exc
        finally:
            browser.close()


def _find_pin_input(page: Any) -> Optional[str]:
    """Try multiple selectors to locate the PIN/code input field."""
    selectors = [
        "input[name='pin']",
        "input#input__phone_verification_pin",
        "input#input__email_verification_pin",
        "input[name='challengeData.pin']",
        "input[type='text'][name*='pin']",
        "input[type='tel']",
        "input[aria-label*='code']",
        "input[aria-label*='Code']",
        "input[placeholder*='code']",
        "input[placeholder*='Code']",
        "input[id*='verification']",
        "input[id*='pin']",
    ]
    for sel in selectors:
        try:
            if page.locator(sel).count() > 0:
                logger.debug("Found PIN input: %s", sel)
                return sel
        except Exception:
            continue
    return None


def _click_submit(page: Any) -> None:
    """Click the submit/verify button on a challenge page."""
    selectors = [
        "button[type='submit']",
        "button#two-step-submit-button",
        "button[data-litms-control-urn*='submit']",
        "button:has-text('Verify')",
        "button:has-text('Submit')",
        "button:has-text('Next')",
    ]
    for sel in selectors:
        try:
            if page.locator(sel).count() > 0:
                page.click(sel)
                return
        except Exception:
            continue
    # Fallback: press Enter
    page.keyboard.press("Enter")


def _wait_for_stable_url(page: Any, timeout: int = 10000) -> None:
    """Wait until the URL stops changing (page redirects have settled)."""
    start = time.time()
    last_url = page.url
    stable_since = start
    while (time.time() - start) * 1000 < timeout:
        time.sleep(0.5)
        current = page.url
        if current != last_url:
            last_url = current
            stable_since = time.time()
        elif (time.time() - stable_since) > 2.0:
            return
    return


# ======================================================================
# MAIN CLIENT
# ======================================================================


class LinkedInClient:
    """Async wrapper around the ``tomquirk/linkedin-api`` Voyager client.

    Authentication flow:
    1. Try cached cookies from ``data/linkedin_cookies.json``.
    2. If missing or expired, run Playwright headless login with 2FA.
    3. Cache new cookies for 7 days.
    4. Pass cookies to ``linkedin-api`` for Voyager API calls.

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
        """Lazy-init the ``linkedin-api`` client using cookies.

        Tries cached cookies first.  If unavailable, performs a full
        Playwright-based login (with 2FA if configured) and caches the
        resulting cookies.

        Returns:
            An authenticated ``Linkedin`` client instance.

        Raises:
            LinkedInSessionExpiredError: If login fails.
        """
        if self._api is not None:
            return self._api

        from linkedin_api import Linkedin  # type: ignore[import-untyped]

        # 1. Try cached cookies
        cookies = _load_cookies()
        if cookies:
            logger.info("Using cached LinkedIn cookies")
        else:
            # 2. Browser login
            cookies = await asyncio.to_thread(
                _browser_login, self.email, self.password, self.totp_secret
            )
            _save_cookies(cookies)

        # 3. Build CookieJar for linkedin-api
        from requests.cookies import RequestsCookieJar

        jar = RequestsCookieJar()
        for c in cookies:
            jar.set(
                c["name"],
                c["value"],
                domain=c.get("domain", ".linkedin.com"),
                path=c.get("path", "/"),
            )

        def _init_with_cookies() -> Any:
            try:
                return Linkedin("", "", cookies=jar, authenticate=True)
            except Exception as exc:
                raise LinkedInSessionExpiredError(
                    f"LinkedIn API init with cookies failed: {exc}"
                ) from exc

        self._api = await asyncio.to_thread(_init_with_cookies)
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
                reactions = api.get_post_reactions(post_urn)
                comments = api.get_post_comments(post_urn)
                return {
                    "likes": len(reactions) if reactions else 0,
                    "comments": len(comments) if comments else 0,
                    "shares": 0,
                    "impressions": 0,
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
        """Force re-authentication by clearing cached cookies and API client."""
        self._api = None
        if COOKIES_PATH.exists():
            COOKIES_PATH.unlink()
            logger.info("Deleted cached LinkedIn cookies")
        logger.info("LinkedIn session cleared; will re-authenticate on next call")
