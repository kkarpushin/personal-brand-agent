"""
Async X/Twitter API v2 client for trend monitoring.

Uses ``httpx`` to call the Twitter API v2 endpoints.  The Trend Scout
agent uses this client to monitor AI/ML thought leaders and trending
hashtags for content discovery.

Fail-fast philosophy: transient HTTP errors are retried with exponential
backoff; auth failures propagate immediately.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from src.utils import with_retry

logger = logging.getLogger(__name__)


class TwitterClient:
    """Async X/Twitter API v2 client for trend monitoring.

    Args:
        bearer_token: Twitter API v2 bearer token.  Falls back to the
            ``TWITTER_BEARER_TOKEN`` environment variable.

    Usage::

        client = TwitterClient()
        tweets = await client.search_recent("#AI #enterprise case study")
        user_tweets = await client.get_user_tweets("karpathy", max_results=5)
    """

    BASE_URL: str = "https://api.twitter.com/2"

    def __init__(self, bearer_token: Optional[str] = None) -> None:
        self.bearer_token: str = bearer_token or os.environ.get(
            "TWITTER_BEARER_TOKEN", ""
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Build authorization headers for API requests.

        Returns:
            Dict with ``Authorization`` and ``Content-Type`` headers.
        """
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Recent search
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(httpx.HTTPError,))
    async def search_recent(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search recent tweets matching a query.

        Uses the ``/tweets/search/recent`` endpoint (requires at least
        Basic tier API access).

        Args:
            query: Twitter search query (supports operators like
                ``#hashtag``, ``from:user``, ``-is:retweet``).
            max_results: Number of results (10--100).

        Returns:
            List of tweet dicts with ``id``, ``text``, ``author_id``,
            and ``public_metrics`` keys.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses after retries.
        """
        # Clamp max_results to API limits
        max_results = max(10, min(max_results, 100))

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/tweets/search/recent",
                headers=self._auth_headers(),
                params={
                    "query": query,
                    "max_results": max_results,
                    "tweet.fields": "created_at,public_metrics,author_id,lang",
                },
            )
            response.raise_for_status()
            data = response.json()

        tweets = data.get("data", [])

        logger.info(
            "Twitter search: query=%r, results=%d",
            query,
            len(tweets),
        )
        return tweets

    # ------------------------------------------------------------------
    # User tweets
    # ------------------------------------------------------------------

    async def get_user_tweets(
        self,
        username: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent tweets from a specific user.

        Resolves the username to a user ID first, then fetches their
        timeline.

        Args:
            username: Twitter handle (without the ``@`` prefix).
            max_results: Number of tweets to retrieve (5--100).

        Returns:
            List of tweet dicts with ``id``, ``text``, and
            ``public_metrics`` keys.
        """
        user_id = await self._resolve_username(username)
        if user_id is None:
            logger.warning("Twitter user not found: @%s", username)
            return []

        return await self._get_user_timeline(user_id, max_results)

    # ------------------------------------------------------------------
    # Username resolution
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(httpx.HTTPError,))
    async def _resolve_username(self, username: str) -> Optional[str]:
        """Resolve a Twitter username to its numeric user ID.

        Args:
            username: Twitter handle (without ``@``).

        Returns:
            User ID string, or ``None`` if the user is not found.
        """
        # Strip @ if accidentally included
        username = username.lstrip("@")

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/users/by/username/{username}",
                headers=self._auth_headers(),
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

        return data.get("data", {}).get("id")

    # ------------------------------------------------------------------
    # User timeline
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(httpx.HTTPError,))
    async def _get_user_timeline(
        self,
        user_id: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch recent tweets from a user's timeline by user ID.

        Args:
            user_id: Numeric Twitter user ID.
            max_results: Number of tweets to retrieve (5--100).

        Returns:
            List of tweet dicts.
        """
        max_results = max(5, min(max_results, 100))

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/users/{user_id}/tweets",
                headers=self._auth_headers(),
                params={
                    "max_results": max_results,
                    "tweet.fields": "created_at,public_metrics,author_id,lang",
                    "exclude": "retweets,replies",
                },
            )
            response.raise_for_status()
            data = response.json()

        tweets = data.get("data", [])

        logger.debug(
            "Twitter user timeline: user_id=%s, results=%d",
            user_id,
            len(tweets),
        )
        return tweets
