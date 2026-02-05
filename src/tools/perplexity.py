"""
Async Perplexity AI search client.

Uses ``httpx`` to call the Perplexity chat-completions endpoint for
real-time web research.  The Trend Scout agent relies on this client to
discover enterprise case studies, research papers, tool releases, and
community discussions.

Fail-fast philosophy: transient HTTP errors are retried with exponential
backoff; permanent failures propagate immediately.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from src.utils import with_retry

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Async wrapper around the Perplexity AI chat-completions API.

    Args:
        api_key: Perplexity API key.  Falls back to the
            ``PERPLEXITY_API_KEY`` environment variable.

    Usage::

        client = PerplexityClient()
        result = await client.search("enterprise AI case study 2025")
        findings = await client.research_topic("LLM deployment in finance")
    """

    BASE_URL: str = "https://api.perplexity.ai"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or os.environ.get("PERPLEXITY_API_KEY", "")

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    @with_retry(
        max_attempts=3,
        retryable_exceptions=(httpx.HTTPError, httpx.TimeoutException),
    )
    async def search(
        self,
        query: str,
        model: str = "sonar",
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Execute a single search query via the Perplexity API.

        Args:
            query: Natural-language search query.
            model: Perplexity model name (``sonar``, ``sonar-pro``, etc.).
            max_tokens: Maximum tokens in the response.

        Returns:
            Raw JSON response from the Perplexity API including
            ``choices``, ``citations``, and usage metadata.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses after retries.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

            logger.debug(
                "Perplexity search completed: model=%s, query_len=%d",
                model,
                len(query),
            )
            return data

    # ------------------------------------------------------------------
    # Multi-query research
    # ------------------------------------------------------------------

    async def research_topic(
        self,
        topic: str,
        num_queries: int = 3,
        model: str = "sonar",
    ) -> List[Dict[str, Any]]:
        """Research a topic with multiple targeted queries.

        Generates ``num_queries`` angle-specific search queries from the
        given topic and executes them sequentially (to respect rate
        limits).

        Args:
            topic: High-level topic to research.
            num_queries: Number of search queries to issue.
            model: Perplexity model to use.

        Returns:
            List of raw API responses, one per query.
        """
        queries = self._generate_research_queries(topic, num_queries)
        results: List[Dict[str, Any]] = []

        for query in queries:
            try:
                result = await self.search(query, model=model)
                results.append(result)
            except Exception:
                logger.warning(
                    "Perplexity research query failed: %s",
                    query,
                    exc_info=True,
                )
                # Continue with remaining queries -- partial results are
                # better than none for research.
                continue

        logger.info(
            "Perplexity research_topic: topic=%s, queries=%d, results=%d",
            topic,
            len(queries),
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Query generation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_research_queries(topic: str, num_queries: int) -> List[str]:
        """Generate diverse search queries for a topic.

        The queries cover different angles: factual overview, recent news
        and developments, and expert analysis / case studies.

        Args:
            topic: The topic to research.
            num_queries: Number of queries to generate (max 5).

        Returns:
            List of query strings.
        """
        templates = [
            f"{topic} latest developments 2025",
            f"{topic} enterprise case study results metrics",
            f"{topic} expert analysis implications",
            f"{topic} research paper findings",
            f"{topic} practical implementation guide",
        ]
        return templates[:num_queries]

    # ------------------------------------------------------------------
    # Content extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text(response: Dict[str, Any]) -> str:
        """Extract the plain-text answer from a Perplexity API response.

        Args:
            response: Raw JSON response from :meth:`search`.

        Returns:
            Extracted text content, or an empty string if the response
            structure is unexpected.
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning("Failed to extract text from Perplexity response")
            return ""

    @staticmethod
    def extract_citations(response: Dict[str, Any]) -> List[str]:
        """Extract citation URLs from a Perplexity API response.

        Args:
            response: Raw JSON response from :meth:`search`.

        Returns:
            List of citation URLs (may be empty).
        """
        return response.get("citations", [])
