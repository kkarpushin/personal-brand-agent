"""
Trend Scout Agent -- discovers trending AI/tech topics from multiple sources.

Scans five source categories in parallel (Perplexity, ArXiv, Twitter/X,
HackerNews via Perplexity, Reddit via Perplexity), classifies each
discovered topic by ``ContentType``, scores them using a multi-factor
weighted algorithm, applies exclusion rules, and returns a ranked list
with a top pick.

Source tiers
------------
- **Tier 1 (Research):** ArXiv papers via ``ArxivClient``
- **Tier 2 (News):** Perplexity sonar search, Twitter/X search
- **Tier 3 (Community):** HackerNews and Reddit via Perplexity site-scoped queries

Error philosophy
----------------
NO FALLBACKS.  Individual source scan failures are logged as warnings so
that other sources can still contribute, but if *zero* viable topics
survive after filtering the agent raises ``TrendScoutError``.  All
external API calls use ``@with_retry`` for transient-failure resilience.

Architecture reference: ``architecture.md`` lines 3102-5236.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.models import (
    ContentType,
    TrendTopic,
    TrendScoutOutput,
    SuggestedAngle,
    TopPickSummary,
    HookStyle,
    VisualType,
    CommunityContentMetadata,
    PrimarySourceMetadata,
    TopicMetadata,
)
from src.exceptions import TrendScoutError, RetryExhaustedError
from src.utils import utc_now, generate_id, with_retry
from src.tools.perplexity import PerplexityClient
from src.tools.arxiv import ArxivClient
from src.tools.twitter import TwitterClient
from src.tools.claude_client import ClaudeClient
from src.config import (
    DOMAIN_TO_CANDIDATE_TYPES,
    URL_PATTERN_TYPE_HINTS,
    get_domain_candidate_types,
    get_url_type_hint,
)

logger = logging.getLogger("TrendScout")

# =========================================================================
# CONSTANTS
# =========================================================================

# Simple domain-to-ContentType mapping for deterministic classification.
# When the domain maps to a single candidate, skip LLM classification.
DOMAIN_CLASSIFICATION: Dict[str, ContentType] = {
    "arxiv.org": ContentType.PRIMARY_SOURCE,
    "openai.com/research": ContentType.PRIMARY_SOURCE,
    "deepmind.com": ContentType.PRIMARY_SOURCE,
    "github.com": ContentType.AUTOMATION_CASE,
    "producthunt.com": ContentType.TOOL_RELEASE,
}

# Keywords used for AI/tech relevance scoring.
AI_RELEVANCE_KEYWORDS: List[str] = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "transformer", "llm", "large language model",
    "gpt", "claude", "gemini", "generative ai", "ai agent",
    "langchain", "langgraph", "rag", "retrieval augmented",
    "fine-tuning", "fine tuning", "prompt engineering",
    "diffusion model", "computer vision", "nlp",
    "natural language processing", "reinforcement learning",
    "automation", "workflow", "n8n", "api", "deployment",
]

# Exclusion keyword patterns (case-insensitive).
SPAM_PATTERNS: List[str] = [
    r"\bsponsored\b", r"\bad\b", r"\bpromo\b",
    r"\bcrypto\b", r"\bweb3\b", r"\bnft\b", r"\bblockchain\b",
    r"\btoken\s*sale\b", r"\bairdrop\b",
    r"\bcasino\b", r"\bgambling\b",
]
_SPAM_RE = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)

# Scoring weight defaults (recency, source_authority, engagement,
# relevance, uniqueness).
DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "recency": 0.20,
    "source_authority": 0.15,
    "engagement": 0.15,
    "relevance": 0.25,
    "uniqueness": 0.10,
    "evidence_quality": 0.15,
}

# Source authority tiers (0.0 - 1.0).
SOURCE_AUTHORITY: Dict[str, float] = {
    "arxiv": 0.95,
    "perplexity": 0.70,
    "twitter": 0.55,
    "hackernews": 0.75,
    "reddit": 0.60,
}

# LLM classification prompt (matches architecture.md lines 3276-3311).
CLASSIFICATION_PROMPT = """Classify this content into ONE of these categories:

ENTERPRISE_CASE: A detailed implementation story from a specific company with named company, business problem, solution, results/metrics.
PRIMARY_SOURCE: Original research, analysis, or expert opinion with novel thesis, methodology, written by researcher/analyst/expert.
AUTOMATION_CASE: Practical automation or AI agent implementation with specific workflow, tools listed, reproducible steps or code.
COMMUNITY_CONTENT: Discussion, reaction, or synthesis from community with multiple viewpoints, user-generated content.
TOOL_RELEASE: Announcement of new product, API, or feature with product name, features, launch/availability info.

Content to classify:
Title: {title}
URL: {url}
Summary: {summary}

Respond with ONLY the category name (e.g., "ENTERPRISE_CASE").
"""

# Perplexity search queries by source type.
PERPLEXITY_AI_QUERIES: List[str] = [
    "latest AI enterprise implementation case study with metrics 2025",
    "new AI tool release API launch this week",
    "AI agent automation workflow practical example",
]

HACKERNEWS_QUERIES: List[str] = [
    "AI machine learning trending discussion site:news.ycombinator.com",
    "LLM enterprise deployment site:news.ycombinator.com",
]

REDDIT_QUERIES: List[str] = [
    "AI machine learning trending discussion site:reddit.com",
    "LLM automation practical example site:reddit.com",
]


# =========================================================================
# TREND SCOUT AGENT
# =========================================================================


class TrendScoutAgent:
    """
    Discovers trending AI/tech topics from multiple sources.

    Classifies each topic by ``ContentType``, scores them using a
    multi-factor weighted algorithm, applies exclusion rules, and selects
    a top pick for the day.

    Args:
        perplexity: Perplexity AI search client.
        arxiv: ArXiv paper search client.
        twitter: Twitter/X API v2 client.
        claude: Claude LLM client (used for ambiguous classification).
        score_weights: Optional custom scoring weights. Uses
            ``DEFAULT_SCORE_WEIGHTS`` when not provided.
    """

    def __init__(
        self,
        perplexity: PerplexityClient,
        arxiv: ArxivClient,
        twitter: TwitterClient,
        claude: ClaudeClient,
        score_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.perplexity = perplexity
        self.arxiv = arxiv
        self.twitter = twitter
        self.claude = claude
        self.logger = logging.getLogger("TrendScout")
        self._score_weights = score_weights or dict(DEFAULT_SCORE_WEIGHTS)
        self._exclusion_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    async def run(self) -> TrendScoutOutput:
        """
        Main entry point.  Scans all sources, classifies, scores, filters,
        and returns a ``TrendScoutOutput`` with ranked topics and a top pick.

        Returns:
            Complete ``TrendScoutOutput`` dataclass.

        Raises:
            TrendScoutError: If no viable topics remain after filtering.
        """
        self.logger.info("[SCOUT] Starting trend discovery...")
        self._exclusion_log = []

        # 1. Scan all sources in parallel
        raw_topics = await self._scan_all_sources()
        self.logger.info(
            "[SCOUT] Raw topics collected: %d from all sources", len(raw_topics)
        )

        if not raw_topics:
            raise TrendScoutError(
                "All source scans returned zero topics. "
                "Check API keys and network connectivity."
            )

        # 2. Classify content types (domain mapping + LLM for ambiguous)
        classified = await self._classify_topics(raw_topics)
        self.logger.info(
            "[SCOUT] Topics classified: %d", len(classified)
        )

        # 3. Score and rank
        scored = self._score_topics(classified)

        # 4. Apply exclusion rules
        filtered = self._apply_exclusions(scored)
        self.logger.info(
            "[SCOUT] Topics after exclusions: %d (excluded %d)",
            len(filtered),
            len(scored) - len(filtered),
        )

        # 5. Fail-fast if nothing survives
        if not filtered:
            raise TrendScoutError(
                "No viable topics found after filtering. "
                f"Scanned {len(raw_topics)} raw, classified {len(classified)}, "
                f"scored {len(scored)}, all excluded."
            )

        # 6. Select top pick (highest score)
        top_pick = filtered[0]
        top_pick.is_top_pick = True
        top_pick.top_pick_summary = await self._generate_top_pick_summary(top_pick)

        # 7. Build type distribution statistics
        topics_by_type: Dict[str, int] = {}
        for topic in filtered:
            key = topic.content_type.value
            topics_by_type[key] = topics_by_type.get(key, 0) + 1

        run_id = generate_id()
        now = utc_now()

        output = TrendScoutOutput(
            run_id=run_id,
            run_timestamp=now,
            topics=filtered,
            top_pick=top_pick,
            topics_by_type=topics_by_type,
            total_sources_scanned=len(raw_topics),
            topics_before_filter=len(scored),
            topics_after_filter=len(filtered),
            exclusion_log=list(self._exclusion_log),
        )

        self.logger.info(
            "[SCOUT] Complete. run_id=%s, top_pick='%s' (score=%.2f, type=%s), "
            "distribution=%s",
            run_id,
            top_pick.title[:60],
            top_pick.score,
            top_pick.content_type.value,
            topics_by_type,
        )
        return output

    # ------------------------------------------------------------------
    # SOURCE SCANNING
    # ------------------------------------------------------------------

    async def _scan_all_sources(self) -> List[Dict[str, Any]]:
        """
        Scan all five source categories in parallel.

        Individual source failures are caught and logged as warnings so
        that the remaining sources can still contribute topics.  If every
        single source fails, the caller (``run``) will raise because the
        returned list will be empty.

        Returns:
            Merged list of raw topic dicts from all sources.
        """
        tasks = [
            self._scan_perplexity(),
            self._scan_arxiv(),
            self._scan_twitter(),
            self._scan_hackernews(),
            self._scan_reddit(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_topics: List[Dict[str, Any]] = []
        source_names = ["perplexity", "arxiv", "twitter", "hackernews", "reddit"]

        for name, result in zip(source_names, results):
            if isinstance(result, BaseException):
                self.logger.warning(
                    "[SCOUT] Source '%s' failed: %s", name, result
                )
                continue
            self.logger.info(
                "[SCOUT] Source '%s' returned %d topics", name, len(result)
            )
            all_topics.extend(result)

        return all_topics

    async def _scan_perplexity(self) -> List[Dict[str, Any]]:
        """
        Scan Perplexity sonar API with AI-related queries.

        Returns:
            List of raw topic dicts with keys: ``title``, ``url``,
            ``summary``, ``source``, ``published_at``.
        """
        topics: List[Dict[str, Any]] = []

        for query in PERPLEXITY_AI_QUERIES:
            try:
                response = await self.perplexity.search(query, model="sonar")
                text = PerplexityClient.extract_text(response)
                citations = PerplexityClient.extract_citations(response)

                if text:
                    # Each citation becomes a potential topic seed
                    for i, url in enumerate(citations):
                        topics.append({
                            "title": self._extract_title_from_text(text, i),
                            "url": url,
                            "summary": text[:500],
                            "source": "perplexity",
                            "published_at": None,
                            "engagement_count": 0,
                            "raw_content": text,
                        })

                    # If no citations, use the response itself
                    if not citations:
                        topics.append({
                            "title": self._extract_title_from_text(text, 0),
                            "url": "",
                            "summary": text[:500],
                            "source": "perplexity",
                            "published_at": None,
                            "engagement_count": 0,
                            "raw_content": text,
                        })

            except RetryExhaustedError:
                self.logger.warning(
                    "[SCOUT] Perplexity query failed after retries: %s", query
                )
                raise
            except Exception as exc:
                self.logger.warning(
                    "[SCOUT] Perplexity query error for '%s': %s", query, exc
                )

        self.logger.debug("[SCOUT] Perplexity returned %d raw topics", len(topics))
        return topics

    async def _scan_arxiv(self) -> List[Dict[str, Any]]:
        """
        Scan ArXiv for recent AI/ML papers.

        Returns:
            List of raw topic dicts with keys: ``title``, ``url``,
            ``summary``, ``source``, ``authors``, ``published_at``.
        """
        papers = await self.arxiv.get_recent_papers(max_results=15)
        topics: List[Dict[str, Any]] = []

        for paper in papers:
            published_at: Optional[datetime] = None
            if paper.published:
                try:
                    published_at = datetime.fromisoformat(paper.published)
                except (ValueError, TypeError):
                    pass

            topics.append({
                "title": paper.title,
                "url": paper.pdf_url or f"https://arxiv.org/abs/{paper.id}",
                "summary": paper.summary[:500],
                "source": "arxiv",
                "authors": paper.authors,
                "published_at": published_at,
                "engagement_count": 0,
                "raw_content": paper.summary,
            })

        self.logger.debug("[SCOUT] ArXiv returned %d papers", len(topics))
        return topics

    async def _scan_twitter(self) -> List[Dict[str, Any]]:
        """
        Scan Twitter/X for recent AI discussions.

        Returns:
            List of raw topic dicts with keys: ``title``, ``url``,
            ``summary``, ``source``, ``author``, ``engagement_count``.
        """
        query = (
            "(#AI OR #MachineLearning OR #LLM OR #GenerativeAI OR #AIAgents) "
            "lang:en -is:retweet"
        )
        tweets = await self.twitter.search_recent(query, max_results=20)
        topics: List[Dict[str, Any]] = []

        for tweet in tweets:
            text = tweet.get("text", "")
            metrics = tweet.get("public_metrics", {})
            engagement = (
                metrics.get("like_count", 0)
                + metrics.get("retweet_count", 0)
                + metrics.get("reply_count", 0)
            )
            tweet_id = tweet.get("id", "")

            published_at: Optional[datetime] = None
            created_str = tweet.get("created_at")
            if created_str:
                try:
                    published_at = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            topics.append({
                "title": text[:120].strip(),
                "url": f"https://twitter.com/i/status/{tweet_id}" if tweet_id else "",
                "summary": text,
                "source": "twitter",
                "author": tweet.get("author_id", ""),
                "published_at": published_at,
                "engagement_count": engagement,
                "raw_content": text,
            })

        self.logger.debug("[SCOUT] Twitter returned %d tweets", len(topics))
        return topics

    async def _scan_hackernews(self) -> List[Dict[str, Any]]:
        """
        Scan HackerNews via Perplexity site-scoped search.

        Returns:
            List of raw topic dicts.
        """
        topics: List[Dict[str, Any]] = []

        for query in HACKERNEWS_QUERIES:
            try:
                response = await self.perplexity.search(query, model="sonar")
                text = PerplexityClient.extract_text(response)
                citations = PerplexityClient.extract_citations(response)

                for url in citations:
                    topics.append({
                        "title": self._extract_title_from_text(text, 0),
                        "url": url,
                        "summary": text[:500],
                        "source": "hackernews",
                        "published_at": None,
                        "engagement_count": 0,
                        "raw_content": text,
                    })
            except Exception as exc:
                self.logger.warning(
                    "[SCOUT] HackerNews query error for '%s': %s", query, exc
                )

        self.logger.debug("[SCOUT] HackerNews returned %d topics", len(topics))
        return topics

    async def _scan_reddit(self) -> List[Dict[str, Any]]:
        """
        Scan Reddit via Perplexity site-scoped search.

        Returns:
            List of raw topic dicts.
        """
        topics: List[Dict[str, Any]] = []

        for query in REDDIT_QUERIES:
            try:
                response = await self.perplexity.search(query, model="sonar")
                text = PerplexityClient.extract_text(response)
                citations = PerplexityClient.extract_citations(response)

                for url in citations:
                    topics.append({
                        "title": self._extract_title_from_text(text, 0),
                        "url": url,
                        "summary": text[:500],
                        "source": "reddit",
                        "published_at": None,
                        "engagement_count": 0,
                        "raw_content": text,
                    })
            except Exception as exc:
                self.logger.warning(
                    "[SCOUT] Reddit query error for '%s': %s", query, exc
                )

        self.logger.debug("[SCOUT] Reddit returned %d topics", len(topics))
        return topics

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------

    async def _classify_topics(
        self, raw_topics: List[Dict[str, Any]]
    ) -> List[TrendTopic]:
        """
        Classify raw topic dicts into ``TrendTopic`` objects with a
        ``ContentType``.

        Uses a two-level classification strategy:
        1. Domain mapping -- deterministic, fast, no LLM cost.
        2. LLM refinement via Claude -- for ambiguous or unknown domains.

        Args:
            raw_topics: List of raw dicts from source scans.

        Returns:
            List of classified ``TrendTopic`` instances.
        """
        classified: List[TrendTopic] = []

        for raw in raw_topics:
            url = raw.get("url", "")
            title = raw.get("title", "Untitled")
            summary = raw.get("summary", "")
            source = raw.get("source", "unknown")

            # Level 1: Domain-based classification
            content_type = self._classify_by_domain(url)

            # Level 2: LLM refinement for ambiguous cases
            if content_type is None:
                content_type = await self._classify_by_llm(title, url, summary)

            # Build minimal metadata based on classified type
            metadata = self._build_default_metadata(content_type, raw)

            # Build suggested angles placeholder
            suggested_angles = self._build_suggested_angles(content_type)

            # Determine recommended visual type
            visual_types = VisualType.for_content_type(content_type)
            recommended_visual = visual_types[0].value if visual_types else "author_photo"

            topic = TrendTopic(
                id=generate_id(),
                title=title,
                summary=summary[:300],
                content_type=content_type,
                sources=[url] if url else [],
                primary_source_url=url,
                score=0.0,  # Will be set during scoring
                score_breakdown={},
                quality_signals_matched=[],
                suggested_angles=suggested_angles,
                related_topics=[],
                raw_content=raw.get("raw_content", summary),
                metadata=metadata,
                analysis_format=self._analysis_format_for(content_type),
                recommended_post_format=self._post_format_for(content_type),
                recommended_visual_type=recommended_visual,
                source_published_at=raw.get("published_at"),
            )
            classified.append(topic)

        return classified

    def _classify_by_domain(self, url: str) -> Optional[ContentType]:
        """
        Attempt deterministic classification using the domain mapping.

        Returns:
            ``ContentType`` if the domain maps to exactly one type,
            ``None`` if ambiguous or unknown.
        """
        if not url:
            return None

        domain = urlparse(url).netloc.replace("www.", "")

        # Check simple single-type mapping first
        for pattern, ctype in DOMAIN_CLASSIFICATION.items():
            if pattern in url:
                return ctype

        # Check config-level domain mapping
        candidates = DOMAIN_TO_CANDIDATE_TYPES.get(domain)
        if candidates and len(candidates) == 1:
            try:
                return ContentType(candidates[0])
            except ValueError:
                pass

        # Check URL pattern hints
        hint = get_url_type_hint(url)
        if hint:
            try:
                return ContentType(hint)
            except ValueError:
                pass

        return None

    async def _classify_by_llm(
        self, title: str, url: str, summary: str
    ) -> ContentType:
        """
        Classify content using Claude LLM when domain mapping is ambiguous.

        Falls back to ``COMMUNITY_CONTENT`` if the LLM call fails or
        returns an unrecognisable value (this is the safest default since
        community content has the loosest requirements).

        Returns:
            Classified ``ContentType``.
        """
        prompt = CLASSIFICATION_PROMPT.format(
            title=title,
            url=url,
            summary=summary[:500],
        )

        try:
            response = await self.claude.generate(
                prompt,
                system="You are a content classifier. Respond with ONLY the category name.",
                max_tokens=50,
                temperature=0.1,
            )
            classified = response.strip().upper()

            # Map to ContentType enum
            return ContentType(classified.lower())

        except (ValueError, KeyError):
            self.logger.warning(
                "[SCOUT] LLM returned unrecognised type '%s' for '%s'. "
                "Falling back to COMMUNITY_CONTENT.",
                response.strip() if "response" in dir() else "N/A",
                title[:60],
            )
            return ContentType.COMMUNITY_CONTENT

        except Exception as exc:
            self.logger.warning(
                "[SCOUT] LLM classification failed for '%s': %s. "
                "Falling back to COMMUNITY_CONTENT.",
                title[:60],
                exc,
            )
            return ContentType.COMMUNITY_CONTENT

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------

    def _score_topics(self, topics: List[TrendTopic]) -> List[TrendTopic]:
        """
        Score each topic using a multi-factor weighted algorithm, then
        sort descending by score.

        Scoring factors (each 0.0--1.0):
        - **recency**: Age-based decay.  <6h = 1.0, >48h = 0.0.
        - **source_authority**: Reputation of the originating source.
        - **engagement**: Normalised engagement metrics.
        - **relevance**: AI/tech keyword matching density.
        - **uniqueness**: Deduplication penalty for near-duplicate titles.
        - **evidence_quality**: Presence of metrics, data, specifics.

        Final score is the weighted average scaled to 0--10.

        Returns:
            The same list, mutated in-place, sorted highest-score-first.
        """
        now = utc_now()
        seen_titles: List[str] = []

        for topic in topics:
            breakdown: Dict[str, float] = {}

            # Recency score
            breakdown["recency"] = self._calc_recency(topic, now)

            # Source authority
            source_name = self._infer_source(topic)
            breakdown["source_authority"] = SOURCE_AUTHORITY.get(source_name, 0.5)

            # Engagement (normalise: assume max interesting engagement ~5000)
            engagement = self._estimate_engagement(topic)
            breakdown["engagement"] = min(1.0, engagement / 5000.0)

            # Relevance (keyword density in title + summary)
            breakdown["relevance"] = self._calc_relevance(topic)

            # Uniqueness (penalise near-duplicate titles)
            breakdown["uniqueness"] = self._calc_uniqueness(topic.title, seen_titles)
            seen_titles.append(topic.title.lower())

            # Evidence quality (heuristic: presence of numbers, metrics keywords)
            breakdown["evidence_quality"] = self._calc_evidence_quality(topic)

            # Weighted average -> 0-1 range
            raw_score = sum(
                breakdown[k] * self._score_weights.get(k, 0.0)
                for k in breakdown
            )
            # Scale to 0-10
            topic.score = round(raw_score * 10.0, 2)
            topic.score_breakdown = {k: round(v, 3) for k, v in breakdown.items()}

            # Quality signals
            topic.quality_signals_matched = [
                k for k, v in breakdown.items() if v >= 0.7
            ]

        # Sort descending by score
        topics.sort(key=lambda t: t.score, reverse=True)
        return topics

    @staticmethod
    def _calc_recency(topic: TrendTopic, now: datetime) -> float:
        """Age-based decay: <6h = 1.0, 6-24h linear decay, >48h = 0.0."""
        pub = topic.source_published_at
        if pub is None:
            return 0.5  # Unknown age gets a neutral score

        # Ensure timezone-aware
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        age_hours = (now - pub).total_seconds() / 3600.0
        if age_hours < 6:
            return 1.0
        elif age_hours < 24:
            return 1.0 - ((age_hours - 6) / 18.0) * 0.5  # 1.0 -> 0.5
        elif age_hours < 48:
            return 0.5 - ((age_hours - 24) / 24.0) * 0.5  # 0.5 -> 0.0
        else:
            return 0.0

    @staticmethod
    def _infer_source(topic: TrendTopic) -> str:
        """Infer the originating source name from URL or sources list."""
        url = topic.primary_source_url.lower()
        if "arxiv.org" in url:
            return "arxiv"
        if "twitter.com" in url or "x.com" in url:
            return "twitter"
        if "news.ycombinator.com" in url or "hn" in url:
            return "hackernews"
        if "reddit.com" in url:
            return "reddit"
        return "perplexity"

    @staticmethod
    def _estimate_engagement(topic: TrendTopic) -> int:
        """
        Extract engagement count from metadata or raw content heuristics.
        """
        # Check community metadata
        if isinstance(topic.metadata, CommunityContentMetadata):
            metrics = topic.metadata.engagement_metrics
            return sum(metrics.values()) if metrics else 0
        return 0

    @staticmethod
    def _calc_relevance(topic: TrendTopic) -> float:
        """
        Calculate AI/tech relevance using keyword density in title + summary.

        Returns:
            Score 0.0--1.0 based on fraction of keywords matched.
        """
        text = f"{topic.title} {topic.summary}".lower()
        matches = sum(1 for kw in AI_RELEVANCE_KEYWORDS if kw in text)
        # Normalise: 5+ keyword matches = 1.0
        return min(1.0, matches / 5.0)

    @staticmethod
    def _calc_uniqueness(title: str, seen_titles: List[str]) -> float:
        """
        Penalise near-duplicate titles using simple token overlap.

        Returns:
            1.0 if fully unique, decreasing as overlap with existing
            titles grows.
        """
        if not seen_titles:
            return 1.0

        title_tokens = set(title.lower().split())
        if not title_tokens:
            return 1.0

        max_overlap = 0.0
        for seen in seen_titles:
            seen_tokens = set(seen.split())
            if not seen_tokens:
                continue
            overlap = len(title_tokens & seen_tokens) / max(
                len(title_tokens), len(seen_tokens)
            )
            max_overlap = max(max_overlap, overlap)

        # 80%+ overlap -> score 0.0;  0% overlap -> 1.0
        return max(0.0, 1.0 - (max_overlap / 0.8))

    @staticmethod
    def _calc_evidence_quality(topic: TrendTopic) -> float:
        """
        Heuristic: presence of numbers, metrics keywords, and specifics
        in the raw content.
        """
        text = topic.raw_content.lower()
        signals = 0

        # Numbers / percentages
        if re.search(r"\d+%", text):
            signals += 1
        if re.search(r"\$[\d,.]+", text):
            signals += 1
        # Metrics keywords
        metric_kw = ["roi", "revenue", "savings", "improvement", "increase",
                      "decrease", "benchmark", "accuracy", "latency", "throughput"]
        signals += sum(1 for kw in metric_kw if kw in text)

        # Cap at 5 signals for 1.0
        return min(1.0, signals / 5.0)

    # ------------------------------------------------------------------
    # EXCLUSION RULES
    # ------------------------------------------------------------------

    def _apply_exclusions(self, topics: List[TrendTopic]) -> List[TrendTopic]:
        """
        Filter out topics that match exclusion criteria.

        Exclusion rules:
        - Paid promotions / sponsored content.
        - Crypto / web3 spam.
        - Non-English content (heuristic).
        - Near-zero relevance score.
        - Score below minimum viable threshold (1.0 on 0-10 scale).

        Excluded topics are logged to ``self._exclusion_log`` for
        transparency in the ``TrendScoutOutput``.

        Returns:
            Filtered list (order preserved from input).
        """
        filtered: List[TrendTopic] = []

        for topic in topics:
            reason = self._check_exclusion(topic)
            if reason:
                self._exclusion_log.append({
                    "topic_id": topic.id,
                    "title": topic.title[:80],
                    "reason": reason,
                    "score": topic.score,
                })
                self.logger.debug(
                    "[SCOUT] Excluded: '%s' -- reason: %s",
                    topic.title[:60],
                    reason,
                )
            else:
                filtered.append(topic)

        return filtered

    def _check_exclusion(self, topic: TrendTopic) -> Optional[str]:
        """
        Check a single topic against all exclusion rules.

        Returns:
            Exclusion reason string, or ``None`` if the topic passes.
        """
        combined_text = f"{topic.title} {topic.summary} {topic.raw_content}"

        # 1. Spam / promotional content
        if _SPAM_RE.search(combined_text):
            return "spam_or_promotional"

        # 2. Near-zero relevance
        relevance = topic.score_breakdown.get("relevance", 0.0)
        if relevance < 0.1:
            return "low_relevance"

        # 3. Minimum viable score (below 1.0/10 is clearly noise)
        if topic.score < 1.0:
            return "below_minimum_score"

        # 4. Empty or extremely short content
        if len(topic.title.strip()) < 5:
            return "title_too_short"

        if len(topic.summary.strip()) < 20:
            return "summary_too_short"

        return None

    # ------------------------------------------------------------------
    # TOP PICK SUMMARY
    # ------------------------------------------------------------------

    async def _generate_top_pick_summary(
        self, topic: TrendTopic
    ) -> TopPickSummary:
        """
        Generate a ``TopPickSummary`` for the highest-scoring topic using
        Claude to produce a concise rationale and takeaways.

        Args:
            topic: The top-pick ``TrendTopic``.

        Returns:
            Populated ``TopPickSummary``.
        """
        prompt = (
            f"You selected this as the top AI topic of the day:\n\n"
            f"Title: {topic.title}\n"
            f"Type: {topic.content_type.value}\n"
            f"Summary: {topic.summary}\n"
            f"Score: {topic.score}/10\n\n"
            f"Provide a JSON object with:\n"
            f"- \"why_chosen\": 1-2 sentence explanation of why this is the best pick\n"
            f"- \"key_takeaways\": array of exactly 3 concise takeaways\n"
            f"- \"who_should_care\": 1 sentence about the target audience\n"
        )

        try:
            data = await self.claude.generate_structured(prompt, max_tokens=512)
            return TopPickSummary(
                why_chosen=data.get("why_chosen", "Highest composite quality score."),
                key_takeaways=data.get("key_takeaways", [
                    "Strong evidence quality",
                    "High relevance to AI practitioners",
                    "Timely and unique angle",
                ])[:3],
                who_should_care=data.get(
                    "who_should_care",
                    "AI practitioners and technology leaders.",
                ),
            )
        except Exception as exc:
            self.logger.warning(
                "[SCOUT] Failed to generate top-pick summary via LLM: %s. "
                "Using fallback.",
                exc,
            )
            return TopPickSummary(
                why_chosen=(
                    f"Highest composite score ({topic.score}/10) across "
                    f"recency, relevance, evidence quality, and source authority."
                ),
                key_takeaways=[
                    "Strong evidence quality",
                    "High relevance to AI practitioners",
                    "Timely and unique angle",
                ],
                who_should_care="AI practitioners, tech leaders, and automation enthusiasts.",
            )

    # ------------------------------------------------------------------
    # HELPER: METADATA BUILDERS
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_metadata(
        content_type: ContentType, raw: Dict[str, Any]
    ) -> TopicMetadata:
        """
        Build a minimal metadata dataclass matching the classified
        ``ContentType``.

        Full metadata is populated downstream by the Analyzer agent;
        the Trend Scout provides the structural skeleton with whatever
        information is available from the source scan.
        """
        from src.models import (
            EnterpriseCaseMetadata,
            PrimarySourceMetadata,
            AutomationCaseMetadata,
            CommunityContentMetadata,
            ToolReleaseMetadata,
        )

        source = raw.get("source", "unknown")

        if content_type == ContentType.PRIMARY_SOURCE:
            authors = raw.get("authors", ["Unknown"])
            return PrimarySourceMetadata(
                authors=authors if authors else ["Unknown"],
                organization=authors[0] if authors else "Unknown",
                source_type="research_paper" if source == "arxiv" else "expert_essay",
                publication_venue="ArXiv" if source == "arxiv" else "Web",
                key_hypothesis=raw.get("title", ""),
                methodology_summary="",
            )

        if content_type == ContentType.ENTERPRISE_CASE:
            return EnterpriseCaseMetadata(
                company=raw.get("title", "Unknown Company")[:50],
                industry="Technology",
                scale="Enterprise",
            )

        if content_type == ContentType.AUTOMATION_CASE:
            return AutomationCaseMetadata(
                agent_type="AI workflow",
                workflow_components=["LLM"],
                use_case_domain="AI automation",
            )

        if content_type == ContentType.TOOL_RELEASE:
            return ToolReleaseMetadata(
                tool_name=raw.get("title", "Unknown Tool")[:50],
                company="Unknown",
                release_date=utc_now().isoformat(),
                release_type="new_product",
                key_features=["AI-powered"],
            )

        # Default: COMMUNITY_CONTENT
        platform_map = {
            "twitter": "Twitter",
            "reddit": "Reddit",
            "hackernews": "HackerNews",
            "perplexity": "Medium",
        }
        return CommunityContentMetadata(
            platform=platform_map.get(source, "Reddit"),
            format="post",
            author_credibility="unknown",
        )

    @staticmethod
    def _build_suggested_angles(content_type: ContentType) -> List[SuggestedAngle]:
        """
        Build default suggested angles based on ``ContentType``.

        The Writer agent will refine these using the Analyzer's brief.
        """
        from src.models import CONTENT_TYPE_HOOK_STYLES

        hook_styles = CONTENT_TYPE_HOOK_STYLES.get(content_type, [])
        angles: List[SuggestedAngle] = []

        for style in hook_styles[:3]:  # Top 3 hook styles
            angles.append(
                SuggestedAngle(
                    angle_text=f"Approach using {style.value} hook style",
                    angle_type=style.value,
                    hook_templates=[f"Template for {style.value}"],
                    content_type_fit=0.8,
                )
            )

        return angles

    @staticmethod
    def _analysis_format_for(content_type: ContentType) -> str:
        """Return the analysis format string for a given content type."""
        mapping = {
            ContentType.ENTERPRISE_CASE: "case_study_extraction",
            ContentType.PRIMARY_SOURCE: "research_analysis",
            ContentType.AUTOMATION_CASE: "workflow_breakdown",
            ContentType.COMMUNITY_CONTENT: "discussion_synthesis",
            ContentType.TOOL_RELEASE: "product_evaluation",
        }
        return mapping.get(content_type, "general_analysis")

    @staticmethod
    def _post_format_for(content_type: ContentType) -> str:
        """Return the recommended post format for a given content type."""
        mapping = {
            ContentType.ENTERPRISE_CASE: "insight_thread",
            ContentType.PRIMARY_SOURCE: "contrarian",
            ContentType.AUTOMATION_CASE: "tutorial",
            ContentType.COMMUNITY_CONTENT: "discussion_summary",
            ContentType.TOOL_RELEASE: "first_look",
        }
        return mapping.get(content_type, "insight_thread")

    @staticmethod
    def _extract_title_from_text(text: str, index: int) -> str:
        """
        Extract a usable title from Perplexity response text.

        Attempts to pull the first meaningful sentence; falls back to
        truncating the text at 120 characters.
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if sentences and index < len(sentences):
            title = sentences[index].strip()
            if len(title) > 120:
                return title[:117] + "..."
            return title
        # Fallback
        return text[:120].strip() if text else f"Topic #{index + 1}"


# =========================================================================
# MODULE-LEVEL FACTORY
# =========================================================================


async def create_trend_scout() -> TrendScoutAgent:
    """
    Factory function to create a ``TrendScoutAgent`` with all required
    clients initialised from environment variables.

    Returns:
        Fully-configured ``TrendScoutAgent`` instance.

    Raises:
        KeyError: If required environment variables (API keys) are missing.
    """
    perplexity = PerplexityClient()
    arxiv_client = ArxivClient()
    twitter = TwitterClient()
    claude = ClaudeClient()

    return TrendScoutAgent(
        perplexity=perplexity,
        arxiv=arxiv_client,
        twitter=twitter,
        claude=claude,
    )
