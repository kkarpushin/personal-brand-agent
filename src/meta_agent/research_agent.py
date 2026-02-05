"""
Research Agent for the LinkedIn Super Agent meta-improvement layer.

Investigates best practices, competitor strategies, and own performance
data to generate actionable recommendations for improving content quality.

The ResearchAgent performs **deep, periodic** research (as opposed to the
ContinuousLearningEngine's shallow, every-iteration micro-learnings).
Together they form the dual-mode self-improvement architecture described
in ``architecture.md`` lines 15420-15477.

Research is triggered by:
    - UNDERPERFORMANCE: 3+ of last 5 posts score below 80% of the average
    - WEEKLY_CYCLE: Sunday + 7+ days since last research
    - External triggers: NEW_CONTENT_TYPE, ALGORITHM_CHANGE, MANUAL_REQUEST

Research sources:
    1. Perplexity AI web search (best practices, trends, algorithm changes)
    2. Competitor LinkedIn post scraping (hooks, structure, visual patterns)
    3. Own post analytics (top 10 vs bottom 10 comparison)

Error philosophy: NO FALLBACKS, FAIL FAST.  Individual query failures are
logged and skipped (partial results are acceptable), but the overall cycle
respects a hard timeout and raises on infrastructure failures.

Architecture references:
    - ``architecture.md`` lines 14350-14374  (Research & Learn overview)
    - ``architecture.md`` lines 17119-17500  (ResearchAgent full spec)
    - ``architecture.md`` lines 15452-15477  (Research Agent role in dual-mode)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.tools.claude_client import ClaudeClient, get_claude
from src.tools.perplexity import PerplexityClient
from src.meta_agent.models import (
    ResearchTrigger,
    ResearchQuery,
    ResearchFinding,
    ResearchRecommendation,
    ResearchReport,
)
from src.utils import utc_now

logger = logging.getLogger("ResearchAgent")


class ResearchAgent:
    """Agent that researches best practices and competitor strategies.

    Uses Perplexity for web research, scrapes competitor LinkedIn posts,
    and analyzes own performance data to produce a comprehensive
    :class:`ResearchReport` with actionable recommendations.

    Args:
        perplexity_client: Async Perplexity API client for web search.
        linkedin_scraper: LinkedIn client capable of fetching profile posts.
        analytics_db: :class:`SupabaseDB` instance for reading post data
            and metrics.
        claude_client: Optional :class:`ClaudeClient` for LLM analysis.
            Defaults to a new client via :func:`get_claude`.

    Architecture reference: ``architecture.md`` lines 17119-17147.
    """

    def __init__(
        self,
        perplexity_client: PerplexityClient,
        linkedin_scraper: Any,
        analytics_db: Any,
        claude_client: Optional[ClaudeClient] = None,
    ) -> None:
        self.perplexity = perplexity_client
        self.linkedin = linkedin_scraper
        self.db = analytics_db
        self.claude = claude_client or get_claude()

        # Top LinkedIn influencers to learn from
        # Architecture reference: architecture.md lines 17140-17147
        self.competitors: List[str] = [
            "https://www.linkedin.com/in/justinwelsh/",
            "https://www.linkedin.com/in/sambhavchaturvedi/",
            "https://www.linkedin.com/in/shanebarker/",
            "https://www.linkedin.com/in/garyvaynerchuk/",
            "https://www.linkedin.com/in/jasonfalls/",
        ]

    # -----------------------------------------------------------------
    # TRIGGER EVALUATION
    # -----------------------------------------------------------------

    async def should_research(self) -> Optional[ResearchTrigger]:
        """Check whether a research cycle should be triggered.

        Evaluates two conditions in order:

        1. **UNDERPERFORMANCE** -- Triggered when 3 or more of the last 5
           posts score below 80% of the historical average.  This indicates
           a potential content strategy issue that strategic research might
           address.

        2. **WEEKLY_CYCLE** -- Triggered every Sunday if 7 or more days
           have elapsed since the last research report.  Regular research
           keeps strategies fresh and incorporates new best practices.

        Returns:
            A :class:`ResearchTrigger` variant if research is needed,
            or ``None`` if no research is currently warranted.

        Architecture reference: ``architecture.md`` lines 17149-17198.
        """
        # --- Check for underperformance ---
        recent_posts = await self.db.get_recent_posts(limit=5)
        avg_score = await self.db.get_average_score()

        underperforming_count = 0
        for post in recent_posts:
            post_score = post.get("score", post.get("qc_score", 0))
            if post_score < avg_score * 0.8:
                underperforming_count += 1

        if underperforming_count >= 3:
            logger.info(
                "[RESEARCH] Underperformance detected: %d of %d recent posts "
                "below 80%% of average (%.1f)",
                underperforming_count,
                len(recent_posts),
                avg_score,
            )
            return ResearchTrigger.UNDERPERFORMANCE

        # --- Check for weekly cycle (Sunday) ---
        now = utc_now()
        if now.weekday() == 6:  # Sunday = 6
            last_research = await self.db.get_last_research_date()
            if last_research is None or (now - last_research).days >= 7:
                logger.info(
                    "[RESEARCH] Weekly cycle triggered (Sunday, %s days since last)",
                    "never" if last_research is None
                    else str((now - last_research).days),
                )
                return ResearchTrigger.WEEKLY_CYCLE

        return None

    # -----------------------------------------------------------------
    # MAIN RESEARCH CYCLE
    # -----------------------------------------------------------------

    async def research(
        self,
        trigger: ResearchTrigger,
        timeout_seconds: int = 300,
    ) -> ResearchReport:
        """Execute a full research cycle.

        Builds a set of queries appropriate to the trigger, executes them
        (respecting the overall timeout), gathers findings, and generates
        synthesized recommendations.

        If individual queries fail (timeout or exception), the cycle
        continues with partial results rather than aborting entirely.

        Args:
            trigger: The :class:`ResearchTrigger` that initiated this cycle.
            timeout_seconds: Hard upper bound on wall-clock time for the
                entire cycle (default ``300`` = 5 minutes).

        Returns:
            A :class:`ResearchReport` containing findings and
            recommendations (may be partial if some queries failed).

        Architecture reference: ``architecture.md`` lines 17200-17298.
        """
        started_at = utc_now()
        queries = self._build_queries(trigger)
        findings: List[ResearchFinding] = []
        failed_queries: List[ResearchQuery] = []

        logger.info(
            "[RESEARCH] Starting research cycle -- "
            "trigger=%s, queries=%d, timeout=%ds",
            trigger.value,
            len(queries),
            timeout_seconds,
        )

        try:
            async with asyncio.timeout(timeout_seconds):
                for query in queries:
                    query_start = utc_now()
                    try:
                        if query.source == "perplexity":
                            result = await self._research_perplexity(query)
                        elif query.source == "competitor_scrape":
                            result = await self._research_competitors(query)
                        elif query.source == "own_data":
                            result = await self._research_own_data(query)
                        else:
                            logger.warning(
                                "[RESEARCH] Unknown source: %s", query.source
                            )
                            result = []

                        findings.extend(result)
                        query_duration = (
                            utc_now() - query_start
                        ).total_seconds()
                        logger.debug(
                            "[RESEARCH] Query completed in %.1fs: "
                            "%s -- %d findings",
                            query_duration,
                            query.source,
                            len(result),
                        )

                    except asyncio.TimeoutError:
                        logger.warning(
                            "[RESEARCH] Query timeout: %s -- %s",
                            query.source,
                            query.query[:50],
                        )
                        failed_queries.append(query)
                    except Exception as exc:
                        logger.warning(
                            "[RESEARCH] Query failed: %s -- %s",
                            query.source,
                            exc,
                        )
                        failed_queries.append(query)
                        # Continue with remaining queries

        except asyncio.TimeoutError:
            logger.warning(
                "[RESEARCH] Overall timeout reached (%ds). Returning "
                "partial results: %d findings from %d/%d queries",
                timeout_seconds,
                len(findings),
                len(queries) - len(failed_queries),
                len(queries),
            )

        # Generate recommendations from findings (even if partial)
        if findings:
            recommendations = await self._generate_recommendations(findings)
        else:
            logger.warning(
                "[RESEARCH] No findings -- skipping recommendation generation"
            )
            recommendations = []

        completed_at = utc_now()
        total_duration = (completed_at - started_at).total_seconds()

        # Build set of successfully executed queries
        failed_set = set(id(q) for q in failed_queries)
        executed = [q for q in queries if id(q) not in failed_set]

        report = ResearchReport(
            trigger=trigger,
            trigger_reason=self._get_trigger_reason(trigger),
            started_at=started_at,
            completed_at=completed_at,
            queries_executed=executed,
            findings=findings,
            recommendations=recommendations,
            total_sources_consulted=len(executed),
            confidence_score=self._calculate_confidence(findings),
        )

        logger.info(
            "[RESEARCH] Research cycle complete in %.1fs -- "
            "findings=%d, recommendations=%d, failed_queries=%d",
            total_duration,
            len(findings),
            len(recommendations),
            len(failed_queries),
        )

        return report

    # -----------------------------------------------------------------
    # PERPLEXITY WEB RESEARCH
    # -----------------------------------------------------------------

    async def _research_perplexity(
        self, query: ResearchQuery
    ) -> List[ResearchFinding]:
        """Search the web via Perplexity and extract actionable insights.

        Calls the Perplexity search API, then uses Claude to distil the
        raw search results into structured :class:`ResearchFinding` objects.

        Args:
            query: The research query to execute.

        Returns:
            List of findings (may be empty if extraction fails).

        Architecture reference: ``architecture.md`` lines 17300-17347.
        """
        response = await self.perplexity.search(query.query)
        text_content = PerplexityClient.extract_text(response)

        if not text_content:
            logger.warning(
                "[RESEARCH] Empty response from Perplexity for query: %s",
                query.query[:50],
            )
            return []

        extraction_prompt = (
            f"Research query: {query.query}\n"
            f"Purpose: {query.purpose}\n\n"
            f"Search results:\n{text_content}\n\n"
            "Extract 3-5 actionable insights for improving LinkedIn posts.\n"
            "For each insight, return a JSON object with these keys:\n"
            '  "finding": what was discovered,\n'
            '  "source": where it came from (URL or description),\n'
            '  "confidence": 0-1 confidence score,\n'
            '  "actionable": true/false,\n'
            '  "suggested_change": specific change to make (or null),\n'
            '  "affected_component": which component to change '
            "(writer/trend_scout/visual_creator/scheduler)\n\n"
            "Return as a JSON array of objects."
        )

        try:
            raw = await self.claude.generate_structured(
                prompt=extraction_prompt,
            )
            # Handle both list and dict-with-list responses
            items = raw if isinstance(raw, list) else raw.get("insights", [])

            findings: List[ResearchFinding] = []
            for item in items:
                findings.append(
                    ResearchFinding(
                        finding=item.get("finding", ""),
                        source=item.get("source", "perplexity"),
                        confidence=float(item.get("confidence", 0.5)),
                        actionable=bool(item.get("actionable", True)),
                        suggested_change=item.get("suggested_change"),
                        affected_component=item.get(
                            "affected_component", "writer"
                        ),
                    )
                )
            return findings

        except Exception as exc:
            logger.error(
                "[RESEARCH] Claude extraction failed for Perplexity results: %s",
                exc,
            )
            return []

    # -----------------------------------------------------------------
    # COMPETITOR ANALYSIS
    # -----------------------------------------------------------------

    async def _research_competitors(
        self, query: ResearchQuery
    ) -> List[ResearchFinding]:
        """Scrape competitor LinkedIn posts and analyze patterns.

        Fetches recent posts from each competitor profile (with rate
        limiting of 7 seconds between requests), sorts by engagement,
        then uses Claude to analyze both text patterns and visual patterns.

        Args:
            query: The research query (the ``query`` field is informational).

        Returns:
            Combined list of text and visual pattern findings.

        Architecture reference: ``architecture.md`` lines 17349-17404.
        """
        # Rate limiting: 7s between LinkedIn requests to stay within limits
        REQUEST_DELAY_SECONDS = 7.0

        all_posts: List[Dict[str, Any]] = []

        for i, competitor_url in enumerate(self.competitors):
            try:
                posts = await self.linkedin.get_recent_posts(
                    profile_url=competitor_url,
                    limit=10,
                )
                if isinstance(posts, list):
                    all_posts.extend(posts)
                logger.debug(
                    "[RESEARCH] Fetched %d posts from competitor %d/%d",
                    len(posts) if isinstance(posts, list) else 0,
                    i + 1,
                    len(self.competitors),
                )

                # Rate limiting -- skip delay after last request
                if i < len(self.competitors) - 1:
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)

            except Exception as exc:
                logger.warning(
                    "[RESEARCH] Failed to fetch posts from %s: %s",
                    competitor_url,
                    exc,
                )
                continue

        if not all_posts:
            logger.warning("[RESEARCH] No competitor posts fetched")
            return []

        # Sort by engagement (comments weighted 3x over likes)
        def _engagement(post: Dict[str, Any]) -> int:
            likes = post.get("likes", 0)
            comments = post.get("comments", 0)
            return likes + comments * 3

        top_posts = sorted(all_posts, key=_engagement, reverse=True)[:20]

        # Analyze text patterns via Claude
        text_data = [
            {
                "text": str(p.get("text", ""))[:500],
                "likes": p.get("likes", 0),
                "comments": p.get("comments", 0),
            }
            for p in top_posts
        ]

        text_analysis_prompt = (
            "Analyze these top-performing LinkedIn posts from influencers:\n\n"
            f"{json.dumps(text_data, ensure_ascii=False)}\n\n"
            "Find patterns:\n"
            "1. Hook styles that work (first 2 lines)\n"
            "2. Post structure patterns\n"
            "3. Common topics/angles\n"
            "4. Call-to-action patterns\n\n"
            "Return a JSON array of finding objects, each with keys:\n"
            '  "finding", "source", "confidence", "actionable", '
            '"suggested_change", "affected_component"'
        )

        text_findings: List[ResearchFinding] = []
        try:
            raw = await self.claude.generate_structured(
                prompt=text_analysis_prompt,
            )
            items = raw if isinstance(raw, list) else raw.get("findings", [])
            for item in items:
                text_findings.append(
                    ResearchFinding(
                        finding=item.get("finding", ""),
                        source=item.get("source", "competitor_analysis"),
                        confidence=float(item.get("confidence", 0.6)),
                        actionable=bool(item.get("actionable", True)),
                        suggested_change=item.get("suggested_change"),
                        affected_component=item.get(
                            "affected_component", "writer"
                        ),
                    )
                )
        except Exception as exc:
            logger.error(
                "[RESEARCH] Text pattern analysis failed: %s", exc
            )

        # Analyze visual patterns via Claude
        visual_findings = await self._analyze_competitor_visuals(top_posts)

        return text_findings + visual_findings

    async def _analyze_competitor_visuals(
        self, posts: List[Dict[str, Any]]
    ) -> List[ResearchFinding]:
        """Analyze visual patterns from competitor posts.

        Extracts data about visual types, styles, and engagement
        correlation, then uses Claude to identify actionable visual
        strategy insights.

        Args:
            posts: List of competitor post dicts (should have visual metadata).

        Returns:
            List of visual pattern findings.

        Architecture reference: ``architecture.md`` lines 17406-17500.
        """
        visual_data = []
        for p in posts:
            visual_data.append(
                {
                    "visual_type": p.get("visual_type", "unknown"),
                    "has_visual": p.get("has_visual", False),
                    "likes": p.get("likes", 0),
                    "comments": p.get("comments", 0),
                }
            )

        prompt = (
            "Analyze visual patterns in these LinkedIn posts:\n\n"
            f"{json.dumps(visual_data, ensure_ascii=False)}\n\n"
            "Identify:\n"
            "1. Which visual types correlate with higher engagement?\n"
            "2. Do posts with visuals outperform those without?\n"
            "3. Any visual style recommendations?\n\n"
            "Return a JSON array of finding objects, each with keys:\n"
            '  "finding", "source", "confidence", "actionable", '
            '"suggested_change", "affected_component"'
        )

        try:
            raw = await self.claude.generate_structured(prompt=prompt)
            items = raw if isinstance(raw, list) else raw.get("findings", [])
            findings: List[ResearchFinding] = []
            for item in items:
                findings.append(
                    ResearchFinding(
                        finding=item.get("finding", ""),
                        source=item.get("source", "competitor_visuals"),
                        confidence=float(item.get("confidence", 0.5)),
                        actionable=bool(item.get("actionable", True)),
                        suggested_change=item.get("suggested_change"),
                        affected_component=item.get(
                            "affected_component", "visual_creator"
                        ),
                    )
                )
            return findings
        except Exception as exc:
            logger.error(
                "[RESEARCH] Visual pattern analysis failed: %s", exc
            )
            return []

    # -----------------------------------------------------------------
    # OWN DATA ANALYSIS
    # -----------------------------------------------------------------

    async def _research_own_data(
        self, query: ResearchQuery
    ) -> List[ResearchFinding]:
        """Analyze own post performance data for patterns.

        Compares the top 10 and bottom 10 posts by engagement to identify
        what differentiates high-performing content from low-performing
        content across hooks, content types, visual choices, and timing.

        Args:
            query: The research query (the ``query`` field is informational).

        Returns:
            List of findings about own performance patterns.

        Architecture reference: ``architecture.md`` lines 14368-14370.
        """
        # Fetch recent posts (enough to get top/bottom 10)
        all_posts = await self.db.get_recent_posts(limit=30)

        if len(all_posts) < 5:
            logger.info(
                "[RESEARCH] Not enough own posts for analysis (%d posts)",
                len(all_posts),
            )
            return []

        # Sort by a composite score: qc_score or likes + comments * 3
        def _score(post: Dict[str, Any]) -> float:
            qc = post.get("qc_score", post.get("score", 0))
            if qc:
                return float(qc)
            return float(post.get("likes", 0) + post.get("comments", 0) * 3)

        sorted_posts = sorted(all_posts, key=_score, reverse=True)
        top_10 = sorted_posts[:10]
        bottom_10 = sorted_posts[-10:]

        # Prepare summaries for Claude
        def _summarize(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            summaries = []
            for p in posts:
                summaries.append(
                    {
                        "text_preview": str(
                            p.get("text_content", p.get("full_text", ""))
                        )[:300],
                        "content_type": p.get("content_type", "unknown"),
                        "score": _score(p),
                        "likes": p.get("likes", 0),
                        "comments": p.get("comments", 0),
                        "visual_type": p.get("visual_type", "unknown"),
                    }
                )
            return summaries

        analysis_prompt = (
            "Compare these TOP-performing and BOTTOM-performing LinkedIn "
            "posts to find patterns:\n\n"
            f"TOP 10 POSTS:\n{json.dumps(_summarize(top_10), ensure_ascii=False)}\n\n"
            f"BOTTOM 10 POSTS:\n{json.dumps(_summarize(bottom_10), ensure_ascii=False)}\n\n"
            "Analyze differences in:\n"
            "1. Hook styles (first line)\n"
            "2. Content types that perform best/worst\n"
            "3. Visual types that correlate with engagement\n"
            "4. Post length and structure\n\n"
            "Return a JSON array of finding objects, each with keys:\n"
            '  "finding", "source", "confidence", "actionable", '
            '"suggested_change", "affected_component"'
        )

        try:
            raw = await self.claude.generate_structured(
                prompt=analysis_prompt,
            )
            items = raw if isinstance(raw, list) else raw.get("findings", [])
            findings: List[ResearchFinding] = []
            for item in items:
                findings.append(
                    ResearchFinding(
                        finding=item.get("finding", ""),
                        source=item.get("source", "own_data_analysis"),
                        confidence=float(item.get("confidence", 0.7)),
                        actionable=bool(item.get("actionable", True)),
                        suggested_change=item.get("suggested_change"),
                        affected_component=item.get(
                            "affected_component", "writer"
                        ),
                    )
                )
            return findings
        except Exception as exc:
            logger.error(
                "[RESEARCH] Own data analysis failed: %s", exc
            )
            return []

    # -----------------------------------------------------------------
    # QUERY BUILDING
    # -----------------------------------------------------------------

    def _build_queries(self, trigger: ResearchTrigger) -> List[ResearchQuery]:
        """Build a list of research queries based on the trigger type.

        Each trigger type produces a different set of queries targeting
        the most relevant sources and questions.

        Args:
            trigger: The :class:`ResearchTrigger` that initiated research.

        Returns:
            List of :class:`ResearchQuery` objects to execute.
        """
        if trigger == ResearchTrigger.UNDERPERFORMANCE:
            return [
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn post engagement best practices 2025",
                    purpose="Find current best practices for improving engagement",
                    priority=1,
                ),
                ResearchQuery(
                    source="perplexity",
                    query="Why LinkedIn posts fail low engagement causes",
                    purpose="Identify common reasons for underperformance",
                    priority=1,
                ),
                ResearchQuery(
                    source="competitor_scrape",
                    query="Analyze top competitor hooks and structures",
                    purpose="Learn from high-performing competitor content",
                    priority=2,
                ),
                ResearchQuery(
                    source="own_data",
                    query="Compare top vs bottom performing posts",
                    purpose="Find internal patterns of success and failure",
                    priority=1,
                ),
            ]

        elif trigger == ResearchTrigger.WEEKLY_CYCLE:
            return [
                ResearchQuery(
                    source="perplexity",
                    query=(
                        "LinkedIn algorithm changes content strategy "
                        "updates this week"
                    ),
                    purpose="Stay current with platform algorithm changes",
                    priority=1,
                ),
                ResearchQuery(
                    source="perplexity",
                    query="Best LinkedIn post hooks engagement tactics 2025",
                    purpose="Refresh knowledge of engagement techniques",
                    priority=2,
                ),
                ResearchQuery(
                    source="competitor_scrape",
                    query="Weekly competitor post analysis",
                    purpose="Track competitor strategy evolution",
                    priority=2,
                ),
                ResearchQuery(
                    source="own_data",
                    query="Weekly performance review",
                    purpose="Assess weekly trends in own content performance",
                    priority=2,
                ),
            ]

        elif trigger == ResearchTrigger.NEW_CONTENT_TYPE:
            return [
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn best practices for AI content types",
                    purpose="Learn strategies for unfamiliar content types",
                    priority=1,
                ),
                ResearchQuery(
                    source="competitor_scrape",
                    query="Competitor content type analysis",
                    purpose="See how competitors handle various content types",
                    priority=2,
                ),
            ]

        elif trigger == ResearchTrigger.ALGORITHM_CHANGE:
            return [
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn algorithm changes 2025 impact on reach",
                    purpose="Understand recent algorithm changes",
                    priority=1,
                ),
                ResearchQuery(
                    source="perplexity",
                    query="How to adapt LinkedIn strategy to algorithm updates",
                    purpose="Find adaptation strategies",
                    priority=1,
                ),
            ]

        else:
            # Default / MANUAL_REQUEST -- broad research
            return [
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn content strategy best practices 2025",
                    purpose="General strategy refresh",
                    priority=2,
                ),
                ResearchQuery(
                    source="competitor_scrape",
                    query="General competitor analysis",
                    purpose="Baseline competitor intelligence",
                    priority=3,
                ),
                ResearchQuery(
                    source="own_data",
                    query="General performance review",
                    purpose="Baseline performance assessment",
                    priority=3,
                ),
            ]

    # -----------------------------------------------------------------
    # TRIGGER DESCRIPTION
    # -----------------------------------------------------------------

    @staticmethod
    def _get_trigger_reason(trigger: ResearchTrigger) -> str:
        """Return a human-readable description of the trigger.

        Args:
            trigger: The research trigger to describe.

        Returns:
            Descriptive string for logging and report metadata.
        """
        descriptions: Dict[ResearchTrigger, str] = {
            ResearchTrigger.UNDERPERFORMANCE: (
                "3+ of last 5 posts scored below 80% of the historical "
                "average, indicating a potential content strategy issue."
            ),
            ResearchTrigger.WEEKLY_CYCLE: (
                "Scheduled weekly deep research cycle (Sunday). Keeps "
                "strategies fresh and incorporates new best practices."
            ),
            ResearchTrigger.NEW_CONTENT_TYPE: (
                "Encountered a content type with no historical data. "
                "Research needed to establish baseline strategy."
            ),
            ResearchTrigger.ALGORITHM_CHANGE: (
                "Detected a shift in engagement patterns that may indicate "
                "a LinkedIn algorithm change."
            ),
            ResearchTrigger.MANUAL_REQUEST: (
                "Research requested manually by a human operator."
            ),
            ResearchTrigger.EVERY_ITERATION: (
                "Continuous learning trigger -- standard iteration research."
            ),
            ResearchTrigger.FIRST_POST: (
                "First-post bootstrap -- researching initial best practices."
            ),
            ResearchTrigger.COMPONENT_FEEDBACK: (
                "Specific pipeline component received negative feedback."
            ),
        }
        return descriptions.get(trigger, f"Research triggered by {trigger.value}")

    # -----------------------------------------------------------------
    # CONFIDENCE CALCULATION
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(findings: List[ResearchFinding]) -> float:
        """Calculate overall confidence in the research findings.

        Uses a weighted average of individual finding confidences, where
        actionable findings are weighted higher (1.5x) than non-actionable
        ones.

        Args:
            findings: All findings from the research cycle.

        Returns:
            Overall confidence score clamped to [0.0, 1.0].
            Returns 0.0 if there are no findings.
        """
        if not findings:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for finding in findings:
            weight = 1.5 if finding.actionable else 1.0
            weighted_sum += finding.confidence * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        return min(1.0, max(0.0, weighted_sum / total_weight))

    # -----------------------------------------------------------------
    # RECOMMENDATION SYNTHESIS
    # -----------------------------------------------------------------

    async def _generate_recommendations(
        self, findings: List[ResearchFinding]
    ) -> List[ResearchRecommendation]:
        """Synthesize findings into 3-5 actionable recommendations.

        Uses Claude to analyze all findings holistically and produce
        a prioritized list of concrete changes for pipeline components.

        Args:
            findings: All findings from the research cycle.

        Returns:
            List of 3-5 :class:`ResearchRecommendation` objects,
            sorted by priority.
        """
        findings_data = [
            {
                "finding": f.finding,
                "source": f.source,
                "confidence": f.confidence,
                "actionable": f.actionable,
                "suggested_change": f.suggested_change,
                "affected_component": f.affected_component,
            }
            for f in findings
        ]

        prompt = (
            "Based on these research findings, generate 3-5 actionable "
            "recommendations for improving LinkedIn content quality.\n\n"
            f"FINDINGS:\n{json.dumps(findings_data, ensure_ascii=False)}\n\n"
            "For each recommendation, return a JSON object with:\n"
            '  "component": target component (writer/trend_scout/'
            "visual_creator/humanizer/qc_agent/scheduler),\n"
            '  "change": what should be changed,\n'
            '  "priority": 1-5 (1 = highest),\n'
            '  "rationale": why this change is recommended,\n'
            '  "confidence": 0-1 confidence score,\n'
            '  "estimated_impact": "high"/"medium"/"low",\n'
            '  "source_findings": list of finding texts that support this\n\n'
            "Return as a JSON array of 3-5 objects, sorted by priority."
        )

        try:
            raw = await self.claude.generate_structured(prompt=prompt)
            items = raw if isinstance(raw, list) else raw.get(
                "recommendations", []
            )

            recommendations: List[ResearchRecommendation] = []
            for item in items:
                recommendations.append(
                    ResearchRecommendation(
                        component=item.get("component", "writer"),
                        change=item.get("change", ""),
                        priority=int(item.get("priority", 3)),
                        rationale=item.get("rationale", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        estimated_impact=item.get("estimated_impact", "medium"),
                        source_findings=item.get("source_findings", []),
                    )
                )

            # Sort by priority (ascending = highest priority first)
            recommendations.sort(key=lambda r: r.priority)
            return recommendations[:5]

        except Exception as exc:
            logger.error(
                "[RESEARCH] Recommendation generation failed: %s", exc
            )
            return []


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "ResearchAgent",
]
