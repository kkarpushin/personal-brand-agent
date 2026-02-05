"""
Analyzer Agent -- deep analysis of a selected topic.

Takes a ``TrendTopic`` + ``ContentType`` and produces an ``AnalysisBrief``
with type-specific extraction data, key findings, hook materials, and
strategic recommendations for the Writer and Visual Creator agents.

The Analyzer is the bridge between raw topic discovery (Trend Scout) and
content creation (Writer).  It uses:
- **Perplexity** for additional real-time research context
- **Claude** for structured extraction and insight generation

Architecture reference: ``architecture.md`` lines 5236-6529
Error philosophy: NO FALLBACKS -- fail fast, retry transient errors only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.exceptions import AnalyzerError, BoundaryValidationError
from src.models import (
    AnalysisBrief,
    ContentType,
    TrendTopic,
    TypeSpecificExtraction,
)
from src.tools.claude_client import ClaudeClient
from src.tools.perplexity import PerplexityClient
from src.utils import generate_id, with_retry

logger = logging.getLogger("Analyzer")


# =============================================================================
# TYPE-SPECIFIC ANALYSIS PROMPTS
#
# Each ContentType has a dedicated extraction prompt that instructs Claude to
# return structured JSON with the fields required by downstream agents.
#
# Architecture reference: architecture.md lines 5462-5898
# =============================================================================

EXTRACTION_PROMPTS: Dict[ContentType, str] = {
    ContentType.ENTERPRISE_CASE: """\
Analyze this enterprise AI implementation case study.

Content: {content}
Source: {source_url}

Extract the following as JSON (use "NOT_FOUND" for missing values):

{{
    "company": "Company name",
    "industry": "Industry sector",
    "scale": "SMB | Mid-Market | Enterprise | Fortune 500",
    "problem_statement": "What problem were they solving?",
    "solution_description": "What AI solution was implemented?",
    "ai_technologies": ["specific models, platforms used"],
    "implementation_timeline": "How long it took",
    "metrics_extracted": {{"kpi_name": "value", ...}},
    "roi_stated": "ROI or cost savings if mentioned",
    "lessons_learned": ["lesson 1", "lesson 2", ...],
    "architecture_notes": "Architecture overview if available",
    "team_size": "Team structure if mentioned"
}}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
""",
    ContentType.PRIMARY_SOURCE: """\
Analyze this research paper, report, or expert essay.

Content: {content}
Source: {source_url}

Extract the following as JSON (use "NOT_FOUND" for missing values):

{{
    "authors": ["Author 1", "Author 2"],
    "organization": "Institution or publisher",
    "source_type": "research_paper | think_tank_report | expert_essay | whitepaper",
    "thesis": "Main argument or hypothesis (1-2 sentences)",
    "methodology": "How they arrived at these conclusions",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "counterintuitive_finding": "Most surprising finding, or null",
    "implications": ["Implication for practitioners", "For business leaders", "For the field"],
    "limitations": "Acknowledged limitations",
    "future_directions": "Suggested future research"
}}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
""",
    ContentType.AUTOMATION_CASE: """\
Analyze this AI automation or agent case study.

Content: {content}
Source: {source_url}

Extract the following as JSON (use "NOT_FOUND" for missing values):

{{
    "task_automated": "What task or process was automated?",
    "tools_used": ["Tool 1", "Tool 2"],
    "workflow_steps": ["Step 1", "Step 2", "Step 3"],
    "time_saved": "Time savings if mentioned",
    "cost_saved": "Cost savings if mentioned",
    "reproducibility_notes": "Can others replicate this? How?",
    "code_snippets": ["Any code examples shared"],
    "gotchas": ["Common pitfalls mentioned"]
}}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
""",
    ContentType.COMMUNITY_CONTENT: """\
Analyze this community discussion, video, or thread.

Content: {content}
Source: {source_url}

Extract the following as JSON (use "NOT_FOUND" for missing values):

{{
    "platform": "YouTube | Reddit | HackerNews | Dev.to | Twitter | Medium | Substack",
    "main_discussion_topic": "What the discussion is about",
    "key_viewpoints": ["Viewpoint 1", "Viewpoint 2", "Viewpoint 3"],
    "engagement_level": "high | medium | low",
    "best_insight_shared": "Most valuable insight from the discussion",
    "most_controversial_take": "Hottest debate point",
    "practical_tip_mentioned": "Actionable advice shared",
    "notable_contributions": ["Notable contributor or insight"],
    "consensus_points": ["What people agree on"],
    "disagreement_points": ["Where experts disagree"],
    "linked_resources": ["URLs or resources referenced"]
}}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
""",
    ContentType.TOOL_RELEASE: """\
Analyze this AI tool or product release.

Content: {content}
Source: {source_url}

Extract the following as JSON (use "NOT_FOUND" for missing values):

{{
    "tool_name": "Product/tool name",
    "company": "Company releasing it",
    "release_date": "When it was released",
    "release_type": "new_product | major_update | api_release | open_source",
    "core_functionality": "Main purpose/function",
    "key_features": ["Feature 1", "Feature 2", "Feature 3"],
    "target_users": "Who this is for",
    "availability": "free | paid | freemium | waitlist",
    "pricing_info": "Pricing details if available",
    "competitive_position": "How it compares to alternatives",
    "limitations": ["Known limitations"],
    "demo_url": "Demo or trial URL if available",
    "early_feedback": ["Early user feedback or expert opinions"]
}}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
""",
}


# =============================================================================
# INSIGHT EXTRACTION PROMPT
# =============================================================================

INSIGHTS_PROMPT = """\
Based on the following analysis of a {content_type} topic, extract 3-5 key
insights that would be valuable for a LinkedIn audience of tech professionals
and business leaders.

Topic: {title}
Analysis summary:
{extraction_summary}

For each insight, provide:
- "finding": A compelling one-sentence statement
- "explanation": Why it matters (1-2 sentences)
- "wow_factor": Rating 1-10 of how surprising/valuable this is

Return as JSON array:
[
    {{
        "finding": "...",
        "explanation": "...",
        "wow_factor": 8
    }},
    ...
]

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
"""


# =============================================================================
# HOOK MATERIALS PROMPT
# =============================================================================

HOOK_MATERIALS_PROMPT = """\
Generate hook materials for a LinkedIn post about this topic.

Content type: {content_type}
Topic: {title}
Key findings: {key_findings}
Extraction data: {extraction_data}

Provide:
{{
    "suggested_angle": "The most compelling angle for a LinkedIn post",
    "controversy_level": "low | medium | high | spicy",
    "debate_angles": ["Potential debate angle 1", "Potential debate angle 2"],
    "complexity_level": "simplify_heavily | simplify_slightly | keep_technical",
    "target_audience": "Who should read this",
    "recommended_post_format": "insight_thread | contrarian | tutorial_light | \
list_post | metrics_story | case_study | first_look",
    "recommended_visual_type": "data_viz | diagram | screenshot | quote_card | \
author_photo | carousel",
    "key_quotes": ["Quotable line 1", "Quotable line 2"],
    "statistics": ["Stat 1", "Stat 2"]
}}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.\
"""


# =============================================================================
# REQUIRED FIELDS BY CONTENT TYPE
#
# These are the extraction fields that MUST be present for the brief to be
# considered complete.  The Analyzer raises AnalyzerError if any are missing.
#
# Architecture reference: architecture.md lines 6322-6378
# =============================================================================

REQUIRED_EXTRACTION_FIELDS: Dict[ContentType, List[str]] = {
    ContentType.ENTERPRISE_CASE: [
        "company",
        "problem_statement",
        "solution_description",
        "metrics_extracted",
    ],
    ContentType.PRIMARY_SOURCE: [
        "authors",
        "thesis",
        "key_findings",
    ],
    ContentType.AUTOMATION_CASE: [
        "task_automated",
        "tools_used",
        "workflow_steps",
    ],
    ContentType.COMMUNITY_CONTENT: [
        "platform",
        "key_viewpoints",
    ],
    ContentType.TOOL_RELEASE: [
        "tool_name",
        "key_features",
        "target_users",
    ],
}


# =============================================================================
# ANALYZER AGENT
# =============================================================================


class AnalyzerAgent:
    """
    Deep analysis agent that extracts structured data from topics.

    Uses type-specific extraction strategies based on ``ContentType`` to
    produce an ``AnalysisBrief`` for the Writer agent.  The analysis
    pipeline consists of:

    1. Additional context gathering via Perplexity search
    2. Type-specific structured extraction via Claude
    3. Key insight distillation
    4. Hook material and strategic recommendation generation
    5. Required-field validation (fail-fast)

    Args:
        claude: The Claude LLM client for structured extraction.
        perplexity: The Perplexity client for additional web research.
    """

    def __init__(self, claude: ClaudeClient, perplexity: PerplexityClient) -> None:
        self.claude = claude
        self.perplexity = perplexity
        self.logger = logging.getLogger("Analyzer")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        topic: TrendTopic,
        content_type: ContentType,
        extraction_focus: List[str],
        required_fields: List[str],
    ) -> AnalysisBrief:
        """
        Analyze topic and produce an ``AnalysisBrief``.

        Args:
            topic: The selected trend topic from Trend Scout.
            content_type: Classified content type for the topic.
            extraction_focus: Fields to prioritise during extraction
                (from ``type_context``).
            required_fields: Mandatory fields that must be present
                in the extraction. If any are missing after analysis,
                ``AnalyzerError`` is raised.

        Returns:
            Populated ``AnalysisBrief`` ready for the Writer agent.

        Raises:
            AnalyzerError: If required fields cannot be extracted or
                any step of the analysis pipeline fails unrecoverably.
        """
        self.logger.info(
            "[ANALYZER] Starting analysis: topic_id=%s, type=%s",
            topic.id,
            content_type.value,
        )

        # Step 1: Gather additional context via Perplexity
        additional_context = await self._gather_context(topic)

        # Step 2: Run type-specific extraction via Claude
        combined_content = self._build_analysis_content(topic, additional_context)
        extraction_data = await self._extract_type_specific(
            content=combined_content,
            source_url=topic.primary_source_url,
            content_type=content_type,
        )

        # Step 3: Validate required extraction fields (fail-fast)
        self._validate_required_fields(
            extraction_data=extraction_data,
            required_fields=required_fields,
            content_type=content_type,
            topic_id=topic.id,
        )

        # Step 4: Extract key insights
        key_findings = await self._extract_insights(
            title=topic.title,
            content_type=content_type,
            extraction_data=extraction_data,
        )

        # Step 5: Generate hook materials and strategic recommendations
        hook_materials = await self._generate_hook_materials(
            title=topic.title,
            content_type=content_type,
            key_findings=key_findings,
            extraction_data=extraction_data,
        )

        # Step 6: Compute extraction completeness
        all_required = REQUIRED_EXTRACTION_FIELDS.get(content_type, [])
        present_fields = [
            f for f in all_required
            if self._field_is_present(extraction_data, f)
        ]
        missing_fields = [
            f for f in all_required
            if not self._field_is_present(extraction_data, f)
        ]
        completeness = len(present_fields) / max(len(all_required), 1)

        # Step 7: Build the TypeSpecificExtraction wrapper
        type_extraction = TypeSpecificExtraction(
            content_type=content_type,
            extracted_fields=extraction_data,
            required_fields_present=present_fields,
            missing_fields=missing_fields,
            extraction_confidence=completeness,
        )

        # Step 8: Build and return the AnalysisBrief
        brief = AnalysisBrief(
            topic_id=topic.id,
            content_type=content_type,
            title=topic.title,
            key_findings=key_findings,
            main_argument=hook_materials.get("suggested_angle", ""),
            suggested_angle=hook_materials.get("suggested_angle", ""),
            hook_materials=hook_materials,
            extraction_data=type_extraction,
            controversy_level=hook_materials.get("controversy_level", "low"),
            complexity_level=hook_materials.get(
                "complexity_level", "simplify_slightly"
            ),
            target_audience=hook_materials.get(
                "target_audience", "AI practitioners and tech leaders"
            ),
            recommended_post_format=hook_materials.get(
                "recommended_post_format",
                topic.recommended_post_format,
            ),
            recommended_visual_type=hook_materials.get(
                "recommended_visual_type",
                topic.recommended_visual_type,
            ),
        )

        self.logger.info(
            "[ANALYZER] Analysis complete: topic_id=%s, findings=%d, "
            "completeness=%.2f, controversy=%s",
            topic.id,
            len(key_findings),
            completeness,
            brief.controversy_level,
        )
        return brief

    # ------------------------------------------------------------------
    # Step 1: Additional context gathering
    # ------------------------------------------------------------------

    async def _gather_context(self, topic: TrendTopic) -> str:
        """
        Use Perplexity to gather additional context about the topic.

        Returns concatenated text from Perplexity search results.
        If Perplexity fails entirely, falls back to an empty string
        (additional context is supplementary, not required).
        """
        self.logger.debug(
            "[ANALYZER] Gathering additional context for: %s", topic.title
        )
        try:
            results = await self.perplexity.research_topic(
                topic=topic.title,
                num_queries=2,
            )
            texts: List[str] = []
            for result in results:
                text = PerplexityClient.extract_text(result)
                if text:
                    texts.append(text)
            combined = "\n\n---\n\n".join(texts)
            self.logger.debug(
                "[ANALYZER] Perplexity returned %d results, %d chars",
                len(results),
                len(combined),
            )
            return combined
        except Exception as exc:
            # Perplexity context is supplementary -- log and continue
            # with just the raw topic content.  This is NOT a fallback
            # (the analysis still proceeds), just graceful handling of
            # an optional enrichment step.
            self.logger.warning(
                "[ANALYZER] Perplexity research failed, proceeding with "
                "topic raw_content only: %s",
                exc,
            )
            return ""

    # ------------------------------------------------------------------
    # Content preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_analysis_content(topic: TrendTopic, additional_context: str) -> str:
        """
        Combine topic raw content with Perplexity research into a single
        analysis payload.  Truncates to a reasonable size for the LLM
        context window.
        """
        parts: List[str] = []
        parts.append(f"TITLE: {topic.title}")
        parts.append(f"SUMMARY: {topic.summary}")

        if topic.raw_content:
            # Limit raw content to ~8000 chars to leave room for prompt
            parts.append(f"SOURCE CONTENT:\n{topic.raw_content[:8000]}")

        if additional_context:
            parts.append(
                f"ADDITIONAL RESEARCH:\n{additional_context[:4000]}"
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 2: Type-specific extraction
    # ------------------------------------------------------------------

    async def _extract_type_specific(
        self,
        content: str,
        source_url: str,
        content_type: ContentType,
    ) -> Dict[str, Any]:
        """
        Run the type-specific extraction prompt through Claude.

        Returns the parsed extraction data as a dict.

        Raises:
            AnalyzerError: If Claude returns invalid JSON or the
                extraction fails after retries.
        """
        prompt_template = EXTRACTION_PROMPTS.get(content_type)
        if prompt_template is None:
            raise AnalyzerError(
                f"No extraction prompt defined for content type: "
                f"{content_type.value}"
            )

        prompt = prompt_template.format(
            content=content,
            source_url=source_url,
        )

        self.logger.debug(
            "[ANALYZER] Running type-specific extraction: %s",
            content_type.value,
        )

        try:
            data = await self.claude.generate_structured(
                prompt=prompt,
                system=(
                    "You are an expert analyst. Extract structured data "
                    "from the provided content precisely. Return only "
                    "valid JSON."
                ),
                max_tokens=4096,
            )
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                f"Claude returned invalid JSON during {content_type.value} "
                f"extraction: {exc}"
            ) from exc
        except Exception as exc:
            raise AnalyzerError(
                f"Type-specific extraction failed for "
                f"{content_type.value}: {exc}"
            ) from exc

        # Clean NOT_FOUND sentinel values
        data = self._clean_not_found(data)

        self.logger.debug(
            "[ANALYZER] Extraction completed: %d fields extracted",
            len(data),
        )
        return data

    # ------------------------------------------------------------------
    # Step 3: Key insight extraction
    # ------------------------------------------------------------------

    async def _extract_insights(
        self,
        title: str,
        content_type: ContentType,
        extraction_data: Dict[str, Any],
    ) -> List[str]:
        """
        Distil the extraction data into 3-5 key findings.

        Returns a list of finding strings suitable for ``AnalysisBrief.key_findings``.

        Raises:
            AnalyzerError: If insight extraction fails.
        """
        extraction_summary = json.dumps(extraction_data, indent=2, default=str)
        # Truncate to fit in context window
        if len(extraction_summary) > 6000:
            extraction_summary = extraction_summary[:6000] + "\n...(truncated)"

        prompt = INSIGHTS_PROMPT.format(
            content_type=content_type.value,
            title=title,
            extraction_summary=extraction_summary,
        )

        try:
            insights_raw = await self.claude.generate_structured(
                prompt=prompt,
                system=(
                    "You are a LinkedIn content strategist. Extract insights "
                    "that are compelling, specific, and actionable."
                ),
                max_tokens=2048,
            )
        except Exception as exc:
            raise AnalyzerError(
                f"Insight extraction failed: {exc}"
            ) from exc

        # Parse insights -- handle both list and dict responses
        findings: List[str] = []
        if isinstance(insights_raw, list):
            for item in insights_raw:
                if isinstance(item, dict) and "finding" in item:
                    findings.append(item["finding"])
                elif isinstance(item, str):
                    findings.append(item)
        elif isinstance(insights_raw, dict) and "insights" in insights_raw:
            for item in insights_raw["insights"]:
                if isinstance(item, dict) and "finding" in item:
                    findings.append(item["finding"])

        if not findings:
            raise AnalyzerError(
                f"No insights could be extracted for topic '{title}'. "
                f"Claude returned: {insights_raw}"
            )

        self.logger.info(
            "[ANALYZER] Extracted %d key findings", len(findings)
        )
        return findings

    # ------------------------------------------------------------------
    # Step 4: Hook materials generation
    # ------------------------------------------------------------------

    async def _generate_hook_materials(
        self,
        title: str,
        content_type: ContentType,
        key_findings: List[str],
        extraction_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate hook materials and strategic recommendations for the
        Writer agent.

        Returns a dict with keys: ``suggested_angle``, ``controversy_level``,
        ``debate_angles``, ``complexity_level``, ``target_audience``,
        ``recommended_post_format``, ``recommended_visual_type``,
        ``key_quotes``, ``statistics``.

        Raises:
            AnalyzerError: If hook material generation fails.
        """
        # Prepare compact representations for the prompt
        findings_str = "\n".join(f"- {f}" for f in key_findings[:5])
        extraction_str = json.dumps(extraction_data, indent=2, default=str)
        if len(extraction_str) > 3000:
            extraction_str = extraction_str[:3000] + "\n...(truncated)"

        prompt = HOOK_MATERIALS_PROMPT.format(
            content_type=content_type.value,
            title=title,
            key_findings=findings_str,
            extraction_data=extraction_str,
        )

        try:
            materials = await self.claude.generate_structured(
                prompt=prompt,
                system=(
                    "You are a LinkedIn growth strategist. Generate "
                    "compelling hook materials optimized for engagement "
                    "and thought leadership."
                ),
                max_tokens=2048,
            )
        except Exception as exc:
            raise AnalyzerError(
                f"Hook materials generation failed: {exc}"
            ) from exc

        if not isinstance(materials, dict):
            raise AnalyzerError(
                f"Expected dict for hook materials, got "
                f"{type(materials).__name__}: {materials}"
            )

        self.logger.debug(
            "[ANALYZER] Hook materials generated: angle='%s', "
            "controversy=%s, format=%s",
            materials.get("suggested_angle", "N/A"),
            materials.get("controversy_level", "N/A"),
            materials.get("recommended_post_format", "N/A"),
        )
        return materials

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_required_fields(
        self,
        extraction_data: Dict[str, Any],
        required_fields: List[str],
        content_type: ContentType,
        topic_id: str,
    ) -> None:
        """
        Validate that all required extraction fields are present and
        non-empty.

        Raises:
            AnalyzerError: If any required field is missing or empty.
                This is FAIL-FAST -- no fallbacks.
        """
        missing: List[str] = []
        for field_name in required_fields:
            if not self._field_is_present(extraction_data, field_name):
                missing.append(field_name)

        if missing:
            raise AnalyzerError(
                f"Required extraction fields missing for "
                f"{content_type.value} topic '{topic_id}': {missing}. "
                f"Available fields: {list(extraction_data.keys())}"
            )

        self.logger.debug(
            "[ANALYZER] Required fields validated: all %d present",
            len(required_fields),
        )

    @staticmethod
    def _field_is_present(data: Dict[str, Any], field_name: str) -> bool:
        """
        Check whether a field is present and non-empty in extraction data.

        A field is considered absent if:
        - It does not exist in the dict
        - Its value is ``None``
        - Its value is an empty string or empty list/dict
        - Its value is the sentinel string ``"NOT_FOUND"``
        """
        value = data.get(field_name)
        if value is None:
            return False
        if isinstance(value, str) and (not value.strip() or value.strip() == "NOT_FOUND"):
            return False
        if isinstance(value, (list, dict)) and not value:
            return False
        return True

    @staticmethod
    def _clean_not_found(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace ``NOT_FOUND`` sentinel strings with ``None`` for
        cleaner downstream handling.
        """
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str) and value.strip() == "NOT_FOUND":
                cleaned[key] = None
            elif isinstance(value, list):
                cleaned[key] = [
                    None if (isinstance(v, str) and v.strip() == "NOT_FOUND") else v
                    for v in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    # ------------------------------------------------------------------
    # Boundary validation (static, for use by orchestrator)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_for_writer(brief: AnalysisBrief) -> List[str]:
        """
        Validate that the brief has all data required by the Writer agent.

        This is called at the Analyzer-to-Writer boundary by the
        orchestrator before routing to the Writer node.

        Args:
            brief: The AnalysisBrief to validate.

        Returns:
            List of validation issue strings.  Empty list means valid.
        """
        issues: List[str] = []

        if not brief.key_findings:
            issues.append("No key findings extracted")

        if not brief.main_argument:
            issues.append("No main argument / suggested angle generated")

        if not brief.extraction_data:
            issues.append("No extraction data present")

        if brief.content_type == ContentType.ENTERPRISE_CASE:
            ext_fields = (
                brief.extraction_data.extracted_fields
                if brief.extraction_data
                else {}
            )
            if not ext_fields.get("company"):
                issues.append("Enterprise case missing company name")
            if not ext_fields.get("metrics_extracted"):
                issues.append("Enterprise case missing metrics")

        elif brief.content_type == ContentType.PRIMARY_SOURCE:
            ext_fields = (
                brief.extraction_data.extracted_fields
                if brief.extraction_data
                else {}
            )
            if not ext_fields.get("thesis"):
                issues.append("Primary source missing thesis")
            if not ext_fields.get("key_findings"):
                issues.append("Primary source missing key findings")

        elif brief.content_type == ContentType.AUTOMATION_CASE:
            ext_fields = (
                brief.extraction_data.extracted_fields
                if brief.extraction_data
                else {}
            )
            if not ext_fields.get("task_automated"):
                issues.append("Automation case missing task_automated")
            if not ext_fields.get("tools_used"):
                issues.append("Automation case missing tools_used")

        elif brief.content_type == ContentType.TOOL_RELEASE:
            ext_fields = (
                brief.extraction_data.extracted_fields
                if brief.extraction_data
                else {}
            )
            if not ext_fields.get("tool_name"):
                issues.append("Tool release missing tool_name")
            if not ext_fields.get("key_features"):
                issues.append("Tool release missing key_features")

        return issues

    @staticmethod
    def validate_for_visual_creator(brief: AnalysisBrief) -> List[str]:
        """
        Validate that the brief has data required by the Visual Creator.

        Args:
            brief: The AnalysisBrief to validate.

        Returns:
            List of validation issue strings.  Empty list means valid.
        """
        issues: List[str] = []

        if not brief.recommended_visual_type:
            issues.append("No recommended_visual_type specified")

        if not brief.extraction_data:
            issues.append("No extraction data for visual content generation")

        return issues


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


async def create_analyzer() -> AnalyzerAgent:
    """
    Factory to create an ``AnalyzerAgent`` with default clients.

    Instantiates ``ClaudeClient`` and ``PerplexityClient`` using
    environment variables for API keys.

    Returns:
        Configured ``AnalyzerAgent`` instance.

    Raises:
        KeyError: If ``ANTHROPIC_API_KEY`` environment variable is missing.
    """
    claude = ClaudeClient()
    perplexity = PerplexityClient()
    return AnalyzerAgent(claude=claude, perplexity=perplexity)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "AnalyzerAgent",
    "create_analyzer",
    "EXTRACTION_PROMPTS",
    "REQUIRED_EXTRACTION_FIELDS",
]
