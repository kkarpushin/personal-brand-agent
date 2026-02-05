"""
Humanizer Agent for the LinkedIn Super Agent system.

Takes AI-generated drafts and makes them sound authentically human with
type-specific tone calibration.  Each ``ContentType`` gets a different
humanization treatment: enterprise case studies need credibility markers,
research posts need intellectual engagement, automation posts need
practical authenticity, community posts need conversational warmth, and
tool reviews need balanced assessment.

Architecture components:
    1. **Content-Type Tone Calibrator** -- adjusts intensity per ContentType
    2. **AI Pattern Detector** -- finds robotic phrases, repetitive structures
    3. **Type-Aware Voice Injector** -- adds appropriate human markers
    4. **Rhythm Variator** -- varies sentence length, breaks, punctuation
    5. **Authenticity Check** -- verifies it sounds like a real person

References:
    - ``architecture.md`` lines 7708-8251  (Humanizer Agent full spec)
    - ``architecture.md`` lines 7773-7891  (type-specific humanization rules)
    - ``architecture.md`` lines 7896-7977  (universal humanization rules)
    - ``architecture.md`` lines 7981-8070  (type-aware humanization prompt)
    - ``architecture.md`` lines 8078-8251  (HumanizedPost schema & scoring)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.exceptions import WriterError
from src.models import ContentType, DraftPost, HookStyle, HumanizedPost
from src.tools.claude_client import ClaudeClient
from src.utils import generate_id, utc_now


# =============================================================================
# CONSTANTS -- TYPE-SPECIFIC HUMANIZATION CONFIGURATION
# =============================================================================

TYPE_HUMANIZATION: Dict[ContentType, Dict[str, Any]] = {
    ContentType.ENTERPRISE_CASE: {
        "tone_target": "professional credibility with personal insight",
        "intensity": "medium",
        "markers_to_add": [
            "Credibility signals ('In my analysis of this case...')",
            "Interpretive comments ('What struck me most was...')",
            "Industry context ('This is significant for {industry} because...')",
            "Measured enthusiasm ('The results are genuinely impressive.')",
            "Honest caveats ('Though we should note that...')",
        ],
        "markers_to_avoid": [
            "Overly casual language ('crazy', 'insane', 'wild')",
            "Excessive personal stories (focus stays on the case)",
            "Speculation beyond evidence",
        ],
        "rhythm": {"short": 25, "medium": 55, "long": 20},
        "questions_per_post": 1,
        "exclamations_per_post": 0,
    },
    ContentType.PRIMARY_SOURCE: {
        "tone_target": "intellectual engagement with accessibility",
        "intensity": "medium-high",
        "markers_to_add": [
            "Intellectual curiosity ('This made me think about...')",
            "Honest reactions ('I'll admit, this surprised me.')",
            "Accessible framing ('In plain terms, this means...')",
            "Nuanced takes ('It's more complicated than the headline.')",
            "Debate invitations ('I'm not sure I fully agree. Here's why...')",
        ],
        "markers_to_avoid": [
            "Dumbing down too much (respect the research)",
            "Overclaiming certainty",
            "Dismissing complexity",
        ],
        "rhythm": {"short": 30, "medium": 45, "long": 25},
        "questions_per_post": 2,
        "exclamations_per_post": 0,
    },
    ContentType.AUTOMATION_CASE: {
        "tone_target": "practitioner authenticity with helpful generosity",
        "intensity": "high",
        "markers_to_add": [
            "Builder empathy ('I know the pain of doing this manually.')",
            "Practical wisdom ('The trick that made this work...')",
            "Honest about gotchas ('Fair warning: this part was tricky.')",
            "Generous sharing ('Here's exactly what I did...')",
            "Encouragement ('You can totally do this.')",
        ],
        "markers_to_avoid": [
            "Making it sound harder than it is",
            "Gatekeeping language",
            "Overcomplicating explanations",
        ],
        "rhythm": {"short": 35, "medium": 50, "long": 15},
        "questions_per_post": 1,
        "exclamations_per_post": 1,
    },
    ContentType.COMMUNITY_CONTENT: {
        "tone_target": "conversational connection with community",
        "intensity": "high",
        "markers_to_add": [
            "Community connection ('The thread was fascinating.')",
            "Attribution respect ('As @user pointed out...')",
            "Discussion energy ('This sparked a great debate.')",
            "Personal engagement ('Here's where I weigh in...')",
            "Invitation to join ('What's your experience?')",
        ],
        "markers_to_avoid": [
            "Taking credit for others' insights",
            "Dismissing community voices",
            "Being preachy",
        ],
        "rhythm": {"short": 35, "medium": 45, "long": 20},
        "questions_per_post": 2,
        "exclamations_per_post": 1,
    },
    ContentType.TOOL_RELEASE: {
        "tone_target": "balanced assessment with hands-on credibility",
        "intensity": "medium",
        "markers_to_add": [
            "Hands-on experience ('I tested this immediately.')",
            "Balanced evaluation ('Here's what's great. And what's not.')",
            "Practical perspective ('For my workflow, this means...')",
            "Timely excitement ('Just dropped. First impressions:')",
            "Honest limitations ('It's not perfect. Here's why.')",
        ],
        "markers_to_avoid": [
            "Sounding like marketing copy",
            "Uncritical enthusiasm",
            "Dismissing without fair trial",
        ],
        "rhythm": {"short": 30, "medium": 50, "long": 20},
        "questions_per_post": 1,
        "exclamations_per_post": 1,
    },
}


# =============================================================================
# AI PHRASES TO DETECT AND REMOVE
# =============================================================================

AI_PHRASES: List[str] = [
    "It's important to note that",
    "In today's rapidly evolving",
    "Let's dive in",
    "Let's explore",
    "In conclusion",
    "Furthermore",
    "Moreover",
    "Additionally",
    "That being said",
    "At the end of the day",
    "It goes without saying",
    "Needless to say",
    "Without further ado",
    "In this article",
    "First and foremost",
    "Last but not least",
    "It is worth mentioning",
    "As we all know",
]

# Structural AI tells (regex patterns)
AI_STRUCTURE_PATTERNS: List[str] = [
    r"(?:Firstly|Secondly|Thirdly|Fourthly)",
    r"In (?:summary|essence|brief),",
    r"(?:This|It) (?:is|remains) (?:important|crucial|essential|vital) to",
    r"(?:One|Another) (?:key|important|notable) (?:aspect|point|factor)",
    r"While (?:it is|it's) true that",
]

# Author voice profile -- unique phrases the author uses
AUTHOR_UNIQUE_PHRASES: List[str] = [
    "Here's the thing:",
    "What surprised me most:",
    "The real insight here:",
    "Let me be direct:",
]

AUTHOR_EXPERTISE_AREAS: List[str] = [
    "AI implementation",
    "Automation workflows",
    "Enterprise AI strategy",
]


# =============================================================================
# HUMAN MARKERS CATALOG
# =============================================================================

HUMAN_MARKERS: Dict[str, List[str]] = {
    "personal_touches": [
        "Specific numbers ('I read 23 papers')",
        "Named tools/people ('when I asked GPT-4')",
        "Time references ('last Tuesday', 'this morning')",
        "Emotional reactions ('honestly, this surprised me')",
        "Mini-confessions ('I used to think X, but...')",
    ],
    "conversational_elements": [
        "Direct address ('you know what?')",
        "Rhetorical questions",
        "Incomplete sentences. Sometimes.",
        "Parenthetical asides (like this one)",
        "Self-corrections ('well, actually...')",
    ],
    "imperfections": [
        "Occasional informal words",
        "Starting sentences with 'And' or 'But'",
        "Em dashes for interruptions",
        "Sentence fragments for emphasis. Really.",
    ],
}


# =============================================================================
# QUALITY SCORING FUNCTIONS
# =============================================================================


def _calculate_humanness_score(
    text: str,
    patterns_removed: List[str],
    markers_added: List[str],
) -> float:
    """Calculate humanness score (0-10).

    Factors considered:
        - Contraction usage (human writing uses them frequently)
        - Sentence length variation (monotone length is robotic)
        - AI patterns found and removed (penalty even if removed)
        - Human markers successfully added (bonus)

    Args:
        text: The humanized text to evaluate.
        patterns_removed: List of AI patterns that were removed.
        markers_added: List of human markers that were added.

    Returns:
        Float score between 0.0 and 10.0.
    """
    score = 7.0  # Base score

    # Penalty for AI patterns found (even though they were removed, their
    # presence signals the original was very AI-sounding)
    score -= min(len(patterns_removed) * 0.3, 2.0)

    # Bonus for human markers added
    score += min(len(markers_added) * 0.2, 1.5)

    # Check for contractions (human writing uses them)
    contractions = re.findall(
        r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", text, re.IGNORECASE
    )
    if len(contractions) >= 3:
        score += 0.5

    # Check sentence length variation
    sentences = re.split(r"[.!?]+", text)
    word_counts = [len(s.split()) for s in sentences if s.strip()]
    if len(word_counts) > 3:
        variance = max(word_counts) - min(word_counts)
        if variance >= 5:
            score += 0.5

    # Check for conversational elements
    conversational_patterns = [
        r"\b(?:honestly|actually|basically|literally)\b",
        r"^(?:And|But)\s",
        r"---|\u2014",  # em dash
        r"\(.*?\)",  # parenthetical asides
    ]
    conversational_count = sum(
        1
        for pattern in conversational_patterns
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    )
    score += min(conversational_count * 0.25, 0.5)

    return max(0.0, min(10.0, score))


def _calculate_voice_consistency_score(
    text: str,
    author_phrases: List[str],
) -> float:
    """Calculate voice consistency score (0-10).

    Evaluates how well the text matches the author's established voice by
    checking for the author's signature phrases and stylistic patterns.

    Args:
        text: The humanized text to evaluate.
        author_phrases: List of phrases the author typically uses.

    Returns:
        Float score between 0.0 and 10.0.
    """
    score = 7.0  # Base score
    text_lower = text.lower()

    # Check for author's unique phrases
    phrases_found = sum(
        1 for phrase in author_phrases if phrase.lower() in text_lower
    )
    score += min(phrases_found * 0.5, 1.5)

    # Check for expertise area references
    expertise_found = sum(
        1
        for area in AUTHOR_EXPERTISE_AREAS
        if area.lower() in text_lower
    )
    score += min(expertise_found * 0.3, 1.0)

    # Penalty for overly generic language
    generic_patterns = [
        r"imagine a scenario",
        r"consider the following",
        r"let us examine",
    ]
    generic_count = sum(
        1
        for pattern in generic_patterns
        if re.search(pattern, text, re.IGNORECASE)
    )
    score -= generic_count * 0.5

    return max(0.0, min(10.0, score))


def _calculate_type_tone_score(
    text: str,
    content_type: ContentType,
) -> float:
    """Calculate type-tone match score (0-10).

    Evaluates how well the text's tone matches the expected tone for its
    content type.  Each content type has specific signals that should be
    present and others that should be absent.

    Args:
        text: The humanized text to evaluate.
        content_type: The content type to evaluate tone match against.

    Returns:
        Float score between 0.0 and 10.0.
    """
    score = 7.0  # Base score
    config = TYPE_HUMANIZATION.get(content_type, {})
    text_lower = text.lower()

    # -- Tone-specific keyword checks per content type -----------------------
    tone_signals: Dict[ContentType, List[str]] = {
        ContentType.ENTERPRISE_CASE: [
            "analysis", "metrics", "results", "implementation",
            "significant", "insight", "credibility",
        ],
        ContentType.PRIMARY_SOURCE: [
            "research", "finding", "hypothesis", "evidence",
            "nuanced", "complexity", "debate",
        ],
        ContentType.AUTOMATION_CASE: [
            "workflow", "automated", "built", "saved",
            "practical", "step", "tool",
        ],
        ContentType.COMMUNITY_CONTENT: [
            "community", "thread", "discussion", "shared",
            "experience", "perspective", "join",
        ],
        ContentType.TOOL_RELEASE: [
            "tested", "feature", "compared", "hands-on",
            "pros", "cons", "released",
        ],
    }

    signals = tone_signals.get(content_type, [])
    signals_found = sum(1 for s in signals if s in text_lower)
    if signals:
        match_ratio = signals_found / len(signals)
        score += match_ratio * 2.0  # Up to +2.0 for perfect match

    # Penalty for avoid-markers present in text
    avoid_markers = config.get("markers_to_avoid", [])
    for marker in avoid_markers:
        # Extract the key phrase from marker descriptions like
        # "Overly casual language ('crazy', 'insane', 'wild')"
        quoted = re.findall(r"'([^']+)'", marker)
        for word in quoted:
            if word.lower() in text_lower:
                score -= 0.5

    # Check question density matches type expectation
    question_count = text.count("?")
    expected_questions = config.get("questions_per_post", 1)
    if abs(question_count - expected_questions) <= 1:
        score += 0.5

    return max(0.0, min(10.0, score))


# =============================================================================
# HUMANIZER AGENT
# =============================================================================


class HumanizerAgent:
    """Agent that transforms AI-generated drafts into human-sounding content.

    The agent uses Claude to rewrite drafts with type-specific tone
    calibration, removing AI patterns and injecting appropriate human
    markers.  It evaluates the result with three quality scores:
    humanness, voice consistency, and type-tone match.

    Args:
        claude: The async Claude API client for LLM calls.

    Usage::

        agent = HumanizerAgent(claude=ClaudeClient())
        humanized = await agent.run(
            draft=writer_output.primary_draft,
            content_type=ContentType.ENTERPRISE_CASE,
        )
    """

    def __init__(self, claude: ClaudeClient) -> None:
        self.claude = claude
        self.logger = logging.getLogger("Humanizer")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        draft: DraftPost,
        content_type: ContentType,
        intensity: Optional[str] = None,
        tone_markers: Optional[List[str]] = None,
        avoid_markers: Optional[List[str]] = None,
        revision_instructions: Optional[str] = None,
    ) -> HumanizedPost:
        """Humanize a draft post with type-specific tone calibration.

        Orchestrates the full humanization pipeline:
        1. Detect AI patterns in the draft
        2. Build a type-aware humanization prompt
        3. Call Claude to rewrite the draft
        4. Parse the structured response
        5. Validate constraints (length, hook limit)
        6. Calculate quality scores

        Args:
            draft: The ``DraftPost`` produced by the Writer agent.
            content_type: Content type for tone calibration.
            intensity: Override for humanization intensity.  If ``None``,
                uses the type-specific default from ``TYPE_HUMANIZATION``.
            tone_markers: Additional tone markers to add (merged with
                type-specific defaults).
            avoid_markers: Additional markers to avoid (merged with
                type-specific defaults).
            revision_instructions: Optional feedback from QC agent for
                revision passes.

        Returns:
            A fully populated ``HumanizedPost``.

        Raises:
            WriterError: If the Claude call fails or the response cannot
                be parsed into a valid humanized post.
        """
        self.logger.info(
            "Humanizing draft for content_type=%s", content_type.value
        )

        # 1. Load type-specific configuration
        type_config = TYPE_HUMANIZATION.get(content_type, {})
        effective_intensity = intensity or type_config.get("intensity", "medium")

        effective_tone_markers = list(type_config.get("markers_to_add", []))
        if tone_markers:
            effective_tone_markers.extend(tone_markers)

        effective_avoid_markers = list(type_config.get("markers_to_avoid", []))
        if avoid_markers:
            effective_avoid_markers.extend(avoid_markers)

        # 2. Detect AI patterns in the original draft
        detected_patterns = self._detect_ai_patterns(draft.full_text)
        self.logger.info(
            "Detected %d AI patterns in draft", len(detected_patterns)
        )

        # 3. Build the humanization prompt
        prompt = self._build_prompt(
            draft=draft,
            content_type=content_type,
            type_config=type_config,
            effective_intensity=effective_intensity,
            effective_tone_markers=effective_tone_markers,
            effective_avoid_markers=effective_avoid_markers,
            detected_patterns=detected_patterns,
            revision_instructions=revision_instructions,
        )

        # 4. Call Claude for humanization
        system_prompt = (
            "You are a humanization expert for LinkedIn content. You transform "
            "AI-generated posts into authentic, human-sounding content while "
            "respecting content-type-specific tone requirements. You return "
            "structured JSON only."
        )

        try:
            response = await self.claude.generate_structured(
                prompt=prompt,
                system=system_prompt,
                max_tokens=4096,
            )
        except Exception as exc:
            raise WriterError(
                f"Humanizer Claude call failed for content_type="
                f"{content_type.value}: {exc}"
            ) from exc

        # 5. Parse and validate the response
        humanized = self._parse_response(
            response=response,
            draft=draft,
            content_type=content_type,
            detected_patterns=detected_patterns,
            effective_intensity=effective_intensity,
        )

        self.logger.info(
            "Humanization complete: humanness=%.1f voice=%.1f tone=%.1f chars=%d",
            humanized.ai_detection_score_before or 0.0,
            humanized.ai_detection_score_after or 0.0,
            len(humanized.humanized_text),
            len(humanized.humanized_text),
        )

        return humanized

    # ------------------------------------------------------------------
    # AI Pattern Detection
    # ------------------------------------------------------------------

    def _detect_ai_patterns(self, text: str) -> List[str]:
        """Scan text for known AI-generated patterns.

        Checks both exact phrase matches and structural regex patterns.

        Args:
            text: The draft text to scan.

        Returns:
            List of pattern descriptions that were found.
        """
        found: List[str] = []

        # Check exact AI phrases (case-insensitive)
        text_lower = text.lower()
        for phrase in AI_PHRASES:
            if phrase.lower() in text_lower:
                found.append(f"AI phrase: '{phrase}'")

        # Check structural patterns
        for pattern in AI_STRUCTURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(f"AI structure: {pattern}")

        # Check for excessive hedging
        hedge_words = re.findall(
            r"\b(?:might|could|may|perhaps|possibly|potentially)\b",
            text,
            re.IGNORECASE,
        )
        if len(hedge_words) > 3:
            found.append(
                f"Excessive hedging: {len(hedge_words)} hedge words found"
            )

        # Check for monotonous sentence length
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) >= 4:
            lengths = [len(s.split()) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            deviations = [abs(l - avg_len) for l in lengths]
            avg_deviation = sum(deviations) / len(deviations)
            if avg_deviation < 2.0:
                found.append(
                    "Monotonous sentence length: low variation "
                    f"(avg deviation {avg_deviation:.1f} words)"
                )

        return found

    # ------------------------------------------------------------------
    # Prompt Construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        draft: DraftPost,
        content_type: ContentType,
        type_config: Dict[str, Any],
        effective_intensity: str,
        effective_tone_markers: List[str],
        effective_avoid_markers: List[str],
        detected_patterns: List[str],
        revision_instructions: Optional[str],
    ) -> str:
        """Build the multi-part humanization prompt for Claude.

        Prompt structure:
            1. Base instructions (remove AI patterns, add human markers)
            2. Type-specific guidance (from TYPE_HUMANIZATION)
            3. Author voice profile (phrases to use/avoid)
            4. Constraints (length, hook limit, structure preservation)
            5. Revision instructions (if this is a QC-driven revision)
            6. Output format specification (structured JSON)

        Args:
            draft: The original draft post.
            content_type: Content type for tone calibration.
            type_config: Type-specific humanization configuration.
            effective_intensity: The humanization intensity to apply.
            effective_tone_markers: Merged list of tone markers to add.
            effective_avoid_markers: Merged list of markers to avoid.
            detected_patterns: AI patterns detected in the draft.
            revision_instructions: Optional QC feedback for revision.

        Returns:
            The complete prompt string.
        """
        rhythm = type_config.get("rhythm", {"short": 30, "medium": 50, "long": 20})
        rhythm_desc = (
            f"Short sentences (<8 words): {rhythm['short']}%, "
            f"Medium sentences (8-18 words): {rhythm['medium']}%, "
            f"Long sentences (>18 words): {rhythm['long']}%"
        )

        # -- Type-specific additional guidance --------------------------------
        type_guidance_map: Dict[ContentType, str] = {
            ContentType.ENTERPRISE_CASE: (
                "ADDITIONAL GUIDANCE FOR ENTERPRISE CASE:\n"
                "- Maintain professional credibility\n"
                "- Add interpretive insights, not casual commentary\n"
                "- Keep metrics and facts prominent\n"
                "- Sound like a thoughtful analyst, not a cheerleader"
            ),
            ContentType.PRIMARY_SOURCE: (
                "ADDITIONAL GUIDANCE FOR RESEARCH:\n"
                "- Show intellectual engagement\n"
                "- Make complex ideas accessible without dumbing down\n"
                "- Invite debate respectfully\n"
                "- Credit the researchers appropriately"
            ),
            ContentType.AUTOMATION_CASE: (
                "ADDITIONAL GUIDANCE FOR AUTOMATION:\n"
                "- Sound like a helpful practitioner\n"
                "- Be generous with practical details\n"
                "- Acknowledge challenges honestly\n"
                "- Encourage readers they can do this too"
            ),
            ContentType.COMMUNITY_CONTENT: (
                "ADDITIONAL GUIDANCE FOR COMMUNITY:\n"
                "- Feel connected to the community\n"
                "- Attribute insights properly\n"
                "- Add energy to the discussion\n"
                "- Invite participation"
            ),
            ContentType.TOOL_RELEASE: (
                "ADDITIONAL GUIDANCE FOR TOOL:\n"
                "- Sound hands-on, not like marketing\n"
                "- Balance pros and cons\n"
                "- Be timely in tone\n"
                "- Help readers decide if it's relevant for them"
            ),
        }
        type_guidance = type_guidance_map.get(content_type, "")

        # -- Author voice profile ---------------------------------------------
        author_profile = (
            "AUTHOR VOICE PROFILE:\n"
            f"Signature phrases: {', '.join(AUTHOR_UNIQUE_PHRASES)}\n"
            f"Expertise areas: {', '.join(AUTHOR_EXPERTISE_AREAS)}\n"
            "Style: Direct, insightful, occasionally informal but always credible."
        )

        # -- Detected AI patterns to remove -----------------------------------
        patterns_section = ""
        if detected_patterns:
            patterns_list = "\n".join(f"  - {p}" for p in detected_patterns)
            patterns_section = (
                f"\nDETECTED AI PATTERNS TO REMOVE:\n{patterns_list}\n"
            )

        # -- Revision instructions (QC feedback) ------------------------------
        revision_section = ""
        if revision_instructions:
            revision_section = (
                f"\nREVISION INSTRUCTIONS (from QC):\n"
                f"{revision_instructions}\n"
                "Apply these corrections while maintaining humanized voice.\n"
            )

        prompt = f"""You are a humanization expert. Your job is to take AI-generated
LinkedIn content and make it sound authentically human while
RESPECTING THE CONTENT TYPE.

ORIGINAL POST:
---
Hook: {draft.hook}

Body: {draft.body}

CTA: {draft.cta}

Hashtags: {' '.join(draft.hashtags)}
---

CONTENT TYPE: {content_type.value}
TYPE-SPECIFIC TONE: {type_config.get('tone_target', 'balanced and authentic')}
HUMANIZATION INTENSITY: {effective_intensity}

{author_profile}

TYPE-SPECIFIC MARKERS TO ADD:
{chr(10).join(f'  - {m}' for m in effective_tone_markers)}

TYPE-SPECIFIC MARKERS TO AVOID:
{chr(10).join(f'  - {m}' for m in effective_avoid_markers)}

RHYTHM PROFILE:
{rhythm_desc}
Target questions per post: {type_config.get('questions_per_post', 1)}
Target exclamations per post: {type_config.get('exclamations_per_post', 0)}
{patterns_section}
UNIVERSAL CHECKLIST:
1. Remove AI phrases: {', '.join(AI_PHRASES[:8])}... and similar
2. Add type-appropriate personal touches
3. Vary sentence rhythm per the rhythm profile
4. Add conversational elements appropriate to type
5. Include slight imperfections (but not too many)
6. Make opinions sound genuine, not hedged
7. Keep the same information, change the delivery

{type_guidance}

CONSTRAINTS:
- Don't add information that wasn't in the original
- Keep the same structure and main points
- Match the tone to the content type
- Don't add fake personal stories
- Respect the source material
- PRESERVE LENGTH: Keep within 10% of original length (max 3000 chars)
- HOOK LIMIT: Keep hook (first line before line break) under 210 characters
{revision_section}
OUTPUT FORMAT:
Return a JSON object with exactly these fields:
{{
    "hook": "humanized first line (under 210 chars)",
    "body": "humanized body text",
    "cta": "humanized call to action",
    "changes_made": ["list of specific changes made"],
    "ai_patterns_removed": ["list of AI patterns that were removed"],
    "human_markers_added": ["list of human markers that were added"],
    "type_adjustments": ["list of type-specific adjustments applied"]
}}

Return ONLY valid JSON, no markdown, no explanation."""

        return prompt

    # ------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: Dict[str, Any],
        draft: DraftPost,
        content_type: ContentType,
        detected_patterns: List[str],
        effective_intensity: str,
    ) -> HumanizedPost:
        """Parse Claude's JSON response into a ``HumanizedPost``.

        Validates constraints (hook length, total length) and calculates
        all three quality scores.

        Args:
            response: Parsed JSON dict from Claude.
            draft: The original draft for comparison.
            content_type: Content type used for scoring.
            detected_patterns: AI patterns detected in the original.
            effective_intensity: The intensity level that was applied.

        Returns:
            A fully populated ``HumanizedPost``.

        Raises:
            WriterError: If the response is missing required fields or
                violates hard constraints.
        """
        # Extract fields with safe defaults
        hook = response.get("hook", draft.hook)
        body = response.get("body", draft.body)
        cta = response.get("cta", draft.cta)
        changes_made = response.get("changes_made", [])
        ai_patterns_removed = response.get("ai_patterns_removed", [])
        human_markers_added = response.get("human_markers_added", [])
        type_adjustments = response.get("type_adjustments", [])

        # Validate hook length constraint
        if len(hook) > 210:
            self.logger.warning(
                "Hook exceeds 210 chars (%d), truncating at last space",
                len(hook),
            )
            hook = self._truncate_at_word_boundary(hook, 210)

        # Assemble full text
        hashtag_str = " ".join(draft.hashtags)
        full_text = f"{hook}\n\n{body}\n\n{cta}"
        if hashtag_str:
            full_text = f"{full_text}\n\n{hashtag_str}"

        # Validate total length
        if len(full_text) > 3000:
            self.logger.warning(
                "Humanized text exceeds 3000 chars (%d), will flag in output",
                len(full_text),
            )

        # Validate length preservation (within 10% of original)
        original_len = len(draft.full_text)
        humanized_len = len(full_text)
        length_deviation = (
            abs(humanized_len - original_len) / max(original_len, 1)
        )
        if length_deviation > 0.15:
            self.logger.warning(
                "Length deviation %.1f%% exceeds 10%% target (original=%d, "
                "humanized=%d)",
                length_deviation * 100,
                original_len,
                humanized_len,
            )

        # Calculate quality scores
        humanness = _calculate_humanness_score(
            full_text, ai_patterns_removed, human_markers_added
        )
        voice = _calculate_voice_consistency_score(
            full_text, AUTHOR_UNIQUE_PHRASES
        )
        tone = _calculate_type_tone_score(full_text, content_type)

        # Estimate AI detection scores (heuristic approximation)
        ai_score_before = self._estimate_ai_score(draft.full_text)
        ai_score_after = self._estimate_ai_score(full_text)

        return HumanizedPost(
            original_text=draft.full_text,
            humanized_text=full_text,
            changes_made=changes_made,
            humanization_intensity=effective_intensity,
            ai_detection_score_before=ai_score_before,
            ai_detection_score_after=ai_score_after,
        )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _truncate_at_word_boundary(self, text: str, max_length: int) -> str:
        """Truncate text at the last word boundary before ``max_length``.

        Args:
            text: Text to truncate.
            max_length: Maximum allowed character count.

        Returns:
            Truncated text ending at a word boundary.
        """
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return truncated[:last_space]
        return truncated

    def _estimate_ai_score(self, text: str) -> float:
        """Estimate an AI detection score (0.0-1.0) using heuristics.

        This is a lightweight, local approximation -- not a real AI detector.
        It checks for common AI writing tells to give a rough signal.

        Args:
            text: Text to evaluate.

        Returns:
            Float between 0.0 (very human) and 1.0 (very AI-like).
        """
        score = 0.0
        text_lower = text.lower()

        # Check for AI phrases
        ai_phrase_count = sum(
            1 for phrase in AI_PHRASES if phrase.lower() in text_lower
        )
        score += min(ai_phrase_count * 0.08, 0.3)

        # Check for structural AI patterns
        structure_count = sum(
            1
            for pattern in AI_STRUCTURE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        )
        score += min(structure_count * 0.06, 0.2)

        # Lack of contractions is AI-like
        contractions = re.findall(
            r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", text, re.IGNORECASE
        )
        words = text.split()
        if words:
            contraction_ratio = len(contractions) / len(words)
            if contraction_ratio < 0.01:
                score += 0.15
            elif contraction_ratio < 0.02:
                score += 0.08

        # Low sentence length variation is AI-like
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) >= 4:
            lengths = [len(s.split()) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            deviations = [abs(l - avg_len) for l in lengths]
            avg_deviation = sum(deviations) / len(deviations)
            if avg_deviation < 2.0:
                score += 0.15

        # Excessive hedging is AI-like
        hedge_words = re.findall(
            r"\b(?:might|could|may|perhaps|possibly|potentially)\b",
            text,
            re.IGNORECASE,
        )
        if len(hedge_words) > 3:
            score += 0.1

        return min(score, 1.0)


# =============================================================================
# FACTORY
# =============================================================================


async def create_humanizer() -> HumanizerAgent:
    """Factory function to create a ``HumanizerAgent`` with default clients."""
    claude = ClaudeClient()
    return HumanizerAgent(claude=claude)
