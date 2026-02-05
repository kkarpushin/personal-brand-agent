"""
Visual Creator Agent for LinkedIn Super Agent.

Generates visual content optimized for each ``ContentType``, leveraging
Nano Banana Pro (Laozhang.ai / gemini-3-pro-image-preview) for AI image
generation and the Photo Selector agent for optional author photo
integration.

Pipeline position:
    Humanizer -> **Visual Creator** -> QC

The agent follows a deterministic flow:

1. Select the visual format based on ``ContentType`` and allowed formats.
2. Optionally integrate author photo (delegated to ``PhotoSelectorAgent``).
3. Build an image-generation prompt using type-specific templates and
   brand color guidelines.
4. Generate the primary image via Nano Banana Pro.
5. Optionally generate alternative images.
6. Return a ``VisualCreatorOutput`` with the primary asset and alternatives.

Fail-fast philosophy: NO FALLBACKS. If image generation fails, the agent
raises ``VisualizerError`` immediately. The orchestrator decides whether
to retry or escalate.

Architecture reference:
    - architecture.md lines 8300-8328  (Visual Creator architecture diagram)
    - architecture.md lines 9542-9639  (Content-Type to Visual Format Mapping)
    - architecture.md lines 9644-9800  (Type-Specific Visual Configuration)
    - architecture.md lines 10024-10083 (VisualAsset, VisualCreatorOutput)
    - architecture.md lines 9978-10016 (Negative prompts by type)
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional

from src.models import (
    ContentType,
    HumanizedPost,
    DraftPost,
    VisualAsset,
    VisualCreatorOutput,
)
from src.exceptions import VisualizerError, ImageGenerationError
from src.utils import utc_now, generate_id
from src.tools.nano_banana import NanoBananaClient
from src.tools.claude_client import ClaudeClient
from src.agents.photo_selector import PhotoSelectorAgent


# =============================================================================
# CONTENT TYPE -> VISUAL FORMAT MAPPING
#
# Each ContentType has primary visual formats (most effective), plus a
# style directive that guides prompt construction.
#
# Architecture reference: lines 9542-9639
# =============================================================================

VISUAL_FORMAT_MAP: Dict[ContentType, Dict[str, Any]] = {
    ContentType.ENTERPRISE_CASE: {
        "primary": ["metrics_card", "data_visualization", "architecture_diagram"],
        "secondary": ["company_logo_card", "timeline_visual"],
        "style": "Professional, data-driven, credible",
    },
    ContentType.PRIMARY_SOURCE: {
        "primary": ["concept_illustration", "quote_card", "paper_highlight"],
        "secondary": ["abstract_visualization", "comparison_visual"],
        "style": "Intellectual, thought-provoking, scientific",
    },
    ContentType.AUTOMATION_CASE: {
        "primary": ["workflow_diagram", "carousel", "tool_screenshot"],
        "secondary": ["before_after_card", "results_metrics_card"],
        "style": "Practical, instructional, clear hierarchy",
    },
    ContentType.COMMUNITY_CONTENT: {
        "primary": ["quote_card", "collage", "platform_screenshot"],
        "secondary": ["discussion_summary", "attribution_card"],
        "style": "Community-connected, multiple voices",
    },
    ContentType.TOOL_RELEASE: {
        "primary": ["product_screenshot", "feature_highlights", "comparison_chart"],
        "secondary": ["demo_animation", "pros_cons_card"],
        "style": "Fresh, timely, balanced",
    },
}


# =============================================================================
# BRAND COLORS
#
# Consistent brand palette applied to all generated visuals. The accent
# color varies per ContentType for visual differentiation in a feed.
# =============================================================================

BRAND_COLORS: Dict[str, str] = {
    "primary": "#1E3A8A",
    "secondary": "#3B82F6",
    "accent": "#10B981",
    "background": "#F8FAFC",
    "text": "#1F2937",
}

TYPE_ACCENT_COLORS: Dict[ContentType, str] = {
    ContentType.ENTERPRISE_CASE: "#1E3A8A",
    ContentType.PRIMARY_SOURCE: "#7C3AED",
    ContentType.AUTOMATION_CASE: "#10B981",
    ContentType.COMMUNITY_CONTENT: "#F59E0B",
    ContentType.TOOL_RELEASE: "#3B82F6",
}


# =============================================================================
# IMAGE PROMPT TEMPLATES BY CONTENT TYPE
#
# Each template is a parameterized string that gets filled with post-specific
# details and brand guidelines before being sent to Nano Banana Pro.
#
# Architecture reference: lines 9644-9800 (visual_config presets)
# =============================================================================

IMAGE_PROMPT_TEMPLATES: Dict[ContentType, Dict[str, str]] = {
    ContentType.ENTERPRISE_CASE: {
        "metrics_card": (
            "Clean professional metrics card for LinkedIn. "
            "Large prominent number showing key metric: {metric_highlight}. "
            "Company context: {company_context}. "
            "Brand colors: primary {primary_color}, accent {accent_color}. "
            "White background, professional typography, data-driven aesthetic. "
            "No watermarks, no stock photo feel."
        ),
        "data_visualization": (
            "Abstract data visualization representing {concept}. "
            "Flowing lines, nodes, neural network aesthetic. "
            "Dark background with glowing elements. "
            "Futuristic, professional. {accent_color} accents. "
            "4K resolution, sharp details. Clean and minimal."
        ),
        "architecture_diagram": (
            "Clean technical architecture diagram showing {system}. "
            "Boxes and arrows. Professional tech aesthetic. "
            "Blue and gray color scheme. White background. "
            "Clear visual hierarchy, no clutter. "
            "Suitable for LinkedIn professional audience."
        ),
    },
    ContentType.PRIMARY_SOURCE: {
        "concept_illustration": (
            "Professional tech illustration representing {concept}. "
            "Abstract, intellectual aesthetic. {mood} mood. "
            "Minimalist style. High quality, 4K resolution. "
            "Color accent: {accent_color}. "
            "No text, no watermark, no logo."
        ),
        "quote_card": (
            "Elegant quote card with text area prominent. "
            "Subtle background pattern or gradient. "
            "Academic, intellectual feel. "
            "Color scheme: {accent_color} on {background_color}. "
            "Professional typography placeholder. "
            "Clean, thought-provoking composition."
        ),
        "paper_highlight": (
            "Styled academic paper excerpt visual. "
            "Highlighted key finding area. Reference citation space. "
            "Clean typography. Scientific aesthetic. "
            "Background: {background_color}. Accent: {accent_color}. "
            "Professional, credible, scholarly."
        ),
    },
    ContentType.AUTOMATION_CASE: {
        "workflow_diagram": (
            "Step-by-step workflow visualization for {workflow_topic}. "
            "Numbered steps with arrow connections. "
            "Tool icons where applicable: {tools}. "
            "Clean, instructional aesthetic. "
            "Color-coded by step type. Green for triggers, blue for actions. "
            "Professional, clear visual hierarchy."
        ),
        "carousel": (
            "Professional carousel slide design for LinkedIn. "
            "Topic: {concept}. Slide purpose: {slide_purpose}. "
            "Clean layout, modern SaaS aesthetic. "
            "Brand color: {accent_color}. Background: {background_color}. "
            "Clear typography, numbered steps."
        ),
        "tool_screenshot": (
            "Modern laptop screen showing {tool_interface}. "
            "Clean UI with relevant interface elements visible. "
            "Professional product photography style. "
            "Subtle shadow, clean background. "
            "Device mockup, not a raw screenshot."
        ),
    },
    ContentType.COMMUNITY_CONTENT: {
        "quote_card": (
            "Community-style quote card for LinkedIn. "
            "Quote area prominent with platform aesthetic. "
            "User attribution space. Engagement metrics subtle. "
            "Discussion feel, warm colors: {accent_color}. "
            "Multiple voices represented. Not corporate, authentic."
        ),
        "collage": (
            "Multi-perspective collage composition for {concept}. "
            "2-4 viewpoints or quotes visually separated. "
            "Community discussion aesthetic. "
            "Warm color palette: {accent_color}. "
            "Platform-native feel, connected, not isolated."
        ),
        "platform_screenshot": (
            "Stylized social media discussion visual for {platform}. "
            "Thread aesthetic with multiple contributors. "
            "Professional LinkedIn-appropriate styling. "
            "Clean composition, highlighted key insights. "
            "Warm, community-connected palette."
        ),
    },
    ContentType.TOOL_RELEASE: {
        "product_screenshot": (
            "Clean device mockup showing {tool_name} interface. "
            "Key features visible: {features}. "
            "Professional product photography. "
            "Subtle shadow, clean gradient background. "
            "Modern SaaS aesthetic, not marketing hype."
        ),
        "feature_highlights": (
            "Tool feature highlights visual for {tool_name}. "
            "3-5 key features with icons. "
            "Clean, modern product aesthetic. "
            "Brand color: {accent_color}. Background: {background_color}. "
            "Balanced, evaluative tone. Not promotional."
        ),
        "comparison_chart": (
            "Tool comparison visual: {tool_name} vs alternatives. "
            "Feature checklist format. "
            "Green checks, neutral indicators. "
            "Professional, balanced evaluation. "
            "Clean typography, data-driven layout."
        ),
    },
}


# =============================================================================
# NEGATIVE PROMPTS BY CONTENT TYPE
#
# Elements to avoid in generated images for each content type.
# Architecture reference: lines 9978-10016
# =============================================================================

NEGATIVE_PROMPTS: Dict[ContentType, List[str]] = {
    ContentType.ENTERPRISE_CASE: [
        "text", "watermark", "logo", "playful", "cartoon",
        "abstract without data connection", "cluttered",
    ],
    ContentType.PRIMARY_SOURCE: [
        "text", "watermark", "oversimplified", "cartoon",
        "childish", "corporate stock",
    ],
    ContentType.AUTOMATION_CASE: [
        "text", "watermark", "abstract only", "no tools",
        "confusing flow", "too technical",
    ],
    ContentType.COMMUNITY_CONTENT: [
        "text", "watermark", "single person", "isolated",
        "cold", "corporate",
    ],
    ContentType.TOOL_RELEASE: [
        "text", "watermark", "outdated", "marketing hype",
        "generic tech", "stock photo",
    ],
}


# =============================================================================
# LINKEDIN IMAGE DIMENSIONS
# =============================================================================

LINKEDIN_SINGLE_IMAGE_SIZE = "1200x627"
LINKEDIN_CAROUSEL_SLIDE_SIZE = "1080x1350"


# =============================================================================
# VISUAL CREATOR AGENT
# =============================================================================


class VisualCreatorAgent:
    """Generates visual content for LinkedIn posts.

    Orchestrates image generation via Nano Banana Pro, optional author
    photo integration via the Photo Selector agent, and brand consistency
    enforcement.

    Args:
        nano_banana: Client for AI image generation.
        claude: Claude LLM client for prompt refinement.
        photo_selector: Optional Photo Selector agent for author photo
            integration. When ``None``, personal photos are never used.
    """

    def __init__(
        self,
        nano_banana: NanoBananaClient,
        claude: ClaudeClient,
        photo_selector: Optional[PhotoSelectorAgent] = None,
    ) -> None:
        self.nano_banana = nano_banana
        self.claude = claude
        self.photo_selector = photo_selector
        self.logger = logging.getLogger("VisualCreator")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def run(
        self,
        post: HumanizedPost,
        visual_brief: str,
        suggested_type: str,
        content_type: ContentType,
        allowed_formats: List[str],
        color_scheme: str,
        revision_instructions: Optional[List[str]] = None,
    ) -> VisualCreatorOutput:
        """Generate visual content for a post.

        This is the main entry point called by the LangGraph orchestrator.

        Args:
            post: The humanized post to create visuals for.
            visual_brief: Visual description from the Writer agent.
            suggested_type: Suggested visual type (e.g. ``"data_viz"``).
            content_type: The pipeline content type.
            allowed_formats: List of visual format names the pipeline
                allows (from type context).
            color_scheme: Color scheme identifier from type context.
            revision_instructions: Optional list of revision feedback
                from QC to incorporate into regeneration.

        Returns:
            :class:`VisualCreatorOutput` with primary and alternative
            visual assets.

        Raises:
            VisualizerError: If visual generation fails at any step.
        """
        self.logger.info(
            "Visual creation started: content_type=%s, suggested=%s, "
            "formats=%s",
            content_type.value,
            suggested_type,
            allowed_formats,
        )

        try:
            # Step 1: Select the best visual format
            selected_format = self._select_format(
                content_type, suggested_type, allowed_formats,
            )
            self.logger.info(
                "Format selected: %s (suggested=%s)",
                selected_format, suggested_type,
            )

            # Step 2: Check if author photo should be used
            photo_result = await self._check_photo_integration(
                post, content_type, visual_brief,
            )

            # Step 3: Build the generation prompt
            prompt = await self._build_prompt(
                post=post,
                visual_brief=visual_brief,
                selected_format=selected_format,
                content_type=content_type,
                color_scheme=color_scheme,
                photo_result=photo_result,
                revision_instructions=revision_instructions,
            )

            # Step 4: Determine image dimensions
            size = (
                LINKEDIN_CAROUSEL_SLIDE_SIZE
                if selected_format == "carousel"
                else LINKEDIN_SINGLE_IMAGE_SIZE
            )

            # Step 5: Generate primary image via Nano Banana Pro
            style = self._get_style_directive(content_type)
            primary_result = await self._generate_image(prompt, size, style)

            primary_asset = VisualAsset(
                asset_type="single_image" if selected_format != "carousel" else "carousel",
                url=primary_result.get("url"),
                prompt_used=primary_result.get("prompt_used", prompt),
                generation_model=primary_result.get("model", "nano-banana-pro"),
                width=int(size.split("x")[0]),
                height=int(size.split("x")[1]),
                metadata={
                    "format_selected": selected_format,
                    "content_type": content_type.value,
                    "color_scheme": color_scheme,
                    "photo_used": photo_result.get("use_photo", False) if photo_result else False,
                    "photo_id": photo_result.get("photo_id") if photo_result else None,
                    "photo_integration_mode": (
                        photo_result.get("integration_mode")
                        if photo_result
                        else None
                    ),
                    "negative_prompt": ", ".join(
                        NEGATIVE_PROMPTS.get(content_type, [])
                    ),
                },
            )

            # Step 6: Generate one alternative with a different format
            alternatives = await self._generate_alternatives(
                post=post,
                visual_brief=visual_brief,
                content_type=content_type,
                color_scheme=color_scheme,
                selected_format=selected_format,
                size=size,
                style=style,
            )

            output = VisualCreatorOutput(
                primary_asset=primary_asset,
                alternatives=alternatives,
                visual_brief_used=visual_brief,
                format_selected=selected_format,
            )

            self.logger.info(
                "Visual creation complete: format=%s, alternatives=%d",
                selected_format,
                len(alternatives),
            )
            return output

        except ImageGenerationError as exc:
            self.logger.error(
                "Image generation failed: %s", exc, exc_info=True,
            )
            raise VisualizerError(
                f"Visual creation failed for content_type="
                f"{content_type.value}: {exc}"
            ) from exc
        except Exception as exc:
            self.logger.error(
                "Unexpected error in visual creation: %s",
                exc,
                exc_info=True,
            )
            raise VisualizerError(
                f"Visual creation failed unexpectedly: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # FORMAT SELECTION
    # ------------------------------------------------------------------

    def _select_format(
        self,
        content_type: ContentType,
        suggested_type: str,
        allowed_formats: List[str],
    ) -> str:
        """Select the best visual format for this post.

        Priority order:
        1. ``suggested_type`` if it is in both ``allowed_formats`` and
           the content type's primary formats.
        2. First matching primary format that is in ``allowed_formats``.
        3. First matching secondary format that is in ``allowed_formats``.
        4. The ``suggested_type`` as-is if nothing else matches (trust
           the Writer agent's judgment).
        """
        format_config = VISUAL_FORMAT_MAP.get(content_type, {})
        primary_formats = format_config.get("primary", [])
        secondary_formats = format_config.get("secondary", [])

        # Prefer suggested_type if it is a valid primary format
        if suggested_type in primary_formats and (
            not allowed_formats or suggested_type in allowed_formats
        ):
            return suggested_type

        # Find first matching primary format
        for fmt in primary_formats:
            if not allowed_formats or fmt in allowed_formats:
                return fmt

        # Fall back to secondary formats
        for fmt in secondary_formats:
            if not allowed_formats or fmt in allowed_formats:
                return fmt

        # Last resort: trust the Writer's suggestion
        self.logger.warning(
            "No format matched allowed_formats=%s for content_type=%s; "
            "using suggested_type=%s",
            allowed_formats,
            content_type.value,
            suggested_type,
        )
        return suggested_type

    # ------------------------------------------------------------------
    # PHOTO INTEGRATION
    # ------------------------------------------------------------------

    async def _check_photo_integration(
        self,
        post: HumanizedPost,
        content_type: ContentType,
        visual_brief: str,
    ) -> Optional[Dict[str, Any]]:
        """Delegate to Photo Selector agent if available.

        Returns the photo selection result dict, or ``None`` if no photo
        selector is configured.
        """
        if self.photo_selector is None:
            self.logger.debug("No photo selector configured; skipping photo")
            return None

        result = await self.photo_selector.select_photo(
            post_content=post.humanized_text,
            content_type=content_type,
            visual_brief=visual_brief,
        )
        if result.get("use_photo"):
            self.logger.info(
                "Photo will be used: mode=%s, position=%s",
                result.get("integration_mode"),
                result.get("position"),
            )
        return result

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------

    async def _build_prompt(
        self,
        post: HumanizedPost,
        visual_brief: str,
        selected_format: str,
        content_type: ContentType,
        color_scheme: str,
        photo_result: Optional[Dict[str, Any]],
        revision_instructions: Optional[List[str]],
    ) -> str:
        """Construct the full image generation prompt.

        Combines:
        - The format-specific template filled with post context
        - Brand color directives
        - Photo integration instructions (if applicable)
        - Revision feedback (if this is a revision cycle)
        - Quality requirements for LinkedIn
        """
        accent_color = TYPE_ACCENT_COLORS.get(
            content_type, BRAND_COLORS["accent"]
        )

        # Try to use a template for the selected format
        type_templates = IMAGE_PROMPT_TEMPLATES.get(content_type, {})
        template = type_templates.get(selected_format)

        if template:
            # Fill template with available context
            prompt = template.format(
                concept=visual_brief[:100],
                metric_highlight=self._extract_key_metric(post.humanized_text),
                company_context=self._extract_company(post.humanized_text),
                system=visual_brief[:80],
                mood=VISUAL_FORMAT_MAP.get(content_type, {}).get("style", "professional"),
                accent_color=accent_color,
                primary_color=BRAND_COLORS["primary"],
                background_color=BRAND_COLORS["background"],
                workflow_topic=visual_brief[:60],
                tools=self._extract_tools(post.humanized_text),
                slide_purpose="key insight",
                platform="social media",
                tool_name=self._extract_tool_name(post.humanized_text),
                features=visual_brief[:80],
                tool_interface=visual_brief[:60],
            )
        else:
            # Fallback: use Claude to generate an appropriate prompt
            prompt = await self._generate_prompt_via_claude(
                visual_brief=visual_brief,
                selected_format=selected_format,
                content_type=content_type,
                post_text=post.humanized_text[:500],
            )

        # Append brand color instructions
        prompt += (
            f"\n\nBRAND GUIDELINES:\n"
            f"- Primary color: {BRAND_COLORS['primary']}\n"
            f"- Accent color: {accent_color}\n"
            f"- Background: {BRAND_COLORS['background']}\n"
            f"- Professional LinkedIn audience"
        )

        # Append photo integration instructions
        if photo_result and photo_result.get("use_photo"):
            mode = photo_result.get("integration_mode", "photo_overlay")
            position = photo_result.get("position", "left_side")
            prompt += (
                f"\n\nPHOTO INTEGRATION:\n"
                f"- Mode: {mode}\n"
                f"- Position: {position}\n"
                f"- Include space for author photo placement"
            )

        # Append negative prompt
        negatives = NEGATIVE_PROMPTS.get(content_type, [])
        if negatives:
            prompt += f"\n\nAVOID: {', '.join(negatives)}"

        # Append revision instructions if this is a revision cycle
        if revision_instructions:
            prompt += "\n\nREVISION FEEDBACK (incorporate these changes):\n"
            for i, instruction in enumerate(revision_instructions, 1):
                prompt += f"  {i}. {instruction}\n"

        # Append quality requirements
        prompt += (
            "\n\nQUALITY REQUIREMENTS:\n"
            "- 4K resolution, sharp details\n"
            "- Realistic lighting and shadows\n"
            "- Professional photography quality\n"
            "- Natural, not obviously AI-generated\n"
            "- Suitable for LinkedIn professional audience"
        )

        return prompt

    async def _generate_prompt_via_claude(
        self,
        visual_brief: str,
        selected_format: str,
        content_type: ContentType,
        post_text: str,
    ) -> str:
        """Use Claude to generate an image prompt when no template matches.

        This is used for formats that do not have a predefined template
        (e.g., secondary formats or custom QC-requested formats).
        """
        style = VISUAL_FORMAT_MAP.get(content_type, {}).get(
            "style", "professional"
        )
        system = (
            "You are a visual prompt engineer for LinkedIn content. "
            "Generate a concise, detailed image generation prompt for "
            "Nano Banana Pro (Gemini 3 Pro Image Preview). "
            "The prompt should describe a professional visual suitable "
            "for a LinkedIn post. Return ONLY the prompt text, no "
            "explanation."
        )
        user_prompt = (
            f"Create an image generation prompt for:\n"
            f"- Visual format: {selected_format}\n"
            f"- Content type: {content_type.value}\n"
            f"- Style: {style}\n"
            f"- Visual brief: {visual_brief}\n"
            f"- Post excerpt: {post_text[:300]}\n"
        )
        return await self.claude.generate(
            prompt=user_prompt,
            system=system,
            max_tokens=500,
            temperature=0.5,
        )

    # ------------------------------------------------------------------
    # IMAGE GENERATION
    # ------------------------------------------------------------------

    async def _generate_image(
        self,
        prompt: str,
        size: str,
        style: str,
    ) -> Dict[str, Any]:
        """Generate a single image via Nano Banana Pro.

        Raises:
            ImageGenerationError: Propagated from the Nano Banana client
                if generation fails after retries.
        """
        return await self.nano_banana.generate_image(
            prompt=prompt,
            size=size,
            style=style,
        )

    async def _generate_alternatives(
        self,
        post: HumanizedPost,
        visual_brief: str,
        content_type: ContentType,
        color_scheme: str,
        selected_format: str,
        size: str,
        style: str,
    ) -> List[VisualAsset]:
        """Generate one alternative visual with a different format.

        Picks the next available primary format that differs from the
        selected format. If generation fails, logs a warning and returns
        an empty list (alternatives are best-effort, but the primary
        asset failure is still fail-fast).
        """
        format_config = VISUAL_FORMAT_MAP.get(content_type, {})
        primary_formats = format_config.get("primary", [])

        alt_format = None
        for fmt in primary_formats:
            if fmt != selected_format:
                alt_format = fmt
                break

        if alt_format is None:
            return []

        try:
            alt_prompt = await self._build_prompt(
                post=post,
                visual_brief=visual_brief,
                selected_format=alt_format,
                content_type=content_type,
                color_scheme=color_scheme,
                photo_result=None,
                revision_instructions=None,
            )

            alt_result = await self._generate_image(alt_prompt, size, style)

            return [
                VisualAsset(
                    asset_type="single_image",
                    url=alt_result.get("url"),
                    prompt_used=alt_result.get("prompt_used", alt_prompt),
                    generation_model=alt_result.get("model", "nano-banana-pro"),
                    width=int(size.split("x")[0]),
                    height=int(size.split("x")[1]),
                    metadata={
                        "format_selected": alt_format,
                        "content_type": content_type.value,
                        "is_alternative": True,
                    },
                )
            ]
        except (ImageGenerationError, Exception) as exc:
            self.logger.warning(
                "Alternative visual generation failed (non-critical): %s",
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # STYLE / UTILITY HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_style_directive(content_type: ContentType) -> str:
        """Get the Nano Banana style parameter for a content type."""
        style_map: Dict[ContentType, str] = {
            ContentType.ENTERPRISE_CASE: "professional",
            ContentType.PRIMARY_SOURCE: "minimal",
            ContentType.AUTOMATION_CASE: "modern",
            ContentType.COMMUNITY_CONTENT: "warm",
            ContentType.TOOL_RELEASE: "clean",
        }
        return style_map.get(content_type, "professional")

    @staticmethod
    def _extract_key_metric(text: str) -> str:
        """Extract the first numeric metric from post text for prompts.

        Scans for patterns like ``42%``, ``3x``, ``$1.2M``, ``10,000``.
        Returns a placeholder if no metric is found.
        """
        import re

        patterns = [
            r'\$[\d,.]+[MBK]?',         # Dollar amounts
            r'\d+[.,]?\d*%',             # Percentages
            r'\d+[.,]?\d*x',             # Multipliers
            r'\d{1,3}(?:,\d{3})+',       # Large numbers with commas
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return "key performance metric"

    @staticmethod
    def _extract_company(text: str) -> str:
        """Extract company name context from post text.

        Simple heuristic: looks for common patterns like 'at Company'
        or 'Company's'. Returns a generic placeholder if not found.
        """
        import re

        patterns = [
            r'(?:at|by|from|for)\s+([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return "enterprise organization"

    @staticmethod
    def _extract_tools(text: str) -> str:
        """Extract tool/technology names from post text."""
        known_tools = [
            "n8n", "Zapier", "Make", "GPT", "Claude", "LangChain",
            "Python", "Docker", "Kubernetes", "Slack", "Notion",
            "Salesforce", "HubSpot", "Cursor", "VS Code",
        ]
        found = [tool for tool in known_tools if tool.lower() in text.lower()]
        return ", ".join(found) if found else "AI tools and workflows"

    @staticmethod
    def _extract_tool_name(text: str) -> str:
        """Extract the primary tool name from post text."""
        known_tools = [
            "ChatGPT", "Claude", "Gemini", "GPT-4", "Copilot",
            "n8n", "Zapier", "Make", "Cursor", "Replit",
            "Salesforce", "HubSpot", "Notion", "Linear",
        ]
        for tool in known_tools:
            if tool.lower() in text.lower():
                return tool
        return "AI Tool"
