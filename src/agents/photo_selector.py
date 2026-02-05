"""
Photo Selector Agent for LinkedIn Super Agent.

Decides whether to use the author's personal photo for a given post, selects
the most appropriate photo from the library, and determines the integration
mode (as-is, overlay, AI-edit, carousel bookend).

The selection process considers:
    1. Content-type-specific photo usage probability
    2. Contextual boosting signals (personal experience, opinion, hot take)
    3. Photo scoring (freshness, favorites, variety)
    4. Integration mode selection based on content type preferences

Fail-fast philosophy: no silent fallbacks. If the photo library is
unavailable or empty, the agent returns ``use_photo=False`` with a clear
rationale rather than hiding the failure.

Architecture reference:
    - architecture.md lines 8332-8431  (Photo Library System)
    - architecture.md lines 8435-8565  (Photo Integration Configuration)
    - architecture.md lines 8802-8888  (Photo Selection Algorithm)
    - architecture.md lines 8608-8625  (PhotoSelectionResult schema)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from src.models import ContentType
from src.utils import utc_now
from src.tools.photo_library import PhotoLibrary, PhotoMetadata


# =============================================================================
# PHOTO USAGE PROBABILITIES BY CONTENT TYPE
#
# These are the base probabilities that a post of a given content type
# will include the author's personal photo. Contextual signals (see
# _compute_boosted_probability) can increase these values.
#
# Architecture reference: lines 8371-8379
# =============================================================================

PHOTO_PROBS: Dict[ContentType, float] = {
    ContentType.ENTERPRISE_CASE: 0.6,
    ContentType.PRIMARY_SOURCE: 0.4,
    ContentType.AUTOMATION_CASE: 0.3,
    ContentType.COMMUNITY_CONTENT: 0.7,
    ContentType.TOOL_RELEASE: 0.2,
}


# =============================================================================
# CONTENT TYPE -> PHOTO PREFERENCES
#
# Maps each content type to preferred photo settings, poses, attire, and
# integration modes. Used to filter the photo library for the best match.
#
# Architecture reference: lines 8471-8514
# =============================================================================

CONTENT_TYPE_PHOTO_PREFS: Dict[ContentType, Dict[str, Any]] = {
    ContentType.ENTERPRISE_CASE: {
        "preferred_settings": ["office", "conference"],
        "preferred_poses": ["portrait", "speaking"],
        "preferred_attire": ["formal", "business_casual"],
        "integration_modes": ["photo_overlay", "photo_as_is"],
        "photo_position": "left_side",
    },
    ContentType.PRIMARY_SOURCE: {
        "preferred_settings": ["studio", "office"],
        "preferred_poses": ["thinking", "portrait"],
        "preferred_attire": ["business_casual"],
        "integration_modes": ["photo_overlay", "carousel_bookend"],
        "photo_position": "quote_card_corner",
    },
    ContentType.AUTOMATION_CASE: {
        "preferred_settings": ["office", "home"],
        "preferred_poses": ["working", "gesturing"],
        "preferred_attire": ["casual", "business_casual"],
        "integration_modes": ["carousel_bookend"],
        "photo_position": "first_last_slide",
    },
    ContentType.COMMUNITY_CONTENT: {
        "preferred_settings": ["any"],
        "preferred_poses": ["portrait", "thinking"],
        "preferred_mood": ["friendly", "thoughtful"],
        "integration_modes": ["photo_as_is", "photo_overlay"],
        "photo_position": "prominent",
    },
    ContentType.TOOL_RELEASE: {
        "preferred_settings": ["office"],
        "preferred_poses": ["working"],
        "preferred_attire": ["business_casual"],
        "integration_modes": ["photo_overlay"],
        "photo_position": "corner_badge",
    },
}


# =============================================================================
# CONTEXTUAL EDIT PROMPTS BY CONTENT TYPE
#
# When integration_mode is ``photo_ai_edit``, these base styles inform the
# AI edit prompt sent to Nano Banana.
#
# Architecture reference: lines 8761-8798
# =============================================================================

_EDIT_STYLE_MAP: Dict[ContentType, str] = {
    ContentType.ENTERPRISE_CASE: "professional, corporate lighting, subtle overlay",
    ContentType.PRIMARY_SOURCE: "clean, research-focused, data visualization overlay",
    ContentType.AUTOMATION_CASE: "tech-forward, workflow diagrams, connection lines",
    ContentType.COMMUNITY_CONTENT: "conversational, social media friendly, quote overlay",
    ContentType.TOOL_RELEASE: "product-focused, demo style, interface mockup overlay",
}


# =============================================================================
# PHOTO SELECTOR AGENT
# =============================================================================


class PhotoSelectorAgent:
    """Selects author photos for LinkedIn posts.

    The agent implements a multi-step selection algorithm:

    1. **Probability gate** -- Determine whether a photo should be used at
       all, based on content type probability and contextual boosting.
    2. **Candidate search** -- Query the photo library for photos matching
       the content type's preferred settings, poses, and attire.
    3. **Scoring** -- Score candidates by freshness (less used is better),
       favorite status, and variety (not recently used).
    4. **Integration mode** -- Select how the photo should be integrated
       (as-is, overlay, AI edit, carousel bookend).

    Args:
        photo_library: The :class:`PhotoLibrary` instance to search.
    """

    def __init__(self, photo_library: PhotoLibrary) -> None:
        self.photo_library = photo_library
        self.logger = logging.getLogger("PhotoSelector")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def select_photo(
        self,
        post_content: str,
        content_type: ContentType,
        visual_brief: str,
    ) -> Dict[str, Any]:
        """Select a photo for a post.

        Returns a dictionary with the following keys:

        - ``use_photo`` (bool): Whether to include a personal photo.
        - ``photo_id`` (Optional[str]): ID of the selected photo, or
          ``None`` if no photo will be used.
        - ``photo_path`` (Optional[str]): File path of the selected photo.
        - ``integration_mode`` (str): One of ``photo_as_is``,
          ``photo_overlay``, ``photo_ai_edit``, ``carousel_bookend``,
          or ``none``.
        - ``position`` (str): Where to place the photo in the visual.
        - ``edit_prompt`` (Optional[str]): AI edit prompt if
          ``integration_mode`` is ``photo_ai_edit``.
        - ``selection_rationale`` (str): Human-readable explanation.

        Args:
            post_content: The full text of the humanized post.
            content_type: The pipeline content type.
            visual_brief: The visual brief from the Writer agent.
        """
        self.logger.info(
            "Photo selection started for content_type=%s",
            content_type.value,
        )

        # Step 1: Compute boosted probability and decide whether to use photo
        base_prob = PHOTO_PROBS.get(content_type, 0.3)
        boosted_prob = self._compute_boosted_probability(
            base_prob, post_content, visual_brief,
        )

        roll = random.random()
        if roll > boosted_prob:
            self.logger.info(
                "Photo skipped: roll=%.2f > prob=%.2f (base=%.2f), "
                "content_type=%s",
                roll, boosted_prob, base_prob, content_type.value,
            )
            return self._no_photo_result(
                f"Skipped photo (prob={boosted_prob:.0%}, "
                f"content_type={content_type.value})"
            )

        # Step 2: Load preferences and search for candidates
        prefs = CONTENT_TYPE_PHOTO_PREFS.get(content_type, {})

        candidates = await self._search_candidates(content_type, prefs)
        if not candidates:
            self.logger.info(
                "No suitable photos found for content_type=%s",
                content_type.value,
            )
            return self._no_photo_result(
                "No suitable photos found in library"
            )

        # Step 3: Score and rank candidates
        scored = self._score_candidates(candidates)
        selected = scored[0][0]

        # Step 4: Choose integration mode
        integration_modes = prefs.get("integration_modes", ["photo_as_is"])
        integration_mode = random.choice(integration_modes)

        # Step 5: Generate AI edit prompt if needed
        edit_prompt: Optional[str] = None
        if integration_mode == "photo_ai_edit":
            edit_prompt = self._generate_edit_prompt(post_content, content_type)

        position = prefs.get("photo_position", "left_side")

        rationale = (
            f"Selected {selected.file_name} "
            f"(setting={selected.setting}, pose={selected.pose}) "
            f"via {integration_mode} at {position}"
        )
        self.logger.info("Photo selected: %s", rationale)

        return {
            "use_photo": True,
            "photo_id": selected.id,
            "photo_path": selected.file_path,
            "integration_mode": integration_mode,
            "position": position,
            "edit_prompt": edit_prompt,
            "selection_rationale": rationale,
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_boosted_probability(
        base_prob: float,
        post_content: str,
        visual_brief: str,
    ) -> float:
        """Apply contextual boosting signals to the base probability.

        Signals that increase photo usage likelihood:
        - Personal experience language (+0.20)
        - Hot take / opinion language (+0.15)

        The result is clamped to [0.0, 1.0].
        """
        prob = base_prob
        content_lower = post_content.lower()
        brief_lower = visual_brief.lower()

        # Personal experience signals (Russian and English)
        personal_signals = [
            "i learned", "my experience", "i recommend",
            "i use", "i built", "i tried",
        ]
        if any(signal in content_lower for signal in personal_signals):
            prob += 0.20

        # Hot take / opinion signals
        opinion_signals = ["hot take", "opinion", "unpopular", "controversial"]
        if any(signal in brief_lower or signal in content_lower for signal in opinion_signals):
            prob += 0.15

        return min(1.0, max(0.0, prob))

    async def _search_candidates(
        self,
        content_type: ContentType,
        prefs: Dict[str, Any],
    ) -> List[PhotoMetadata]:
        """Search the photo library for candidates matching preferences.

        First attempts a filtered search by content type suitability. If
        that yields no results, falls back to the least-used photos from
        the library (any setting/pose).
        """
        # Try content-type-specific search first
        candidates = await self.photo_library.search_photos(
            content_type=content_type.value,
            setting=None,
            mood=None,
            limit=10,
        )

        if candidates:
            return candidates

        # Broader fallback: least-used photos regardless of content type
        self.logger.debug(
            "No content-type-specific photos; falling back to least-used",
        )
        candidates = await self.photo_library.get_least_used(limit=5)
        return candidates

    def _score_candidates(
        self,
        candidates: List[PhotoMetadata],
    ) -> List[tuple]:
        """Score and sort photo candidates.

        Scoring factors:
        - **Freshness**: Less used photos score higher (max 1.0).
        - **Favorite bonus**: +0.3 for favorite photos.
        - **Variety bonus**: +0.2 for photos not recently used.

        Returns a list of ``(PhotoMetadata, score)`` tuples sorted by
        descending score.
        """
        recent_ids = self.photo_library.recent_used_ids
        scored: List[tuple] = []

        for photo in candidates:
            score = 0.0

            # Freshness bonus: less used = higher score
            score += max(0.0, 1.0 - (photo.times_used / 10.0))

            # Favorite bonus
            if photo.favorite:
                score += 0.3

            # Variety bonus: different from recently used
            if photo.id not in recent_ids:
                score += 0.2

            scored.append((photo, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _generate_edit_prompt(
        post_content: str,
        content_type: ContentType,
    ) -> str:
        """Generate a contextual AI edit prompt for photo_ai_edit mode.

        Architecture reference: lines 8761-8798.
        """
        base_style = _EDIT_STYLE_MAP.get(
            content_type, "professional LinkedIn style"
        )

        # Extract themes from post content
        themes: List[str] = []
        content_lower = post_content.lower()
        if "result" in content_lower:
            themes.append("success visualization")
        if "process" in content_lower:
            themes.append("workflow elements")
        if any(w in content_lower for w in ["grew", "boost", "increase", "growth"]):
            themes.append("growth indicators")

        theme_str = ", ".join(themes) if themes else "professional context"

        return (
            f"Edit this author photo with:\n"
            f"- Style: {base_style}\n"
            f"- Theme elements: {theme_str}\n"
            f"- Keep face clearly visible and professional\n"
            f"- Add subtle branded elements if appropriate\n"
            f"- Maintain authentic, not over-produced look"
        )

    @staticmethod
    def _no_photo_result(rationale: str) -> Dict[str, Any]:
        """Return a standardized 'no photo' result dict."""
        return {
            "use_photo": False,
            "photo_id": None,
            "photo_path": None,
            "integration_mode": "none",
            "position": "none",
            "edit_prompt": None,
            "selection_rationale": rationale,
        }
