"""
Author Profile Agent for the LinkedIn Super Agent system.

Creates and maintains an author's voice profile based on their existing
LinkedIn posts.  The Writer Agent uses the generated style guide to match
the author's authentic voice when generating new content.

Workflow:
    1. Collect 20-50 existing posts (via ``ProfileImporter``).
    2. Call ``create_profile_from_posts()`` to build the initial profile.
    3. Periodically call ``update_profile_incrementally()`` with new posts.
    4. Before each Writer run, call ``generate_style_guide_for_writer()``
       to produce a compact style-guide string for the Writer's system prompt.

Error philosophy: NO FALLBACKS.  If Claude fails to analyze posts or the
profile cannot be saved, exceptions propagate immediately.

Authoritative definition: ``architecture.md`` lines 23522-23757.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.tools.claude_client import ClaudeClient
from src.author.models import AuthorVoiceProfile
from src.utils import utc_now

logger = logging.getLogger("AuthorProfileAgent")


class AuthorProfileAgent:
    """Creates and maintains author voice profiles.

    Used by the Writer Agent to match the author's authentic voice.
    Profiles are persisted to the ``author_profiles`` Supabase table.

    Args:
        claude_client: An async Claude API client instance.  When ``None``,
            a new ``ClaudeClient`` is created with default settings.
        db: An async ``SupabaseDB`` instance for persistence.  When
            ``None``, profile save / load operations will raise.
    """

    # ------------------------------------------------------------------
    # PROMPTS
    # ------------------------------------------------------------------

    ANALYSIS_PROMPT: str = """
    Analyze these LinkedIn posts from {author_name} to extract their unique voice and style.

    Posts (sorted by engagement, highest first):
    {posts_text}

    Extract the following:

    1. CHARACTERISTIC PHRASES
       - Phrases they use repeatedly
       - Their unique ways of starting/ending thoughts
       - Signature expressions
       - Quote specific examples from their posts

    2. WRITING STYLE
       - Average sentence length (short/medium/long/varied)
       - Paragraph structure (how many sentences per paragraph)
       - Use of lists, bullets, formatting
       - Emoji patterns (with examples)

    3. TONE & VOICE
       - Formality level (0-1 scale, with explanation)
       - Humor usage (never/rarely/sometimes/often)
       - How they express opinions (confident/hedging/questioning)
       - How they address the reader

    4. CONTENT PATTERNS
       - Favorite topics (list with frequency)
       - How they structure arguments
       - Typical hooks they use (quote examples)
       - Common CTAs (quote examples)

    5. OPINIONS & POSITIONS
       - Strong opinions they've expressed (with quotes)
       - Contrarian views they hold
       - Topics they're passionate about

    6. WHAT TO AVOID
       - Phrases they never use
       - Topics they avoid
       - Styles that don't fit them

    Be SPECIFIC. Quote actual text from their posts as evidence.

    Return as structured JSON with these exact keys:
    {{
        "author_name": "{author_name}",
        "author_role": "<inferred role>",
        "expertise_areas": ["..."],
        "characteristic_phrases": ["..."],
        "avoided_phrases": ["..."],
        "sentence_length_preference": "short|medium|varied",
        "paragraph_length": "1-2 sentences|3-4 sentences",
        "formality_level": 0.0-1.0,
        "humor_frequency": "never|rarely|sometimes|often",
        "emoji_usage": "none|minimal|moderate|heavy",
        "favorite_topics": ["..."],
        "topics_to_avoid": ["..."],
        "typical_post_length": <int chars>,
        "preferred_cta_styles": ["..."],
        "known_opinions": {{"topic": "stance", ...}},
        "contrarian_positions": ["..."],
        "best_performing_hooks": ["..."],
        "best_performing_structures": ["..."],
        "posting_frequency": "<frequency>",
        "best_posting_times": ["..."]
    }}
    """

    UPDATE_PROMPT: str = """
    Current author profile for {author_name}:

    Characteristic phrases: {characteristic_phrases}
    Best hooks: {best_performing_hooks}
    Typical length: {typical_post_length} chars
    Formality: {formality_level}

    New posts to incorporate (analyze for new patterns):
    {posts_text}

    Update the profile:
    1. Add any NEW characteristic phrases discovered (don't remove existing)
    2. Update best_performing_hooks if any new posts outperformed existing
    3. Note any evolution in style or topics
    4. Update statistics (typical_post_length, etc.)
    5. Keep everything else stable unless clear change detected

    Return the complete updated profile as structured JSON with these exact keys:
    {{
        "author_name": "{author_name}",
        "author_role": "<role>",
        "expertise_areas": ["..."],
        "characteristic_phrases": ["..."],
        "avoided_phrases": ["..."],
        "sentence_length_preference": "short|medium|varied",
        "paragraph_length": "1-2 sentences|3-4 sentences",
        "formality_level": 0.0-1.0,
        "humor_frequency": "never|rarely|sometimes|often",
        "emoji_usage": "none|minimal|moderate|heavy",
        "favorite_topics": ["..."],
        "topics_to_avoid": ["..."],
        "typical_post_length": <int chars>,
        "preferred_cta_styles": ["..."],
        "known_opinions": {{"topic": "stance", ...}},
        "contrarian_positions": ["..."],
        "best_performing_hooks": ["..."],
        "best_performing_structures": ["..."],
        "posting_frequency": "<frequency>",
        "best_posting_times": ["..."]
    }}
    """

    STYLE_GUIDE_PROMPT: str = """
    Based on this author voice profile, generate a concise style guide
    that a Writer AI can use to match the author's voice when creating
    LinkedIn posts.

    Author Profile:
    {profile_json}

    The style guide should be:
    - Actionable (clear DOs and DON'Ts)
    - Concise (fit in a system prompt section)
    - Specific (use actual phrases and examples from the profile)
    - Prioritized (most important voice markers first)

    Return ONLY the style guide text, no JSON wrapper.
    """

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        db: Optional[Any] = None,
    ) -> None:
        self.claude: ClaudeClient = claude_client or ClaudeClient()
        self.db = db

    # ------------------------------------------------------------------
    # PROFILE CREATION
    # ------------------------------------------------------------------

    async def create_profile_from_posts(
        self,
        author_name: str,
        posts: List[Dict[str, Any]],
    ) -> AuthorVoiceProfile:
        """Analyze existing posts to create an author voice profile.

        Should be run once with 10-50 posts for a reliable profile.

        Args:
            author_name: Name of the author.
            posts: List of post dicts, each containing at minimum
                ``text`` (str).  Optional keys: ``likes`` (int),
                ``comments`` (int), ``date`` (str).

        Returns:
            A fully populated ``AuthorVoiceProfile``.

        Raises:
            ValueError: If fewer than 10 posts are provided.
        """
        if len(posts) < 10:
            raise ValueError(
                f"Need at least 10 posts for a reliable profile. Got {len(posts)}."
            )

        # Sort by engagement (comments weighted 3x more than likes)
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get("likes", 0) + p.get("comments", 0) * 3,
            reverse=True,
        )

        # Format top 30 posts for analysis
        posts_text = self._format_posts(sorted_posts[:30])

        logger.info(
            "Creating profile for '%s' from %d posts (top 30 used for analysis)",
            author_name,
            len(posts),
        )

        # Call Claude to analyze posts
        raw_profile: Dict[str, Any] = await self.claude.generate_structured(
            prompt=self.ANALYSIS_PROMPT.format(
                author_name=author_name,
                posts_text=posts_text,
            ),
        )

        # Build AuthorVoiceProfile from Claude's response
        profile = self._dict_to_profile(raw_profile)

        # Override identity fields with authoritative values
        profile.author_name = author_name
        profile.posts_analyzed = len(posts)
        profile.created_at = utc_now()
        profile.last_updated = utc_now()

        # Extract best-performing hooks from top 5 posts
        extracted_hooks = self._extract_hooks(sorted_posts[:5])
        if extracted_hooks:
            profile.best_performing_hooks = extracted_hooks

        logger.info(
            "Profile created for '%s': %d phrases, %d hooks, formality=%.2f",
            author_name,
            len(profile.characteristic_phrases),
            len(profile.best_performing_hooks),
            profile.formality_level,
        )

        return profile

    # ------------------------------------------------------------------
    # INCREMENTAL UPDATE
    # ------------------------------------------------------------------

    async def update_profile_incrementally(
        self,
        profile: AuthorVoiceProfile,
        new_posts: List[Dict[str, Any]],
    ) -> AuthorVoiceProfile:
        """Update an existing profile with new posts.

        Does not replace the profile -- adds new patterns and updates
        statistics while keeping stable elements intact.

        Args:
            profile: The current author voice profile to update.
            new_posts: List of new post dicts to incorporate.

        Returns:
            The updated ``AuthorVoiceProfile``.
        """
        if not new_posts:
            logger.warning("No new posts provided for incremental update")
            return profile

        posts_text = self._format_posts(new_posts)

        logger.info(
            "Updating profile for '%s' with %d new posts",
            profile.author_name,
            len(new_posts),
        )

        raw_update: Dict[str, Any] = await self.claude.generate_structured(
            prompt=self.UPDATE_PROMPT.format(
                author_name=profile.author_name,
                characteristic_phrases=profile.characteristic_phrases,
                best_performing_hooks=profile.best_performing_hooks,
                typical_post_length=profile.typical_post_length,
                formality_level=profile.formality_level,
                posts_text=posts_text,
            ),
        )

        updated_profile = self._dict_to_profile(raw_update)

        # Preserve identity and metadata
        updated_profile.author_name = profile.author_name
        updated_profile.author_role = profile.author_role
        updated_profile.created_at = profile.created_at
        updated_profile.last_updated = utc_now()
        updated_profile.posts_analyzed = profile.posts_analyzed + len(new_posts)

        logger.info(
            "Profile updated for '%s': now %d posts analyzed",
            profile.author_name,
            updated_profile.posts_analyzed,
        )

        return updated_profile

    # ------------------------------------------------------------------
    # STYLE GUIDE GENERATION
    # ------------------------------------------------------------------

    async def generate_style_guide_for_writer(
        self,
        profile: AuthorVoiceProfile,
    ) -> str:
        """Generate a concise style guide for the Writer agent.

        Produces a compact text block suitable for injection into the
        Writer's system prompt.  Uses Claude to synthesise the profile
        into actionable writing instructions.

        Args:
            profile: The author's voice profile.

        Returns:
            Style guide string for the Writer agent's system prompt.
        """
        profile_dict = asdict(profile)

        # Remove datetime fields that are not JSON-serialisable by default
        profile_dict["created_at"] = profile.created_at.isoformat()
        profile_dict["last_updated"] = profile.last_updated.isoformat()

        profile_json = json.dumps(profile_dict, indent=2, ensure_ascii=False)

        logger.debug(
            "Generating style guide for writer from profile '%s'",
            profile.author_name,
        )

        style_guide: str = await self.claude.generate(
            prompt=self.STYLE_GUIDE_PROMPT.format(profile_json=profile_json),
            temperature=0.3,
        )

        return style_guide

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    async def save_profile(self, profile: AuthorVoiceProfile) -> None:
        """Save an author voice profile to the database.

        Uses upsert semantics: if a profile for the same ``author_name``
        already exists, it is updated.

        Args:
            profile: The profile to persist.

        Raises:
            RuntimeError: If no database client was provided at init.
        """
        if self.db is None:
            raise RuntimeError(
                "Cannot save profile: no database client provided. "
                "Pass a SupabaseDB instance to AuthorProfileAgent(db=...)."
            )

        # Map AuthorVoiceProfile fields to the actual DB column names.
        profile_dict = {
            "name": profile.author_name,
            "title": profile.author_role,
            "expertise_areas": profile.expertise_areas,
            "signature_phrases": profile.characteristic_phrases,
            "avoided_topics": profile.topics_to_avoid,
            "tone": f"formality_{profile.formality_level:.1f}",
            "vocabulary_level": profile.sentence_length_preference,
            "emoji_usage": profile.emoji_usage,
            "preferred_post_length": str(profile.typical_post_length),
            "preferred_content_types": profile.preferred_cta_styles,
            "topics_of_interest": profile.favorite_topics,
            "posting_frequency": profile.posting_frequency,
            "best_posting_times": profile.best_posting_times,
            "updated_at": profile.last_updated.isoformat(),
        }

        await self.db.save_author_profile(profile_dict)

        logger.info(
            "Profile saved for '%s' (%d posts analyzed)",
            profile.author_name,
            profile.posts_analyzed,
        )

    async def load_profile(
        self,
        author_name: str,
    ) -> Optional[AuthorVoiceProfile]:
        """Load an author voice profile from the database.

        Args:
            author_name: The name of the author to look up.

        Returns:
            ``AuthorVoiceProfile`` if found, ``None`` otherwise.

        Raises:
            RuntimeError: If no database client was provided at init.
        """
        if self.db is None:
            raise RuntimeError(
                "Cannot load profile: no database client provided. "
                "Pass a SupabaseDB instance to AuthorProfileAgent(db=...)."
            )

        data: Optional[Dict[str, Any]] = await self.db.get_author_profile()

        if data is None:
            logger.warning("No profile found for '%s'", author_name)
            return None

        # Verify we got the right author (single-author system, but be safe).
        # DB column is "name", not "author_name".
        db_name = data.get("name", data.get("author_name"))
        if db_name != author_name:
            logger.warning(
                "Profile name mismatch: expected '%s', got '%s'",
                author_name,
                db_name,
            )
            return None

        # Map DB columns back to AuthorVoiceProfile field names
        data["author_name"] = data.pop("name", author_name)
        if "title" in data:
            data.setdefault("author_role", data.pop("title", ""))
        if "signature_phrases" in data:
            data.setdefault("characteristic_phrases", data.pop("signature_phrases", []))
        if "topics_of_interest" in data:
            data.setdefault("favorite_topics", data.pop("topics_of_interest", []))
        if "avoided_topics" in data:
            data.setdefault("topics_to_avoid", data.pop("avoided_topics", []))
        if "preferred_post_length" in data:
            try:
                data.setdefault("typical_post_length", int(data.pop("preferred_post_length", 1200)))
            except (ValueError, TypeError):
                data.setdefault("typical_post_length", 1200)
        if "updated_at" in data:
            data.setdefault("last_updated", data.pop("updated_at"))

        return self._dict_to_profile(data)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _format_posts(self, posts: List[Dict[str, Any]]) -> str:
        """Format posts for inclusion in an analysis prompt.

        Args:
            posts: List of post dicts.

        Returns:
            Formatted string with engagement scores and text.
        """
        formatted: List[str] = []
        for i, p in enumerate(posts):
            engagement = p.get("likes", 0) + p.get("comments", 0) * 3
            text = p.get("text", p.get("content", ""))[:1500]  # Truncate long posts
            formatted.append(
                f"--- Post {i + 1} (engagement score: {engagement}) ---\n"
                f"{text}\n"
            )
        return "\n".join(formatted)

    def _extract_hooks(self, posts: List[Dict[str, Any]]) -> List[str]:
        """Extract opening hooks (first 2 non-empty lines) from top posts.

        Args:
            posts: List of post dicts sorted by engagement.

        Returns:
            List of hook strings (max 200 chars each).
        """
        hooks: List[str] = []
        for p in posts:
            text = p.get("text", p.get("content", ""))
            lines = text.split("\n")
            non_empty = [line.strip() for line in lines if line.strip()]
            hook = " ".join(non_empty[:2]).strip()
            if hook:
                hooks.append(hook[:200])  # Max 200 chars
        return hooks

    def _dict_to_profile(self, data: Dict[str, Any]) -> AuthorVoiceProfile:
        """Convert a raw dict (from Claude or DB) into an AuthorVoiceProfile.

        Applies safe defaults for any missing keys to avoid ``TypeError``
        from the dataclass constructor.

        Args:
            data: Raw dict with profile data.

        Returns:
            A populated ``AuthorVoiceProfile`` instance.
        """
        return AuthorVoiceProfile(
            author_name=data.get("author_name", ""),
            author_role=data.get("author_role", ""),
            expertise_areas=data.get("expertise_areas", []),
            characteristic_phrases=data.get("characteristic_phrases", []),
            avoided_phrases=data.get("avoided_phrases", []),
            sentence_length_preference=data.get("sentence_length_preference", "varied"),
            paragraph_length=data.get("paragraph_length", "1-2 sentences"),
            formality_level=float(data.get("formality_level", 0.5)),
            humor_frequency=data.get("humor_frequency", "rarely"),
            emoji_usage=data.get("emoji_usage", "minimal"),
            favorite_topics=data.get("favorite_topics", []),
            topics_to_avoid=data.get("topics_to_avoid", []),
            typical_post_length=int(data.get("typical_post_length", 1200)),
            preferred_cta_styles=data.get("preferred_cta_styles", []),
            known_opinions=data.get("known_opinions", {}),
            contrarian_positions=data.get("contrarian_positions", []),
            best_performing_hooks=data.get("best_performing_hooks", []),
            best_performing_structures=data.get("best_performing_structures", []),
            posting_frequency=data.get("posting_frequency", ""),
            best_posting_times=data.get("best_posting_times", []),
            posts_analyzed=int(data.get("posts_analyzed", 0)),
        )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "AuthorProfileAgent",
]
