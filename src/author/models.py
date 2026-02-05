"""
Author profile data models for the LinkedIn Super Agent system.

Defines the ``AuthorVoiceProfile`` dataclass that captures an author's
complete voice and style profile based on their existing LinkedIn posts.
The Writer Agent consumes this profile to match the author's authentic
voice when generating new content.

Authoritative definition: ``architecture.md`` lines 23472-23520.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from src.utils import utc_now


@dataclass
class AuthorVoiceProfile:
    """Complete voice and style profile of the author.

    Created by ``AuthorProfileAgent.create_profile_from_posts()`` from an
    author's existing LinkedIn posts (minimum 10, recommended 20-50).
    Updated incrementally as new posts are published via
    ``AuthorProfileAgent.update_profile_incrementally()``.

    The ``generate_style_guide_for_writer()`` method on
    ``AuthorProfileAgent`` converts this profile into a concise text block
    injected into the Writer agent's system prompt.

    Attributes:
        author_name: Full name of the author (e.g. "John Doe").
        author_role: Professional title / positioning (e.g. "AI consultant").
        expertise_areas: List of domains the author is known for.
        characteristic_phrases: Signature phrases the author uses repeatedly
            (e.g. "Here's the thing:", "What struck me:").
        avoided_phrases: Phrases the author never uses and should be excluded
            from generated content.
        sentence_length_preference: Predominant sentence length style --
            one of ``"short"``, ``"medium"``, or ``"varied"``.
        paragraph_length: Typical paragraph density -- e.g.
            ``"1-2 sentences"`` or ``"3-4 sentences"``.
        formality_level: Scale from 0 (casual) to 1 (formal).
        humor_frequency: How often humor appears -- one of ``"never"``,
            ``"rarely"``, ``"sometimes"``, ``"often"``.
        emoji_usage: Emoji density -- one of ``"none"``, ``"minimal"``,
            ``"moderate"``, ``"heavy"``.
        favorite_topics: Topics the author writes about most frequently.
        topics_to_avoid: Topics the author deliberately avoids.
        typical_post_length: Average post length in characters.
        preferred_cta_styles: CTA patterns that work well for this author
            (e.g. "question to audience", "share your experience").
        known_opinions: Mapping of topic to the author's stated stance.
        contrarian_positions: Positions where the author disagrees with
            mainstream consensus.
        best_performing_hooks: Top hooks by engagement score.
        best_performing_structures: Post structures that drive the highest
            engagement.
        posting_frequency: How often the author publishes (e.g. "daily",
            "3x/week").
        best_posting_times: Times of day when posts perform best.
        created_at: When this profile was first created.
        last_updated: When this profile was last updated.
        posts_analyzed: Total number of posts used to build / update this
            profile.
    """

    # Identity
    author_name: str
    author_role: str  # "AI consultant", "Tech entrepreneur"
    expertise_areas: List[str]

    # Writing style
    characteristic_phrases: List[str]  # "Here's the thing:", "What struck me:"
    avoided_phrases: List[str]  # Phrases author never uses
    sentence_length_preference: str  # "short", "medium", "varied"
    paragraph_length: str  # "1-2 sentences", "3-4 sentences"

    # Tone
    formality_level: float  # 0=casual, 1=formal (e.g., 0.4)
    humor_frequency: str  # "never", "rarely", "sometimes", "often"
    emoji_usage: str  # "none", "minimal", "moderate", "heavy"

    # Content preferences
    favorite_topics: List[str]
    topics_to_avoid: List[str]
    typical_post_length: int  # characters
    preferred_cta_styles: List[str]

    # Opinions and positions
    known_opinions: Dict[str, str]  # topic -> stance
    contrarian_positions: List[str]  # Where author disagrees with mainstream

    # Patterns from data
    best_performing_hooks: List[str]  # Top hooks by engagement
    best_performing_structures: List[str]
    posting_frequency: str  # "daily", "3x/week", etc.
    best_posting_times: List[str]

    # Visual content preferences
    visual_content_ratio: float = 0.0
    preferred_visual_types: List[str] = field(default_factory=list)
    visual_type_performance: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=utc_now)
    last_updated: datetime = field(default_factory=utc_now)
    posts_analyzed: int = 0


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "AuthorVoiceProfile",
]
