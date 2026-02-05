"""
Photo library manager for author's personal photos.

Handles indexing, searching, and usage tracking of the author's photos
for LinkedIn post personalization.  The Photo Selector agent uses this
library to pick contextually appropriate photos and avoid repetition.

Storage model:
    - Photo files live in a local directory (``photos/`` by default).
    - Metadata is stored in Supabase via the ``db`` parameter.
    - An in-memory cache avoids repeated database reads.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import utc_now

logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


# =========================================================================
# Data model
# =========================================================================


@dataclass
class PhotoMetadata:
    """Metadata for a single photo in the library.

    Attributes:
        file_path: Absolute or relative path to the image file.
        file_name: Base file name.
        setting: Photo setting (``office``, ``conference``, ``outdoor``,
            ``studio``, ``home``).
        pose: Subject pose (``portrait``, ``speaking``, ``working``,
            ``thinking``, ``gesturing``).
        mood: Perceived mood (``professional``, ``friendly``, ``focused``,
            ``excited``, ``thoughtful``).
        attire: Subject attire (``formal``, ``business_casual``,
            ``casual``).
        suitable_for: Content types this photo is suitable for.
        times_used: Number of times the photo has been used in posts.
        last_used_date: When the photo was last used.
        last_used_post_id: ID of the last post that used this photo.
        favorite: Whether the photo is marked as a favorite.
        disabled: Whether the photo is disabled (excluded from selection).
        custom_tags: Free-form tags for additional categorization.
    """

    file_path: str
    file_name: str
    setting: Optional[str] = None
    pose: Optional[str] = None
    mood: Optional[str] = None
    attire: Optional[str] = None
    suitable_for: Optional[List[str]] = None
    times_used: int = 0
    last_used_date: Optional[datetime] = None
    last_used_post_id: Optional[str] = None
    favorite: bool = False
    disabled: bool = False
    custom_tags: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Deterministic ID derived from the file path."""
        return hashlib.sha256(self.file_path.encode()).hexdigest()[:16]


# =========================================================================
# Library
# =========================================================================


class PhotoLibrary:
    """Photo indexing and search for post personalization.

    Args:
        photos_dir: Directory containing the author's photos.
        db: Optional Supabase database client for persisting metadata.
            When ``None``, the library operates in memory-only mode
            (useful for testing).

    Usage::

        library = PhotoLibrary(photos_dir="photos", db=supabase_client)
        count = await library.index_photos()
        candidates = await library.search_photos(content_type="enterprise_case")
        await library.record_usage(candidates[0].file_path, post_id="abc-123")
    """

    def __init__(
        self,
        photos_dir: str = "photos",
        db: Any = None,
    ) -> None:
        self.photos_dir: Path = Path(photos_dir)
        self.db = db
        self._cache: Dict[str, PhotoMetadata] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_photos(self) -> int:
        """Scan the photos directory and index new photos.

        Discovers image files that are not yet in the cache, creates
        default :class:`PhotoMetadata` entries, and persists them to the
        database (if configured).

        Returns:
            Number of newly indexed photos.
        """
        if not self.photos_dir.exists():
            logger.warning("Photos directory does not exist: %s", self.photos_dir)
            return 0

        new_count = 0

        for file_path in self.photos_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue

            path_str = str(file_path)
            photo_id = hashlib.sha256(path_str.encode()).hexdigest()[:16]

            if photo_id in self._cache:
                continue

            metadata = PhotoMetadata(
                file_path=path_str,
                file_name=file_path.name,
                suitable_for=[],
            )
            self._cache[photo_id] = metadata
            new_count += 1

            # Persist to database if available
            if self.db is not None:
                try:
                    await self.db.upsert_photo_metadata(photo_id, {
                        "file_path": path_str,
                        "file_name": file_path.name,
                        "times_used": 0,
                    })
                except Exception:
                    logger.warning(
                        "Failed to persist photo metadata: %s",
                        path_str,
                        exc_info=True,
                    )

        self._loaded = True
        logger.info(
            "Photo indexing complete: %d new, %d total",
            new_count,
            len(self._cache),
        )
        return new_count

    # ------------------------------------------------------------------
    # Loading from database
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load all photo metadata from the database into the cache.

        No-op if already loaded or if no database is configured.
        """
        if self._loaded:
            return

        if self.db is not None:
            try:
                photos = await self.db.get_all_photos()
                for p in photos:
                    photo_id = p.get("id") or hashlib.sha256(
                        p.get("file_path", "").encode()
                    ).hexdigest()[:16]
                    self._cache[photo_id] = PhotoMetadata(
                        file_path=p.get("file_path", ""),
                        file_name=p.get("file_name", ""),
                        setting=p.get("setting"),
                        pose=p.get("pose"),
                        mood=p.get("mood"),
                        attire=p.get("attire"),
                        suitable_for=p.get("suitable_for", []),
                        times_used=p.get("times_used", 0),
                        last_used_post_id=p.get("last_used_post_id"),
                        favorite=p.get("favorite", False),
                        disabled=p.get("disabled", False),
                        custom_tags=p.get("custom_tags", []),
                    )
            except Exception:
                logger.warning(
                    "Failed to load photos from database; falling back to empty cache",
                    exc_info=True,
                )

        self._loaded = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_photos(
        self,
        content_type: Optional[str] = None,
        setting: Optional[str] = None,
        mood: Optional[str] = None,
        limit: int = 5,
    ) -> List[PhotoMetadata]:
        """Search photos by criteria.

        Args:
            content_type: Filter by suitability for this content type
                (e.g. ``"enterprise_case"``).
            setting: Filter by setting (e.g. ``"office"``).
            mood: Filter by mood (e.g. ``"professional"``).
            limit: Maximum number of results to return.

        Returns:
            List of matching :class:`PhotoMetadata`, sorted by relevance
            (favorites first, then least-used).
        """
        await self.load()

        candidates = [p for p in self._cache.values() if not p.disabled]

        if content_type:
            candidates = [
                p
                for p in candidates
                if p.suitable_for and content_type in p.suitable_for
            ]

        if setting:
            candidates = [
                p for p in candidates if p.setting == setting
            ]

        if mood:
            candidates = [
                p for p in candidates if p.mood == mood
            ]

        # Sort: favorites first, then least-used
        candidates.sort(key=lambda p: (-int(p.favorite), p.times_used))

        return candidates[:limit]

    # ------------------------------------------------------------------
    # Least-used photos
    # ------------------------------------------------------------------

    async def get_least_used(
        self,
        suitable_for: Optional[str] = None,
        limit: int = 3,
    ) -> List[PhotoMetadata]:
        """Get the least-used photos to avoid repetition.

        Args:
            suitable_for: Optional content type filter.
            limit: Number of photos to return.

        Returns:
            List of :class:`PhotoMetadata` sorted by ascending usage
            count.
        """
        await self.load()

        candidates = [p for p in self._cache.values() if not p.disabled]

        if suitable_for:
            candidates = [
                p
                for p in candidates
                if p.suitable_for and suitable_for in p.suitable_for
            ]

        candidates.sort(key=lambda p: p.times_used)
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    async def record_usage(self, photo_path: str, post_id: str) -> None:
        """Record that a photo was used in a post.

        Updates both the in-memory cache and the database.

        Args:
            photo_path: Path of the photo that was used.
            post_id: ID of the post that used the photo.
        """
        photo_id = hashlib.sha256(photo_path.encode()).hexdigest()[:16]

        if photo_id in self._cache:
            photo = self._cache[photo_id]
            photo.times_used += 1
            photo.last_used_date = utc_now()
            photo.last_used_post_id = post_id

        if self.db is not None:
            try:
                await self.db.update_photo_usage(photo_id, post_id)
            except Exception:
                logger.warning(
                    "Failed to persist photo usage: photo=%s, post=%s",
                    photo_path,
                    post_id,
                    exc_info=True,
                )

        logger.debug("Photo usage recorded: %s -> post %s", photo_path, post_id)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, photo_id: str) -> Optional[PhotoMetadata]:
        """Get a photo by its ID from the in-memory cache.

        Args:
            photo_id: Photo identifier (SHA-256 prefix of file path).

        Returns:
            :class:`PhotoMetadata` if found, or ``None``.
        """
        return self._cache.get(photo_id)

    def get_by_path(self, file_path: str) -> Optional[PhotoMetadata]:
        """Get a photo by its file path from the in-memory cache.

        Args:
            file_path: Photo file path.

        Returns:
            :class:`PhotoMetadata` if found, or ``None``.
        """
        photo_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        return self._cache.get(photo_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of photos currently in the cache."""
        return len(self._cache)

    @property
    def recent_used_ids(self) -> set:
        """Set of photo IDs used recently (sync property for quick access).

        Note: For authoritative recent-usage data, query the database
        directly.  This property returns IDs from the in-memory cache
        where ``last_used_post_id`` is set.
        """
        return {
            pid
            for pid, photo in self._cache.items()
            if photo.last_used_post_id is not None
        }
