"""
Profile Importer for the LinkedIn Super Agent system.

Imports posts from LinkedIn or local JSON files for use with the
``AuthorProfileAgent`` to build or update an author's voice profile.

Supported sources:
    - **LinkedIn** -- retrieves recent posts via the ``LinkedInClient``
      async wrapper around the Voyager API.
    - **JSON file** -- reads a local JSON file containing an array of
      post objects.

All imported posts are normalised to a consistent dict structure::

    {
        "text": str,          # Post body text (required)
        "likes": int,         # Like count (default 0)
        "comments": int,      # Comment count (default 0)
        "date": str,          # Publication date ISO string (default "")
        "content_type": str,  # e.g. "text", "image", "video" (default "text")
    }

Error philosophy: invalid / missing data is filtered out with warnings,
but structural problems in the JSON file raise immediately.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ProfileImporter")


class ProfileImporter:
    """Imports posts from LinkedIn or JSON files for profile creation.

    Args:
        linkedin_client: An optional async ``LinkedInClient`` instance.
            Required only for ``import_from_linkedin()``.
        db: An optional async ``SupabaseDB`` instance.  When provided,
            imported posts are persisted to the ``posts`` table and
            baseline metrics snapshots are stored.
    """

    def __init__(
        self,
        linkedin_client: Optional[Any] = None,
        db: Optional[Any] = None,
    ) -> None:
        self.linkedin_client = linkedin_client
        self.db = db

    # ------------------------------------------------------------------
    # LINKEDIN IMPORT
    # ------------------------------------------------------------------

    async def import_from_linkedin(
        self,
        profile_url: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Import recent posts from a LinkedIn profile.

        Uses ``linkedin_client.get_post_comments`` / Voyager API to
        retrieve the author's recent posts and normalises them into the
        standard post dict format.

        Args:
            profile_url: LinkedIn profile URL (e.g.
                ``"https://www.linkedin.com/in/johndoe"``).
            limit: Maximum number of posts to retrieve (default 50).

        Returns:
            List of normalised post dicts.

        Raises:
            RuntimeError: If no ``linkedin_client`` was provided.
            LinkedInAPIError: On API failure (propagated from client).
        """
        if self.linkedin_client is None:
            raise RuntimeError(
                "Cannot import from LinkedIn: no linkedin_client provided. "
                "Pass a LinkedInClient instance to ProfileImporter(linkedin_client=...)."
            )

        logger.info(
            "Importing up to %d posts from LinkedIn profile: %s",
            limit,
            profile_url,
        )

        # The linkedin-api library uses a public_id derived from the URL
        # e.g. "https://www.linkedin.com/in/johndoe" -> "johndoe"
        public_id = self._extract_public_id(profile_url)

        # Use the underlying linkedin-api to get the profile's posts.
        # The LinkedInClient wraps the sync library with asyncio.to_thread.
        api = await self.linkedin_client._get_api()

        import asyncio

        def _resolve_urn_id() -> str:
            """Resolve public_id to a URN ID for get_profile_posts.

            LinkedIn deprecated the ``/identity/profiles/{id}/profileView``
            endpoint (returns 410), so ``get_profile_posts(public_id=...)``
            fails.  We resolve the URN via ``/me`` (own profile) or fall
            back to passing public_id directly.
            """
            try:
                res = api._fetch("/me")
                data = res.json()
                mini = data.get("miniProfile", {})
                if mini.get("publicIdentifier") == public_id:
                    urn = mini.get("entityUrn", "")
                    # entityUrn format: "urn:li:fs_miniProfile:ACoAAA..."
                    # We need just the ID part after the last colon
                    if urn:
                        return urn.split(":")[-1]
            except Exception:
                logger.debug("Failed to resolve URN via /me", exc_info=True)
            return ""

        def _fetch_posts() -> List[Dict[str, Any]]:
            # Try to resolve URN to bypass broken get_profile endpoint
            urn_id = _resolve_urn_id()
            if urn_id:
                logger.info("Resolved URN ID for '%s', fetching posts", public_id)
                return list(api.get_profile_posts(urn_id=urn_id, post_count=limit))

            # Fall back to public_id (may work on older library versions)
            logger.info("Using public_id '%s' directly", public_id)
            try:
                return list(api.get_profile_posts(public_id, post_count=limit))
            except Exception as exc:
                logger.error(
                    "Failed to fetch posts for '%s': %s", public_id, exc,
                )
                raise

        raw_posts: List[Dict[str, Any]] = await asyncio.to_thread(_fetch_posts)

        posts: List[Dict[str, Any]] = []
        for raw in raw_posts[:limit]:
            post = self._normalize_linkedin_post(raw)
            posts.append(post)

        validated = self._validate_posts(posts)

        logger.info(
            "Imported %d posts from LinkedIn (%d after validation)",
            len(posts),
            len(validated),
        )

        await self._persist_posts(validated)

        return validated

    # ------------------------------------------------------------------
    # JSON FILE IMPORT
    # ------------------------------------------------------------------

    async def import_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Import posts from a local JSON file.

        The JSON file must contain an array of post objects.  Each object
        should have at minimum a ``text`` key.  Optional keys: ``likes``,
        ``comments``, ``date``, ``content_type``, ``linkedin_post_id``.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of normalised post dicts.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not contain a JSON array.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        logger.info("Importing posts from JSON file: %s", file_path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected JSON array of posts, got {type(data).__name__}. "
                f"File: {file_path}"
            )

        posts: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict item in JSON file: %s", type(item).__name__
                )
                continue
            normalized = self._normalize_post(item)
            posts.append(normalized)

        validated = self._validate_posts(posts)

        logger.info(
            "Imported %d posts from JSON (%d after validation)",
            len(posts),
            len(validated),
        )

        await self._persist_posts(validated)

        return validated

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure a post dict has consistent keys.

        Handles common alternative key names (``content`` -> ``text``,
        ``reactions`` -> ``likes``, etc.).

        Args:
            post: Raw post dict from any source.

        Returns:
            Normalised post dict with keys: ``text``, ``likes``,
            ``comments``, ``date``, ``content_type``, and optionally
            ``linkedin_post_id``.
        """
        normalized: Dict[str, Any] = {
            "text": post.get("text", post.get("content", post.get("body", ""))),
            "likes": int(post.get("likes", post.get("reactions", 0))),
            "comments": int(post.get("comments", post.get("comment_count", 0))),
            "date": str(post.get("date", post.get("published_at", post.get("created_at", "")))),
            "content_type": str(post.get("content_type", post.get("type", "text"))),
        }

        # Preserve linkedin_post_id if present (for DB persistence)
        lid = post.get("linkedin_post_id", post.get("id", ""))
        if lid:
            normalized["linkedin_post_id"] = str(lid)

        return normalized

    def _normalize_linkedin_post(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a raw LinkedIn Voyager API post into standard format.

        The Voyager API returns deeply nested structures.  This method
        extracts the relevant fields and flattens them.

        Args:
            raw: Raw post dict from the linkedin-api library.

        Returns:
            Normalised post dict.
        """
        # Extract text from various Voyager response structures.
        # The Voyager v2 API nests text as:
        #   commentary -> text -> {textDirection, text}
        # while older versions use commentary -> text (str).
        text = ""
        commentary = raw.get("commentary", {})
        if isinstance(commentary, dict):
            raw_text = commentary.get("text", "")
            if isinstance(raw_text, dict):
                text = raw_text.get("text", "")
            else:
                text = raw_text
        elif isinstance(commentary, str):
            text = commentary

        if not text:
            # Try alternative field paths
            text = raw.get("text", raw.get("content", ""))

        # Extract engagement metrics
        social_detail = raw.get("socialDetail", {}) or {}
        likes = social_detail.get("totalSocialActivityCounts", {}).get("numLikes", 0)
        comments = social_detail.get("totalSocialActivityCounts", {}).get("numComments", 0)

        # Try simpler paths if nested ones are empty
        if likes == 0:
            likes = raw.get("likes", raw.get("numLikes", 0))
        if comments == 0:
            comments = raw.get("comments", raw.get("numComments", 0))

        # Extract date
        date = ""
        created_at = raw.get("createdAt", raw.get("publishedAt", ""))
        if created_at:
            date = str(created_at)

        # Extract LinkedIn URN as stable post identifier
        linkedin_post_id = (
            raw.get("entityUrn")
            or raw.get("urn")
            or raw.get("dashEntityUrn")
            or ""
        )

        result: Dict[str, Any] = {
            "text": text,
            "likes": int(likes),
            "comments": int(comments),
            "date": date,
            "content_type": "text",
        }
        if linkedin_post_id:
            result["linkedin_post_id"] = str(linkedin_post_id)

        return result

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out posts with missing or empty text.

        Args:
            posts: List of normalised post dicts.

        Returns:
            Filtered list containing only posts with non-empty ``text``.
        """
        valid: List[Dict[str, Any]] = []
        skipped = 0

        for post in posts:
            text = post.get("text", "")
            if not text or not text.strip():
                skipped += 1
                continue
            valid.append(post)

        if skipped > 0:
            logger.warning(
                "Filtered out %d posts with missing or empty text (kept %d)",
                skipped,
                len(valid),
            )

        return valid

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    async def _persist_posts(self, posts: List[Dict[str, Any]]) -> None:
        """Persist validated posts to the database (no-op when ``db`` is None).

        For each post that has a ``linkedin_post_id``, builds a row for the
        ``posts`` table and calls ``db.upsert_imported_posts()``.  If the
        post has engagement data (``likes`` / ``comments``), a baseline
        metrics snapshot is also stored via ``db.store_metrics_snapshot()``.
        """
        if self.db is None:
            return

        db_rows: List[Dict[str, Any]] = []
        for post in posts:
            lid = post.get("linkedin_post_id")
            if not lid:
                continue

            text = post.get("text", "")
            first_line = text.split("\n")[0].strip()[:210] if text else ""

            row: Dict[str, Any] = {
                "linkedin_post_id": lid,
                "text_content": text,
                "content_type": "community_content",
                "template_used": "imported",
                "hook": first_line,
            }

            # Map publication date if available
            date_str = post.get("date", "")
            if date_str:
                row["published_at"] = date_str

            db_rows.append(row)

        if not db_rows:
            return

        count = await self.db.upsert_imported_posts(db_rows)
        logger.info("Persisted %d imported posts to database", count)

        # Store baseline metrics for posts that have engagement data.
        # We need the DB-assigned post IDs, so look them up by linkedin_post_id.
        for post in posts:
            lid = post.get("linkedin_post_id")
            likes = post.get("likes", 0)
            comments = post.get("comments", 0)
            if not lid or (likes == 0 and comments == 0):
                continue

            # Look up the post row to get its UUID
            try:
                result = await (
                    self.db.client.table("posts")
                    .select("id")
                    .eq("linkedin_post_id", lid)
                    .limit(1)
                    .execute()
                )
                if not result.data:
                    continue

                post_id = result.data[0]["id"]
                await self.db.store_metrics_snapshot({
                    "post_id": post_id,
                    "likes": likes,
                    "comments": comments,
                    "minutes_after_post": 0,
                })
            except Exception:
                logger.warning(
                    "Failed to store baseline metrics for %s", lid, exc_info=True
                )

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_public_id(profile_url: str) -> str:
        """Extract the LinkedIn public ID from a profile URL.

        Args:
            profile_url: Full LinkedIn profile URL or bare public ID.

        Returns:
            The public ID string (e.g. ``"johndoe"``).
        """
        # Handle full URLs like "https://www.linkedin.com/in/johndoe/"
        url = profile_url.rstrip("/")
        if "/in/" in url:
            return url.split("/in/")[-1].split("/")[0].split("?")[0]
        # Assume it is already a public ID
        return url


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "ProfileImporter",
]
