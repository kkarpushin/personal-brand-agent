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

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
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

        # Download images locally before persisting (CDN URLs expire)
        if self.linkedin_client is not None:
            validated = await self._download_images(validated)

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
            "shares": int(post.get("shares", 0)),
            "date": str(post.get("date", post.get("published_at", post.get("created_at", "")))),
            "content_type": str(post.get("content_type", post.get("type", "text"))),
        }

        # Preserve linkedin_post_id if present (for DB persistence)
        lid = post.get("linkedin_post_id", post.get("id", ""))
        if lid:
            normalized["linkedin_post_id"] = str(lid)

        # Pass through visual fields if present in input JSON
        _passthrough_str = [
            "visual_type", "visual_url", "article_url", "article_title",
            "document_title", "document_url", "video_thumbnail", "original_author",
        ]
        for field in _passthrough_str:
            if field in post:
                normalized[field] = str(post[field])

        if "visual_urls" in post:
            normalized["visual_urls"] = list(post["visual_urls"])
        if "image_count" in post:
            normalized["image_count"] = int(post["image_count"])
        if "page_count" in post:
            normalized["page_count"] = int(post["page_count"])
        if "video_duration" in post:
            normalized["video_duration"] = float(post["video_duration"])
        if "reactions_by_type" in post:
            normalized["reactions_by_type"] = dict(post["reactions_by_type"])
        if "is_reshare" in post:
            normalized["is_reshare"] = bool(post["is_reshare"])

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
            # Try alternative field paths (but avoid picking up the content
            # component dict — only accept strings)
            alt = raw.get("text", "")
            if isinstance(alt, str):
                text = alt

        # Extract engagement metrics
        social_detail = raw.get("socialDetail", {}) or {}
        activity_counts = social_detail.get("totalSocialActivityCounts", {}) or {}
        likes = activity_counts.get("numLikes", 0)
        comments = activity_counts.get("numComments", 0)
        shares = activity_counts.get("numShares", 0)

        # Try simpler paths if nested ones are empty
        if likes == 0:
            likes = raw.get("likes", raw.get("numLikes", 0))
        if comments == 0:
            comments = raw.get("comments", raw.get("numComments", 0))
        if shares == 0:
            shares = raw.get("shares", raw.get("numShares", 0))

        # Extract reaction type breakdown (LIKE, EMPATHY, PRAISE, etc.)
        reaction_types: Dict[str, int] = {}
        reaction_counts = activity_counts.get("reactionTypeCounts", [])
        if isinstance(reaction_counts, list):
            for entry in reaction_counts:
                if isinstance(entry, dict):
                    rtype = entry.get("reactionType") or entry.get("type", "")
                    rcount = entry.get("count", 0)
                    if rtype:
                        reaction_types[str(rtype)] = int(rcount)

        # Extract LinkedIn URN as stable post identifier
        linkedin_post_id = (
            raw.get("entityUrn")
            or raw.get("urn")
            or raw.get("dashEntityUrn")
            or ""
        )

        # Extract date: try explicit fields first, fall back to snowflake ID
        date = ""
        created_at = raw.get("createdAt", raw.get("publishedAt", ""))
        if created_at:
            date = str(created_at)
        else:
            # Voyager often omits createdAt; extract from activity snowflake ID
            activity_urn = (
                raw.get("updateMetadata", {}).get("urn", "")
                or linkedin_post_id
            )
            date = self._extract_date_from_urn(activity_urn)

        # Extract media info from the raw Voyager response
        media_info = self._extract_media_info(raw)

        visual_urls = media_info["visual_urls"]

        # Detect reshares (reposts of other users' content)
        is_reshare = False
        original_author = ""
        reshared = raw.get("resharedUpdate")
        if isinstance(reshared, dict):
            is_reshare = True
            # Extract original author name from resharedUpdate.actor
            orig_actor = reshared.get("actor", {})
            if isinstance(orig_actor, dict):
                name_obj = orig_actor.get("name", {})
                if isinstance(name_obj, dict):
                    original_author = name_obj.get("text", "")
                elif isinstance(name_obj, str):
                    original_author = name_obj

        result: Dict[str, Any] = {
            "text": text,
            "likes": int(likes),
            "comments": int(comments),
            "shares": int(shares),
            "date": date,
            "content_type": media_info["visual_type"] if media_info["visual_type"] != "none" else "text",
            "visual_type": media_info["visual_type"],
            "visual_url": media_info["visual_url"],
            "visual_urls": visual_urls,
            "image_count": len(visual_urls),
            # Article metadata
            "article_url": media_info["article_url"],
            "article_title": media_info["article_title"],
            # Document (carousel/PDF) metadata
            "document_title": media_info["document_title"],
            "document_url": media_info["document_url"],
            "page_count": media_info["page_count"],
            # Video metadata
            "video_duration": media_info["video_duration"],
            "video_thumbnail": media_info["video_thumbnail"],
            # Reshare metadata
            "is_reshare": is_reshare,
            "original_author": original_author,
        }
        if linkedin_post_id:
            result["linkedin_post_id"] = str(linkedin_post_id)
        if reaction_types:
            result["reactions_by_type"] = reaction_types

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

            row: Dict[str, Any] = {
                "linkedin_post_id": lid,
                "text_content": text,
                "content_type": "community_content",
                "template_used": "imported",
                "hook": self._extract_hook(text),
            }

            # Map publication date if available
            date_str = post.get("date", "")
            if date_str:
                row["published_at"] = date_str

            # Include visual media fields if present
            visual_type = post.get("visual_type")
            if visual_type:
                row["visual_type"] = visual_type
            visual_url = post.get("visual_url")
            if visual_url:
                row["visual_url"] = visual_url
            visual_urls = post.get("visual_urls")
            if visual_urls:
                row["visual_urls"] = visual_urls
            image_count = post.get("image_count")
            if image_count is not None:
                row["image_count"] = image_count

            # Rich content metadata
            _optional_str_fields = [
                "article_url", "article_title", "document_title",
                "document_url", "video_thumbnail",
            ]
            for field in _optional_str_fields:
                val = post.get(field)
                if val:
                    row[field] = val

            shares = post.get("shares")
            if shares:
                row["shares"] = int(shares)
            page_count = post.get("page_count")
            if page_count:
                row["page_count"] = int(page_count)
            video_duration = post.get("video_duration")
            if video_duration:
                row["video_duration"] = float(video_duration)
            reactions_by_type = post.get("reactions_by_type")
            if reactions_by_type:
                row["reactions_by_type"] = reactions_by_type

            # Reshare metadata
            is_reshare = post.get("is_reshare")
            if is_reshare:
                row["is_reshare"] = True
            original_author = post.get("original_author")
            if original_author:
                row["original_author"] = original_author

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
    # MEDIA EXTRACTION
    # ------------------------------------------------------------------

    def _extract_media_info(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Detect media type and extract rich metadata from a raw Voyager post.

        Iterates over the ``content`` dict keys looking for known component
        type substrings (ImageComponent, DocumentComponent, VideoComponent,
        ArticleComponent).

        Args:
            raw: Raw post dict from the linkedin-api library.

        Returns:
            Dict with ``visual_type``, ``visual_url``, ``visual_urls``,
            and content-type-specific metadata (article_url, article_title,
            document_title, document_url, page_count, video_duration,
            video_thumbnail).
        """
        result: Dict[str, Any] = {
            "visual_type": "none",
            "visual_url": "",
            "visual_urls": [],
            "article_url": "",
            "article_title": "",
            "document_title": "",
            "document_url": "",
            "page_count": 0,
            "video_duration": 0.0,
            "video_thumbnail": "",
        }

        content = raw.get("content", {})
        if not isinstance(content, dict):
            return result

        for key, value in content.items():
            if "ImageComponent" in key:
                result["visual_type"] = "image"
                result["visual_urls"] = self._extract_image_urls(value)
                break
            elif "DocumentComponent" in key:
                result["visual_type"] = "document"
                doc_info = self._extract_document_info(value)
                result["document_title"] = doc_info["document_title"]
                result["document_url"] = doc_info["document_url"]
                result["page_count"] = doc_info["page_count"]
                result["visual_urls"] = doc_info["cover_images"]
                break
            elif "VideoComponent" in key or "LinkedInVideoComponent" in key:
                result["visual_type"] = "video"
                vid_info = self._extract_video_info(value)
                result["video_duration"] = vid_info["video_duration"]
                result["video_thumbnail"] = vid_info["video_thumbnail"]
                if vid_info["video_thumbnail"]:
                    result["visual_urls"] = [vid_info["video_thumbnail"]]
                break
            elif "CarouselComponent" in key:
                result["visual_type"] = "carousel"
                break
            elif "ArticleComponent" in key:
                result["visual_type"] = "article"
                art_info = self._extract_article_info(value)
                result["article_url"] = art_info["article_url"]
                result["article_title"] = art_info["article_title"]
                img_url = self._extract_article_image_url(value)
                if img_url:
                    result["visual_urls"] = [img_url]
                break

        result["visual_url"] = result["visual_urls"][0] if result["visual_urls"] else ""
        return result

    @staticmethod
    def _extract_image_urls(image_component: Any) -> List[str]:
        """Extract highest-resolution URLs for ALL images in an ImageComponent.

        Multi-image LinkedIn posts store multiple entries in the ``images``
        array within a single ``ImageComponent``.

        Args:
            image_component: The ImageComponent value from the content dict.

        Returns:
            List of full image URL strings (may be empty).
        """
        urls: List[str] = []
        try:
            if not isinstance(image_component, dict):
                return urls
            images = image_component.get("images", [])
            for img in images:
                attrs = img.get("attributes", [])
                if not attrs:
                    continue
                vector_image = attrs[0].get("vectorImage", {})
                if not vector_image:
                    continue
                root_url = vector_image.get("rootUrl", "")
                artifacts = vector_image.get("artifacts", [])
                if not artifacts or not root_url:
                    continue
                best = max(artifacts, key=lambda a: a.get("width", 0))
                segment = best.get("fileIdentifyingUrlPathSegment", "")
                if segment:
                    urls.append(root_url + segment)
        except (IndexError, KeyError, TypeError):
            pass
        return urls

    @staticmethod
    def _extract_document_info(doc_component: Any) -> Dict[str, Any]:
        """Extract metadata from a DocumentComponent (carousel/PDF).

        The Voyager API nests document data under a ``document`` key::

            {
              "showDownloadCTA": false,
              "document": {
                "title": "...",
                "totalPageCount": 19,
                "transcribedDocumentUrl": "...",
                "coverPages": {
                  "pagesPerResolution": [
                    {"width": 483, "imageUrls": ["...", "..."]},
                    {"width": 1282, "imageUrls": ["...", "..."]}
                  ]
                }
              }
            }

        Args:
            doc_component: The DocumentComponent value from the content dict.

        Returns:
            Dict with ``document_title``, ``document_url``, ``page_count``,
            and ``cover_images`` (list of cover page image URLs at highest
            resolution).
        """
        info: Dict[str, Any] = {
            "document_title": "",
            "document_url": "",
            "page_count": 0,
            "cover_images": [],
        }
        try:
            if not isinstance(doc_component, dict):
                return info
            # Metadata lives under the "document" key
            doc = doc_component.get("document", doc_component)
            if not isinstance(doc, dict):
                return info
            info["document_title"] = str(doc.get("title", "") or "")
            info["page_count"] = int(doc.get("totalPageCount", 0) or 0)
            info["document_url"] = str(
                doc.get("transcribedDocumentUrl", "") or ""
            )
            # Cover pages: pick highest resolution from pagesPerResolution
            cover_pages = doc.get("coverPages", {})
            if isinstance(cover_pages, dict):
                resolutions = cover_pages.get("pagesPerResolution", [])
                if isinstance(resolutions, list) and resolutions:
                    best_res = max(
                        resolutions, key=lambda r: r.get("width", 0)
                    )
                    image_urls = best_res.get("imageUrls", [])
                    if isinstance(image_urls, list):
                        info["cover_images"] = [
                            u for u in image_urls if isinstance(u, str) and u
                        ]
        except (IndexError, KeyError, TypeError, ValueError):
            pass
        return info

    @staticmethod
    def _extract_video_info(video_component: Any) -> Dict[str, Any]:
        """Extract metadata from a VideoComponent / LinkedInVideoComponent.

        The thumbnail structure uses ``rootUrl`` + ``artifacts`` directly
        (no ``vectorImage`` wrapper)::

            "thumbnail": {
              "rootUrl": "https://media.licdn.com/.../videocover-",
              "artifacts": [
                {"width": 1314, "fileIdentifyingUrlPathSegment": "high/..."},
                {"width": 656, "fileIdentifyingUrlPathSegment": "low/..."}
              ]
            }

        Args:
            video_component: The VideoComponent value from the content dict.

        Returns:
            Dict with ``video_duration`` (float, raw API value) and
            ``video_thumbnail`` (str URL at highest resolution).
        """
        info: Dict[str, Any] = {
            "video_duration": 0.0,
            "video_thumbnail": "",
        }
        try:
            if not isinstance(video_component, dict):
                return info
            meta = video_component.get("videoPlayMetadata", video_component)
            if not isinstance(meta, dict):
                return info
            info["video_duration"] = float(meta.get("duration", 0) or 0)
            # Thumbnail: rootUrl + artifacts (direct, not nested under vectorImage)
            thumb = meta.get("thumbnail", {})
            if isinstance(thumb, dict):
                root_url = thumb.get("rootUrl", "")
                artifacts = thumb.get("artifacts", [])
                if root_url and isinstance(artifacts, list) and artifacts:
                    best = max(artifacts, key=lambda a: a.get("width", 0))
                    segment = best.get("fileIdentifyingUrlPathSegment", "")
                    if segment:
                        info["video_thumbnail"] = root_url + segment
        except (IndexError, KeyError, TypeError, ValueError):
            pass
        return info

    @staticmethod
    def _extract_article_info(article_component: Any) -> Dict[str, Any]:
        """Extract URL and title from an ArticleComponent.

        Args:
            article_component: The ArticleComponent value from content dict.

        Returns:
            Dict with ``article_url`` and ``article_title``.
        """
        info: Dict[str, Any] = {
            "article_url": "",
            "article_title": "",
        }
        try:
            if not isinstance(article_component, dict):
                return info
            # Article URL from navigationContext
            nav = article_component.get("navigationContext", {})
            if isinstance(nav, dict):
                info["article_url"] = str(nav.get("actionTarget", "") or "")
            # Title
            title_obj = article_component.get("title", {})
            if isinstance(title_obj, dict):
                info["article_title"] = str(title_obj.get("text", "") or "")
            elif isinstance(title_obj, str):
                info["article_title"] = title_obj
        except (IndexError, KeyError, TypeError):
            pass
        return info

    @staticmethod
    def _extract_article_image_url(article_component: Any) -> str:
        """Extract the large image URL from an ArticleComponent.

        Args:
            article_component: The ArticleComponent value from content dict.

        Returns:
            Image URL string, or empty string if not found.
        """
        try:
            if not isinstance(article_component, dict):
                return ""
            large_image = article_component.get("largeImage", {})
            if not isinstance(large_image, dict):
                return ""
            attrs = large_image.get("attributes", [])
            if not attrs:
                return ""
            vector_image = attrs[0].get("vectorImage", {})
            if not vector_image:
                return ""
            root_url = vector_image.get("rootUrl", "")
            artifacts = vector_image.get("artifacts", [])
            if not artifacts or not root_url:
                return ""
            best = max(artifacts, key=lambda a: a.get("width", 0))
            segment = best.get("fileIdentifyingUrlPathSegment", "")
            if segment:
                return root_url + segment
        except (IndexError, KeyError, TypeError):
            pass
        return ""

    # ------------------------------------------------------------------
    # IMAGE DOWNLOAD
    # ------------------------------------------------------------------

    async def _download_images(
        self,
        posts: List[Dict[str, Any]],
        output_dir: str = "data/imported_images",
    ) -> List[Dict[str, Any]]:
        """Download images from LinkedIn CDN to local storage.

        LinkedIn CDN URLs expire, so local copies preserve images for
        future analysis.  Downloads run in a background thread to avoid
        blocking the event loop.

        Args:
            posts: List of normalised post dicts (may contain ``visual_url``).
            output_dir: Directory to save images to.

        Returns:
            The same list with ``visual_url`` updated to local paths on
            success.  On failure, the CDN URL is kept as-is.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Get the requests session from the LinkedIn client for auth cookies
        try:
            api = await self.linkedin_client._get_api()
            session = api.client.session
        except Exception:
            logger.warning("Could not get LinkedIn session for image downloads")
            return posts

        def _download_one(url: str, dest: Path) -> bool:
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                return True
            except Exception as exc:
                logger.warning("Failed to download image %s: %s", url, exc)
                return False

        for post in posts:
            urls = post.get("visual_urls", [])
            if not urls:
                # Fallback for single-URL posts (e.g. from JSON import)
                single = post.get("visual_url", "")
                if single and single.startswith("http"):
                    urls = [single]
                else:
                    continue

            urn = post.get("linkedin_post_id", "")
            local_paths: List[str] = []

            for idx, url in enumerate(urls):
                if not url.startswith("http"):
                    local_paths.append(url)
                    continue

                # Hash URN + index for unique filenames per image
                hash_input = f"{urn or url}:{idx}"
                file_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
                dest = out_path / f"{file_hash}.jpg"

                if dest.exists():
                    local_paths.append(str(dest))
                    continue

                ok = await asyncio.to_thread(_download_one, url, dest)
                if ok:
                    logger.debug("Downloaded image to %s", dest)
                    local_paths.append(str(dest))
                else:
                    local_paths.append(url)  # keep CDN URL on failure

            post["visual_urls"] = local_paths
            post["visual_url"] = local_paths[0] if local_paths else ""
            post["image_count"] = len(local_paths)

        return posts

    # ------------------------------------------------------------------
    # HOOK EXTRACTION
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hook(text: str, max_length: int = 300) -> str:
        """Extract the hook (above-the-fold text) from a LinkedIn post.

        LinkedIn authors use a blank line (``\\n\\n``) to separate the
        attention-grabbing hook from the body.  This method takes the
        text before the first blank line, capped at *max_length*
        characters on a word boundary.

        Args:
            text: Full post text.
            max_length: Maximum hook length in characters (default 300).

        Returns:
            The extracted hook string.
        """
        if not text or not text.strip():
            return ""

        # Split on double newline — the author's intentional boundary
        paragraphs = text.split("\n\n")
        hook = paragraphs[0].strip()

        # If the first block is very short (e.g. just an emoji or a tag),
        # include the next paragraph too
        if len(hook) < 10 and len(paragraphs) > 1:
            hook = hook + "\n" + paragraphs[1].strip()

        # Cap at max_length on a word boundary
        if len(hook) > max_length:
            truncated = hook[:max_length]
            # Find the last space to avoid cutting mid-word
            last_space = truncated.rfind(" ")
            if last_space > max_length // 2:
                hook = truncated[:last_space] + "…"
            else:
                hook = truncated + "…"

        return hook

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_date_from_urn(urn: str) -> str:
        """Extract an ISO date string from a LinkedIn activity snowflake ID.

        LinkedIn activity IDs encode a millisecond timestamp in the upper
        bits (``activity_id >> 22``).  This gives an accurate publication
        date even when the API omits ``createdAt``.

        Args:
            urn: A LinkedIn URN string containing ``activity:NNNN``.

        Returns:
            ISO 8601 timestamp string (e.g. ``"2023-09-19T18:30:32+00:00"``),
            or empty string if the URN cannot be parsed.
        """
        match = re.search(r"activity:(\d+)", urn)
        if not match:
            return ""
        try:
            activity_id = int(match.group(1))
            ts_ms = activity_id >> 22
            # Sanity check: should be between 2010 and 2030
            if ts_ms < 1_262_304_000_000 or ts_ms > 1_893_456_000_000:
                return ""
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            return dt.isoformat()
        except (ValueError, OverflowError, OSError):
            return ""

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
