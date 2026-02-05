"""Tests for ProfileImporter media extraction and visual pattern analysis."""

import pytest

from src.author.profile_importer import ProfileImporter
from src.author.author_profile_agent import AuthorProfileAgent


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def importer():
    """ProfileImporter with no LinkedIn client or DB."""
    return ProfileImporter()


# =========================================================================
# TestMediaExtraction — _extract_media_info()
# =========================================================================

class TestMediaExtraction:
    """Tests for _extract_media_info() with various Voyager content types."""

    def test_text_only_post(self, importer):
        raw = {"commentary": {"text": "Hello world"}}
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "none"
        assert result["visual_url"] == ""
        assert result["visual_urls"] == []

    def test_image_post(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ImageComponent": {
                    "images": [
                        {
                            "attributes": [
                                {
                                    "vectorImage": {
                                        "rootUrl": "https://media.licdn.com/",
                                        "artifacts": [
                                            {
                                                "width": 800,
                                                "fileIdentifyingUrlPathSegment": "img_800.jpg",
                                            },
                                            {
                                                "width": 400,
                                                "fileIdentifyingUrlPathSegment": "img_400.jpg",
                                            },
                                        ],
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "image"
        assert result["visual_url"] == "https://media.licdn.com/img_800.jpg"
        assert result["visual_urls"] == ["https://media.licdn.com/img_800.jpg"]

    def test_multi_image_post(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ImageComponent": {
                    "images": [
                        {
                            "attributes": [{
                                "vectorImage": {
                                    "rootUrl": "https://media.licdn.com/",
                                    "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "img1.jpg"}],
                                }
                            }]
                        },
                        {
                            "attributes": [{
                                "vectorImage": {
                                    "rootUrl": "https://media.licdn.com/",
                                    "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "img2.jpg"}],
                                }
                            }]
                        },
                        {
                            "attributes": [{
                                "vectorImage": {
                                    "rootUrl": "https://media.licdn.com/",
                                    "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "img3.jpg"}],
                                }
                            }]
                        },
                    ]
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "image"
        assert len(result["visual_urls"]) == 3
        assert result["visual_url"] == "https://media.licdn.com/img1.jpg"
        assert result["visual_urls"][1] == "https://media.licdn.com/img2.jpg"
        assert result["visual_urls"][2] == "https://media.licdn.com/img3.jpg"

    def test_video_post(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.LinkedInVideoComponent": {
                    "videoPlayMetadata": {}
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "video"
        assert result["visual_url"] == ""

    def test_video_component_alt_key(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.VideoComponent": {
                    "videoPlayMetadata": {}
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "video"

    def test_carousel_post(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.CarouselComponent": {
                    "slides": []
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "carousel"
        assert result["visual_url"] == ""

    def test_article_post(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ArticleComponent": {
                    "largeImage": {
                        "attributes": [
                            {
                                "vectorImage": {
                                    "rootUrl": "https://media.licdn.com/",
                                    "artifacts": [
                                        {
                                            "width": 1200,
                                            "fileIdentifyingUrlPathSegment": "article_1200.jpg",
                                        },
                                    ],
                                }
                            }
                        ]
                    }
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "article"
        assert result["visual_url"] == "https://media.licdn.com/article_1200.jpg"
        assert result["visual_urls"] == ["https://media.licdn.com/article_1200.jpg"]

    def test_empty_content_dict(self, importer):
        raw = {"content": {}}
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "none"

    def test_content_not_dict(self, importer):
        raw = {"content": "some string"}
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "none"

    def test_no_content_key(self, importer):
        raw = {"commentary": "text"}
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "none"

    def test_image_missing_artifacts(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ImageComponent": {
                    "images": [{"attributes": [{"vectorImage": {"rootUrl": "x", "artifacts": []}}]}]
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "image"
        assert result["visual_url"] == ""

    def test_image_missing_images(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ImageComponent": {"images": []}
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "image"
        assert result["visual_url"] == ""


# =========================================================================
# TestNormalizeLinkedInPost — _normalize_linkedin_post()
# =========================================================================

class TestNormalizeLinkedInPost:
    """Tests for _normalize_linkedin_post() with visual fields."""

    def test_text_post_has_visual_type_none(self, importer):
        raw = {
            "commentary": {"text": "Hello"},
            "socialDetail": {
                "totalSocialActivityCounts": {"numLikes": 10, "numComments": 2}
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["visual_type"] == "none"
        assert result["visual_url"] == ""
        assert result["visual_urls"] == []
        assert result["image_count"] == 0
        assert result["content_type"] == "text"

    def test_image_post_sets_visual_fields(self, importer):
        raw = {
            "commentary": {"text": "Check this out"},
            "content": {
                "com.linkedin.voyager.feed.render.ImageComponent": {
                    "images": [
                        {
                            "attributes": [
                                {
                                    "vectorImage": {
                                        "rootUrl": "https://cdn.licdn.com/",
                                        "artifacts": [
                                            {"width": 600, "fileIdentifyingUrlPathSegment": "pic.jpg"},
                                        ],
                                    }
                                }
                            ]
                        }
                    ]
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["visual_type"] == "image"
        assert result["visual_url"] == "https://cdn.licdn.com/pic.jpg"
        assert result["visual_urls"] == ["https://cdn.licdn.com/pic.jpg"]
        assert result["image_count"] == 1
        assert result["content_type"] == "image"

    def test_multi_image_post_sets_all_urls(self, importer):
        raw = {
            "commentary": {"text": "Three photos"},
            "content": {
                "com.linkedin.voyager.feed.render.ImageComponent": {
                    "images": [
                        {"attributes": [{"vectorImage": {"rootUrl": "https://cdn.licdn.com/", "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "a.jpg"}]}}]},
                        {"attributes": [{"vectorImage": {"rootUrl": "https://cdn.licdn.com/", "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "b.jpg"}]}}]},
                        {"attributes": [{"vectorImage": {"rootUrl": "https://cdn.licdn.com/", "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "c.jpg"}]}}]},
                    ]
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["visual_type"] == "image"
        assert result["image_count"] == 3
        assert len(result["visual_urls"]) == 3
        assert result["visual_url"] == "https://cdn.licdn.com/a.jpg"

    def test_video_post_content_type(self, importer):
        raw = {
            "commentary": {"text": "Watch this"},
            "content": {
                "com.linkedin.voyager.feed.render.LinkedInVideoComponent": {}
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["visual_type"] == "video"
        assert result["content_type"] == "video"


# =========================================================================
# TestNormalizePost — _normalize_post() for JSON imports
# =========================================================================

class TestNormalizePost:
    """Tests for _normalize_post() passing through visual fields."""

    def test_passthrough_visual_fields(self, importer):
        post = {
            "text": "Some post",
            "visual_type": "image",
            "visual_url": "/path/to/image.jpg",
        }
        result = importer._normalize_post(post)
        assert result["visual_type"] == "image"
        assert result["visual_url"] == "/path/to/image.jpg"

    def test_no_visual_fields_by_default(self, importer):
        post = {"text": "Just text"}
        result = importer._normalize_post(post)
        assert "visual_type" not in result
        assert "visual_url" not in result


# =========================================================================
# TestExtractHook — _extract_hook()
# =========================================================================

class TestExtractHook:
    """Tests for _extract_hook() LinkedIn hook extraction."""

    def test_hook_before_blank_line(self):
        text = "Bold statement 🔥\n\nBody text here that continues..."
        assert ProfileImporter._extract_hook(text) == "Bold statement 🔥"

    def test_multiline_hook(self):
        text = "Line one\nLine two\n\nBody paragraph."
        assert ProfileImporter._extract_hook(text) == "Line one\nLine two"

    def test_long_paragraph_truncated_at_word(self):
        text = "Word " * 100  # 500 chars, no \n\n
        result = ProfileImporter._extract_hook(text)
        assert len(result) <= 301  # 300 + ellipsis
        assert result.endswith("…")
        assert "  " not in result  # didn't cut mid-word

    def test_short_first_block_merges_next(self):
        text = "🚀\n\nThe real hook starts here."
        result = ProfileImporter._extract_hook(text)
        assert "The real hook starts here." in result

    def test_empty_text(self):
        assert ProfileImporter._extract_hook("") == ""
        assert ProfileImporter._extract_hook("   ") == ""

    def test_no_blank_line(self):
        text = "Single paragraph without any double newlines, just regular text."
        assert ProfileImporter._extract_hook(text) == text

    def test_real_linkedin_post(self):
        text = (
            "Why did I stop giving my team specific goals? 🎯\n\n"
            "As a leader, my job isn't to micromanage — it's to empower my team. "
            "That's why I decided to stop giving specific goals and instead "
            "focus on creating an environment where my team could thrive."
        )
        assert ProfileImporter._extract_hook(text) == "Why did I stop giving my team specific goals? 🎯"

    def test_preserves_exact_text_no_truncation(self):
        text = "Perfection is overrated 💯\n\nFor years, I believed..."
        assert ProfileImporter._extract_hook(text) == "Perfection is overrated 💯"


# =========================================================================
# TestVisualPatterns — _compute_visual_patterns()
# =========================================================================

class TestVisualPatterns:
    """Tests for AuthorProfileAgent._compute_visual_patterns()."""

    def test_empty_posts(self):
        result = AuthorProfileAgent._compute_visual_patterns([])
        assert result["visual_content_ratio"] == 0.0
        assert result["preferred_visual_types"] == []
        assert result["visual_type_performance"] == {}

    def test_all_text_posts(self):
        posts = [
            {"text": "a", "likes": 5, "comments": 1},
            {"text": "b", "likes": 3, "comments": 0},
        ]
        result = AuthorProfileAgent._compute_visual_patterns(posts)
        assert result["visual_content_ratio"] == 0.0
        assert result["preferred_visual_types"] == []

    def test_mixed_posts(self):
        posts = [
            {"text": "a", "likes": 10, "comments": 2, "visual_type": "image"},
            {"text": "b", "likes": 5, "comments": 0, "visual_type": "image"},
            {"text": "c", "likes": 20, "comments": 5, "visual_type": "video"},
            {"text": "d", "likes": 3, "comments": 1, "visual_type": "none"},
        ]
        result = AuthorProfileAgent._compute_visual_patterns(posts)
        # 3 out of 4 posts have visual content
        assert result["visual_content_ratio"] == 0.75
        # image appears 2x, video 1x
        assert result["preferred_visual_types"] == ["image", "video"]
        # Engagement: image avg = (10+6+5+0)/2 = 10.5 ... wait
        # engagement = likes + comments * 3
        # post a: 10 + 2*3 = 16
        # post b: 5 + 0*3 = 5
        # post c: 20 + 5*3 = 35
        # post d: 3 + 1*3 = 6
        # image avg = (16 + 5) / 2 = 10.5
        # video avg = 35 / 1 = 35.0
        # none avg = 6 / 1 = 6.0
        assert result["visual_type_performance"]["image"] == 10.5
        assert result["visual_type_performance"]["video"] == 35.0
        assert result["visual_type_performance"]["none"] == 6.0

    def test_single_visual_type(self):
        posts = [
            {"text": "a", "likes": 10, "comments": 0, "visual_type": "carousel"},
            {"text": "b", "likes": 8, "comments": 0, "visual_type": "carousel"},
        ]
        result = AuthorProfileAgent._compute_visual_patterns(posts)
        assert result["visual_content_ratio"] == 1.0
        assert result["preferred_visual_types"] == ["carousel"]
        assert result["visual_type_performance"]["carousel"] == 9.0

    def test_no_visual_type_key_treated_as_none(self):
        posts = [
            {"text": "a", "likes": 5, "comments": 0},
            {"text": "b", "likes": 3, "comments": 0, "visual_type": "image"},
        ]
        result = AuthorProfileAgent._compute_visual_patterns(posts)
        assert result["visual_content_ratio"] == 0.5
        assert result["preferred_visual_types"] == ["image"]


# =========================================================================
# TestExtractImageUrl — static helpers
# =========================================================================

class TestExtractImageUrls:
    """Tests for _extract_image_urls and _extract_article_image_url."""

    def test_extract_image_urls_picks_max_width(self):
        component = {
            "images": [
                {
                    "attributes": [
                        {
                            "vectorImage": {
                                "rootUrl": "https://media.licdn.com/",
                                "artifacts": [
                                    {"width": 400, "fileIdentifyingUrlPathSegment": "small.jpg"},
                                    {"width": 1200, "fileIdentifyingUrlPathSegment": "large.jpg"},
                                    {"width": 800, "fileIdentifyingUrlPathSegment": "medium.jpg"},
                                ],
                            }
                        }
                    ]
                }
            ]
        }
        result = ProfileImporter._extract_image_urls(component)
        assert result == ["https://media.licdn.com/large.jpg"]

    def test_extract_image_urls_multiple(self):
        component = {
            "images": [
                {"attributes": [{"vectorImage": {"rootUrl": "https://cdn.licdn.com/", "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "a.jpg"}]}}]},
                {"attributes": [{"vectorImage": {"rootUrl": "https://cdn.licdn.com/", "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "b.jpg"}]}}]},
            ]
        }
        result = ProfileImporter._extract_image_urls(component)
        assert len(result) == 2
        assert result[0] == "https://cdn.licdn.com/a.jpg"
        assert result[1] == "https://cdn.licdn.com/b.jpg"

    def test_extract_image_urls_none_input(self):
        assert ProfileImporter._extract_image_urls(None) == []

    def test_extract_image_urls_empty_dict(self):
        assert ProfileImporter._extract_image_urls({}) == []

    def test_extract_article_image_url(self):
        component = {
            "largeImage": {
                "attributes": [
                    {
                        "vectorImage": {
                            "rootUrl": "https://media.licdn.com/",
                            "artifacts": [
                                {"width": 800, "fileIdentifyingUrlPathSegment": "art.jpg"},
                            ],
                        }
                    }
                ]
            }
        }
        result = ProfileImporter._extract_article_image_url(component)
        assert result == "https://media.licdn.com/art.jpg"

    def test_extract_article_image_url_empty(self):
        assert ProfileImporter._extract_article_image_url({}) == ""
        assert ProfileImporter._extract_article_image_url(None) == ""


# =========================================================================
# TestDocumentExtraction — DocumentComponent (carousels/PDFs)
# =========================================================================

class TestDocumentExtraction:
    """Tests for DocumentComponent detection and metadata extraction."""

    def test_document_post_detected(self, importer):
        """DocumentComponent nests data under 'document' key."""
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.DocumentComponent": {
                    "document": {
                        "title": "My Carousel",
                        "totalPageCount": 10,
                        "transcribedDocumentUrl": "https://example.com/doc.pdf",
                    }
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "document"
        assert result["document_title"] == "My Carousel"
        assert result["page_count"] == 10
        assert result["document_url"] == "https://example.com/doc.pdf"

    def test_document_with_cover_pages(self, importer):
        """Cover pages use pagesPerResolution with imageUrls arrays."""
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.DocumentComponent": {
                    "document": {
                        "title": "Slides",
                        "totalPageCount": 5,
                        "coverPages": {
                            "pagesPerResolution": [
                                {
                                    "width": 483,
                                    "imageUrls": [
                                        "https://media.licdn.com/cover1_small.jpg",
                                        "https://media.licdn.com/cover2_small.jpg",
                                    ],
                                },
                                {
                                    "width": 1282,
                                    "imageUrls": [
                                        "https://media.licdn.com/cover1_large.jpg",
                                        "https://media.licdn.com/cover2_large.jpg",
                                    ],
                                },
                            ]
                        },
                    }
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "document"
        # Should pick highest resolution (width=1282)
        assert len(result["visual_urls"]) == 2
        assert result["visual_url"] == "https://media.licdn.com/cover1_large.jpg"
        assert result["visual_urls"][1] == "https://media.licdn.com/cover2_large.jpg"

    def test_document_empty_component(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.DocumentComponent": {}
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "document"
        assert result["document_title"] == ""
        assert result["page_count"] == 0

    def test_extract_document_info_not_dict(self):
        result = ProfileImporter._extract_document_info("string")
        assert result["document_title"] == ""
        assert result["page_count"] == 0
        assert result["cover_images"] == []

    def test_extract_document_info_none(self):
        result = ProfileImporter._extract_document_info(None)
        assert result["document_title"] == ""

    def test_document_in_normalize_linkedin_post(self, importer):
        raw = {
            "commentary": {"text": "Check out my slides"},
            "content": {
                "com.linkedin.voyager.feed.render.DocumentComponent": {
                    "document": {
                        "title": "AI Trends 2025",
                        "totalPageCount": 12,
                        "transcribedDocumentUrl": "https://example.com/slides.pdf",
                    }
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["visual_type"] == "document"
        assert result["content_type"] == "document"
        assert result["document_title"] == "AI Trends 2025"
        assert result["page_count"] == 12
        assert result["document_url"] == "https://example.com/slides.pdf"


# =========================================================================
# TestArticleExtraction — ArticleComponent URL and title
# =========================================================================

class TestArticleExtraction:
    """Tests for ArticleComponent URL and title extraction."""

    def test_article_with_url_and_title(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ArticleComponent": {
                    "navigationContext": {
                        "actionTarget": "https://example.com/my-article",
                    },
                    "title": {"text": "My Article Title"},
                    "largeImage": {
                        "attributes": [{
                            "vectorImage": {
                                "rootUrl": "https://media.licdn.com/",
                                "artifacts": [{"width": 800, "fileIdentifyingUrlPathSegment": "art.jpg"}],
                            }
                        }]
                    },
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "article"
        assert result["article_url"] == "https://example.com/my-article"
        assert result["article_title"] == "My Article Title"
        assert result["visual_url"] == "https://media.licdn.com/art.jpg"

    def test_article_without_nav_context(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.ArticleComponent": {
                    "title": {"text": "Title Only"},
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "article"
        assert result["article_url"] == ""
        assert result["article_title"] == "Title Only"

    def test_article_string_title(self):
        result = ProfileImporter._extract_article_info({"title": "Plain String Title"})
        assert result["article_title"] == "Plain String Title"

    def test_article_info_empty(self):
        result = ProfileImporter._extract_article_info({})
        assert result["article_url"] == ""
        assert result["article_title"] == ""

    def test_article_info_none(self):
        result = ProfileImporter._extract_article_info(None)
        assert result["article_url"] == ""

    def test_article_in_normalize_linkedin_post(self, importer):
        raw = {
            "commentary": {"text": "Read my article"},
            "content": {
                "com.linkedin.voyager.feed.render.ArticleComponent": {
                    "navigationContext": {"actionTarget": "https://blog.example.com/post"},
                    "title": {"text": "Blog Post"},
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["article_url"] == "https://blog.example.com/post"
        assert result["article_title"] == "Blog Post"


# =========================================================================
# TestVideoExtraction — VideoComponent duration and thumbnail
# =========================================================================

class TestVideoExtraction:
    """Tests for VideoComponent duration and thumbnail extraction."""

    def test_video_with_metadata(self, importer):
        """Video thumbnail uses rootUrl + artifacts directly (no vectorImage)."""
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.LinkedInVideoComponent": {
                    "videoPlayMetadata": {
                        "duration": 38800,
                        "thumbnail": {
                            "rootUrl": "https://media.licdn.com/videocover-",
                            "artifacts": [
                                {"width": 1314, "fileIdentifyingUrlPathSegment": "high/thumb.jpg"},
                                {"width": 656, "fileIdentifyingUrlPathSegment": "low/thumb.jpg"},
                            ],
                        },
                    }
                }
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "video"
        assert result["video_duration"] == 38800.0
        assert result["video_thumbnail"] == "https://media.licdn.com/videocover-high/thumb.jpg"
        assert result["visual_url"] == "https://media.licdn.com/videocover-high/thumb.jpg"

    def test_video_without_metadata(self, importer):
        raw = {
            "content": {
                "com.linkedin.voyager.feed.render.LinkedInVideoComponent": {}
            }
        }
        result = importer._extract_media_info(raw)
        assert result["visual_type"] == "video"
        assert result["video_duration"] == 0.0
        assert result["video_thumbnail"] == ""

    def test_extract_video_info_not_dict(self):
        result = ProfileImporter._extract_video_info("string")
        assert result["video_duration"] == 0.0
        assert result["video_thumbnail"] == ""

    def test_extract_video_info_none(self):
        result = ProfileImporter._extract_video_info(None)
        assert result["video_duration"] == 0.0

    def test_video_in_normalize_linkedin_post(self, importer):
        raw = {
            "commentary": {"text": "Watch this"},
            "content": {
                "com.linkedin.voyager.feed.render.LinkedInVideoComponent": {
                    "videoPlayMetadata": {
                        "duration": 60000,
                    }
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["visual_type"] == "video"
        assert result["video_duration"] == 60000.0


# =========================================================================
# TestSharesAndReactions — engagement metadata
# =========================================================================

class TestSharesAndReactions:
    """Tests for shares count and reaction type breakdown extraction."""

    def test_shares_extracted(self, importer):
        raw = {
            "commentary": {"text": "Hello"},
            "socialDetail": {
                "totalSocialActivityCounts": {
                    "numLikes": 10,
                    "numComments": 2,
                    "numShares": 5,
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["shares"] == 5

    def test_reaction_types_extracted(self, importer):
        raw = {
            "commentary": {"text": "Hello"},
            "socialDetail": {
                "totalSocialActivityCounts": {
                    "numLikes": 20,
                    "numComments": 3,
                    "numShares": 1,
                    "reactionTypeCounts": [
                        {"reactionType": "LIKE", "count": 15},
                        {"reactionType": "EMPATHY", "count": 3},
                        {"reactionType": "PRAISE", "count": 2},
                    ],
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert result["reactions_by_type"]["LIKE"] == 15
        assert result["reactions_by_type"]["EMPATHY"] == 3
        assert result["reactions_by_type"]["PRAISE"] == 2

    def test_no_shares_defaults_zero(self, importer):
        raw = {"commentary": {"text": "Hello"}}
        result = importer._normalize_linkedin_post(raw)
        assert result["shares"] == 0

    def test_no_reactions_no_key(self, importer):
        raw = {"commentary": {"text": "Hello"}}
        result = importer._normalize_linkedin_post(raw)
        assert "reactions_by_type" not in result

    def test_empty_reaction_list(self, importer):
        raw = {
            "commentary": {"text": "Hello"},
            "socialDetail": {
                "totalSocialActivityCounts": {
                    "numLikes": 5,
                    "reactionTypeCounts": [],
                }
            },
        }
        result = importer._normalize_linkedin_post(raw)
        assert "reactions_by_type" not in result

    def test_shares_in_normalize_post(self, importer):
        post = {"text": "Some post", "shares": 3}
        result = importer._normalize_post(post)
        assert result["shares"] == 3
