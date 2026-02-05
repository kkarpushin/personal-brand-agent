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
        assert result["content_type"] == "image"

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

class TestExtractImageUrl:
    """Tests for _extract_image_url and _extract_article_image_url."""

    def test_extract_image_url_picks_max_width(self):
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
        result = ProfileImporter._extract_image_url(component)
        assert result == "https://media.licdn.com/large.jpg"

    def test_extract_image_url_none_input(self):
        assert ProfileImporter._extract_image_url(None) == ""

    def test_extract_image_url_empty_dict(self):
        assert ProfileImporter._extract_image_url({}) == ""

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
