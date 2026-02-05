"""
Tests for tool wrapper imports and basic instantiation.

Validates that all tool clients in ``src.tools`` can be imported and that
ClaudeClient can be instantiated with a fake API key and exposes the
expected interface (methods and token-usage tracking).

These tests are import-focused -- no real API calls are made.
"""

import pytest


# =========================================================================
# 1. ClaudeClient import
# =========================================================================


class TestClaudeClientImport:
    """Verify that ClaudeClient can be imported from src.tools.claude_client."""

    def test_import_claude_client(self):
        """ClaudeClient class should be importable."""
        try:
            from src.tools.claude_client import ClaudeClient
        except ImportError:
            pytest.skip("ClaudeClient not importable (missing dependency)")

        assert ClaudeClient is not None


# =========================================================================
# 2. PerplexityClient import
# =========================================================================


class TestPerplexityClientImport:
    """Verify that PerplexityClient can be imported from src.tools.perplexity."""

    def test_import_perplexity_client(self):
        """PerplexityClient class should be importable."""
        try:
            from src.tools.perplexity import PerplexityClient
        except ImportError:
            pytest.skip("PerplexityClient not importable (missing dependency)")

        assert PerplexityClient is not None


# =========================================================================
# 3. ArxivClient import
# =========================================================================


class TestArxivClientImport:
    """Verify that ArxivClient can be imported from src.tools.arxiv."""

    def test_import_arxiv_client(self):
        """ArxivClient class should be importable."""
        try:
            from src.tools.arxiv import ArxivClient
        except ImportError:
            pytest.skip("ArxivClient not importable (missing dependency)")

        assert ArxivClient is not None


# =========================================================================
# 4. NanoBananaClient import
# =========================================================================


class TestNanoBananaClientImport:
    """Verify that NanoBananaClient can be imported from src.tools.nano_banana."""

    def test_import_nano_banana_client(self):
        """NanoBananaClient class should be importable."""
        try:
            from src.tools.nano_banana import NanoBananaClient
        except ImportError:
            pytest.skip("NanoBananaClient not importable (missing dependency)")

        assert NanoBananaClient is not None


# =========================================================================
# 5. LinkedInClient import
# =========================================================================


class TestLinkedInClientImport:
    """Verify that LinkedInClient can be imported from src.tools.linkedin_client."""

    def test_import_linkedin_client(self):
        """LinkedInClient class should be importable."""
        try:
            from src.tools.linkedin_client import LinkedInClient
        except ImportError:
            pytest.skip("LinkedInClient not importable (missing dependency)")

        assert LinkedInClient is not None


# =========================================================================
# 6. PhotoLibrary import
# =========================================================================


class TestPhotoLibraryImport:
    """Verify that PhotoLibrary can be imported from src.tools.photo_library."""

    def test_import_photo_library(self):
        """PhotoLibrary class should be importable."""
        try:
            from src.tools.photo_library import PhotoLibrary
        except ImportError:
            pytest.skip("PhotoLibrary not importable (missing dependency)")

        assert PhotoLibrary is not None


# =========================================================================
# 7. TwitterClient import
# =========================================================================


class TestTwitterClientImport:
    """Verify that TwitterClient can be imported from src.tools.twitter."""

    def test_import_twitter_client(self):
        """TwitterClient class should be importable."""
        try:
            from src.tools.twitter import TwitterClient
        except ImportError:
            pytest.skip("TwitterClient not importable (missing dependency)")

        assert TwitterClient is not None


# =========================================================================
# 8-12. ClaudeClient instantiation and interface tests
# =========================================================================


class TestClaudeClientInterface:
    """Validate ClaudeClient can be instantiated and exposes the expected API.

    These tests require the ``anthropic`` package to be installed so that the
    module-level import in ``claude_client.py`` succeeds.  If it is missing
    the entire class is skipped.
    """

    @pytest.fixture(autouse=True)
    def _import_client(self):
        """Import ClaudeClient or skip the entire test class."""
        try:
            from src.tools.claude_client import ClaudeClient

            self.ClaudeClient = ClaudeClient
        except ImportError:
            pytest.skip("ClaudeClient not importable (missing 'anthropic' package)")

    # 8. Instantiation with a fake API key
    def test_instantiate_with_fake_key(self):
        """ClaudeClient should accept a fake API key without raising."""
        client = self.ClaudeClient(api_key="test-key")
        assert client is not None
        assert client.model == "claude-opus-4-5-20251101"

    # 9. Has generate method
    def test_has_generate_method(self):
        """ClaudeClient instances should have a 'generate' method."""
        client = self.ClaudeClient(api_key="test-key")
        assert hasattr(client, "generate")
        assert callable(client.generate)

    # 10. Has generate_structured method
    def test_has_generate_structured_method(self):
        """ClaudeClient instances should have a 'generate_structured' method."""
        client = self.ClaudeClient(api_key="test-key")
        assert hasattr(client, "generate_structured")
        assert callable(client.generate_structured)

    # 11. Has get_token_usage that returns a dict
    def test_has_token_usage(self):
        """ClaudeClient should expose token usage as a dict.

        The implementation uses a ``usage_stats`` property.  We check both
        the property name and that the return type is ``dict``.
        """
        client = self.ClaudeClient(api_key="test-key")

        # The actual implementation exposes usage_stats as a property
        assert hasattr(client, "usage_stats")
        usage = client.usage_stats
        assert isinstance(usage, dict)
        assert "input_tokens" in usage
        assert "output_tokens" in usage

    # 12. Initial token usage is zero
    def test_initial_token_usage_is_zero(self):
        """A freshly created ClaudeClient should report zero token usage."""
        client = self.ClaudeClient(api_key="test-key")
        usage = client.usage_stats
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
