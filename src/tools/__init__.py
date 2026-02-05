"""
External tool wrappers for LinkedIn Super Agent.

This package provides async clients for all external services used by the
agent pipeline:

- ClaudeClient: Anthropic Claude API for LLM calls (not Claude Code CLI)
- PerplexityClient: Perplexity AI search API for trend research
- ArxivClient: ArXiv paper search for primary source discovery
- NanoBananaClient: Image generation via Laozhang.ai (Nano Banana Pro)
- LinkedInClient: LinkedIn Voyager API wrapper (publish, metrics, comments)
- PhotoLibrary: Photo indexing and search for post personalization
- TwitterClient: X/Twitter API v2 for trend monitoring
"""

from src.tools.claude_client import ClaudeClient
from src.tools.perplexity import PerplexityClient
from src.tools.arxiv import ArxivClient, ArxivPaper
from src.tools.nano_banana import NanoBananaClient
from src.tools.linkedin_client import LinkedInClient
from src.tools.photo_library import PhotoLibrary, PhotoMetadata
from src.tools.twitter import TwitterClient

__all__ = [
    "ClaudeClient",
    "PerplexityClient",
    "ArxivClient",
    "ArxivPaper",
    "NanoBananaClient",
    "LinkedInClient",
    "PhotoLibrary",
    "PhotoMetadata",
    "TwitterClient",
]
