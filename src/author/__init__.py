"""
Author profile module for the LinkedIn Super Agent system.

Provides author voice profiling, profile creation from existing posts,
and profile import utilities.

- ``AuthorVoiceProfile``: Data model for the author's voice and style.
- ``AuthorProfileAgent``: Agent that creates and maintains profiles.
- ``ProfileImporter``: Imports posts from LinkedIn or JSON files.
"""

from src.author.models import AuthorVoiceProfile
from src.author.author_profile_agent import AuthorProfileAgent
from src.author.profile_importer import ProfileImporter

__all__ = [
    "AuthorVoiceProfile",
    "AuthorProfileAgent",
    "ProfileImporter",
]
