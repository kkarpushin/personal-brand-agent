"""
Async Claude API client for agent LLM calls.

Uses the ``anthropic`` Python SDK (``AsyncAnthropic``) to interact with
Anthropic's Messages API.  This is the standard LLM client used by all
pipeline agents; it is **not** the Claude Code CLI wrapper (that lives in
``src/meta_agent/``).

Key features:
    - Automatic retry with exponential backoff via ``@with_retry``
    - Token usage tracking (input + output)
    - Structured JSON generation with markdown-fence stripping
    - Configurable model, temperature, and max_tokens

Fail-fast philosophy: if all retry attempts are exhausted the original
``anthropic`` exception propagates wrapped in ``RetryExhaustedError``.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from src.utils import with_retry

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Async Claude API client for agent LLM calls.

    Args:
        api_key: Anthropic API key.  Falls back to the
            ``ANTHROPIC_API_KEY`` environment variable.
        model: Model identifier.  Defaults to ``claude-opus-4-5-20251101``.

    Raises:
        KeyError: If no API key is provided and the environment variable
            is missing.

    Usage::

        client = ClaudeClient()
        answer = await client.generate("Explain LangGraph in one paragraph.")
        data   = await client.generate_structured("Return {score: int, reason: str}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5-20251101",
    ) -> None:
        self.client = AsyncAnthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
        )
        self.model = model
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Generate a plain-text response from the model.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 -- 1.0).

        Returns:
            The model's text response.
        """
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        # Track token usage
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens

        logger.debug(
            "Claude generate: in=%d out=%d tokens",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return response.content[0].text

    # ------------------------------------------------------------------
    # Structured (JSON) generation
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def generate_structured(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Generate a structured JSON response.

        Appends an explicit JSON instruction to the prompt and uses a low
        temperature (0.3) to encourage deterministic output.  Markdown
        code fences (` ```json ... ``` `) are stripped before parsing.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            max_tokens: Maximum tokens in the response.

        Returns:
            Parsed JSON object as a Python dict.

        Raises:
            json.JSONDecodeError: If the model returns invalid JSON after
                stripping.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Return ONLY valid JSON, no markdown, no explanation."
        )

        messages: List[Dict[str, str]] = [{"role": "user", "content": json_prompt}]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        # Track token usage
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens

        logger.debug(
            "Claude generate_structured: in=%d out=%d tokens",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        text = response.content[0].text

        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (e.g. ```json)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            else:
                cleaned = cleaned[3:]
            # Remove closing fence
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]

        return json.loads(cleaned.strip())

    # ------------------------------------------------------------------
    # Multi-turn conversation
    # ------------------------------------------------------------------

    @with_retry(max_attempts=3, retryable_exceptions=(Exception,))
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from a full conversation history.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            system: Optional system prompt.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            The model's text response.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens

        return response.content[0].text

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    @property
    def usage_stats(self) -> Dict[str, int]:
        """Cumulative token usage since this client was instantiated.

        Returns:
            Dict with ``input_tokens`` and ``output_tokens`` keys.
        """
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }

    def reset_usage(self) -> None:
        """Reset cumulative token counters to zero."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0


# ---------------------------------------------------------------------------
# MODULE-LEVEL FACTORY
# ---------------------------------------------------------------------------


def get_claude(
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-5-20251101",
) -> ClaudeClient:
    """Create and return a new :class:`ClaudeClient` instance.

    Convenience factory used by meta-agent components that accept an
    optional ``claude_client`` constructor argument with a default of
    ``get_claude()``.

    Args:
        api_key: Anthropic API key.  Falls back to the
            ``ANTHROPIC_API_KEY`` environment variable.
        model: Model identifier.  Defaults to ``claude-opus-4-5-20251101``.

    Returns:
        A new ``ClaudeClient`` instance.
    """
    return ClaudeClient(api_key=api_key, model=model)
