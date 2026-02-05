"""
Image generation client via Laozhang.ai (Nano Banana Pro).

Uses the ``gemini-3-pro-image-preview`` model exposed through the
Laozhang.ai compatible API to generate professional images for LinkedIn
posts.  The Visual Creator agent delegates all image generation to this
client.

Fail-fast philosophy: transient HTTP errors are retried with exponential
backoff; after all attempts are exhausted an ``ImageGenerationError`` is
raised.
"""

import base64
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from src.exceptions import ImageGenerationError
from src.utils import with_retry

logger = logging.getLogger(__name__)


class NanoBananaClient:
    """Image generation client via Laozhang.ai Nano Banana Pro.

    Uses the ``gemini-3-pro-image-preview`` model behind the Laozhang.ai
    OpenAI-compatible ``/v1/images/generations`` endpoint.

    Args:
        api_key: Laozhang API key.  Falls back to the
            ``LAOZHANG_API_KEY`` environment variable.

    Usage::

        client = NanoBananaClient()
        result = await client.generate_image(
            prompt="Professional tech office with holographic displays",
            size="1200x627",
            style="professional",
        )
        print(result["url"])
    """

    BASE_URL: str = "https://api.laozhang.ai/v1"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or os.environ.get("LAOZHANG_API_KEY", "")

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    @with_retry(
        max_attempts=3,
        retryable_exceptions=(httpx.HTTPError, httpx.TimeoutException),
    )
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        style: str = "professional",
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions as ``"WxH"`` (e.g. ``"1024x1024"``,
                ``"1200x627"`` for LinkedIn recommended).
            style: Visual style hint appended to the prompt
                (``"professional"``, ``"modern"``, ``"minimal"``, etc.).

        Returns:
            Dict with keys:
                - ``url`` -- Public URL of the generated image.
                - ``prompt_used`` -- The prompt that was sent.
                - ``model`` -- Model identifier.
                - ``size`` -- Requested image size.

        Raises:
            ImageGenerationError: If the API returns a non-2xx status
                after all retry attempts.
        """
        # Incorporate style into the prompt for better results
        styled_prompt = f"{prompt}. Style: {style}" if style else prompt

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/images/generations",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gemini-3-pro-image-preview",
                    "prompt": styled_prompt,
                    "size": size,
                    "n": 1,
                },
            )

            if response.status_code != 200:
                error_text = response.text
                raise ImageGenerationError(
                    f"Nano Banana API error {response.status_code}: {error_text}"
                )

            data = response.json()
            item = data["data"][0]

            # API may return a URL or base64-encoded image
            if "url" in item:
                image_url = item["url"]
            elif "b64_json" in item:
                # Save base64 image to disk and return local path
                image_bytes = base64.b64decode(item["b64_json"])
                images_dir = Path("data/images")
                images_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{uuid.uuid4().hex}.jpg"
                image_path = images_dir / filename
                image_path.write_bytes(image_bytes)
                image_url = str(image_path)
                logger.info("NanoBanana image saved to %s", image_url)
            else:
                raise ImageGenerationError(
                    f"Unexpected API response format: {list(item.keys())}"
                )

            logger.info(
                "NanoBanana image generated: size=%s, prompt_len=%d",
                size,
                len(prompt),
            )

            return {
                "url": image_url,
                "prompt_used": styled_prompt,
                "model": "nano-banana-pro",
                "size": size,
            }

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    async def generate_multiple(
        self,
        prompts: list[str],
        size: str = "1024x1024",
        style: str = "professional",
    ) -> list[Dict[str, Any]]:
        """Generate images for multiple prompts sequentially.

        Args:
            prompts: List of text prompts.
            size: Image dimensions.
            style: Visual style hint.

        Returns:
            List of result dicts from :meth:`generate_image`.  Failed
            generations are omitted (logged as warnings).
        """
        results: list[Dict[str, Any]] = []
        for prompt in prompts:
            try:
                result = await self.generate_image(
                    prompt=prompt, size=size, style=style
                )
                results.append(result)
            except ImageGenerationError:
                logger.warning(
                    "NanoBanana generation failed for prompt: %s",
                    prompt[:80],
                    exc_info=True,
                )
                continue
        return results
