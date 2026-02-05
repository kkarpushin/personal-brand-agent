"""
Entry point: run the content pipeline with Telegram human-in-the-loop approval.

Usage::

    python run.py
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


async def main() -> None:
    from src.agents.orchestrator import run_pipeline
    from src.tools.linkedin_client import LinkedInClient
    from src.ui.telegram_bot import TelegramBot, TelegramNotifier

    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    # Shared event + decision container for the approval flow
    decision_event = asyncio.Event()
    decision: dict = {}  # {"action": "approve"|"reject", "run_id": str}

    async def on_approve(run_id: str) -> None:
        decision["action"] = "approve"
        decision["run_id"] = run_id
        decision_event.set()

    async def on_reject(run_id: str) -> None:
        decision["action"] = "reject"
        decision["run_id"] = run_id
        decision_event.set()

    notifier = TelegramNotifier(bot_token, chat_id)
    bot = TelegramBot(bot_token, chat_id, on_approve=on_approve, on_reject=on_reject)

    # Start bot polling so it can receive button presses
    await bot.start()

    try:
        await notifier.send("Pipeline starting...")

        # ---- Run the pipeline ------------------------------------------------
        try:
            result = await run_pipeline()
        except Exception:
            logger.exception("Pipeline failed")
            await notifier.send("Pipeline FAILED. Check logs.")
            return

        run_id = result["run_id"]
        status = result["status"]
        content = result.get("content")
        stats = result.get("statistics", {})

        if status != "success" or content is None:
            error_msg = f"Pipeline finished with status={status}, run_id={run_id}"
            logger.error(error_msg)
            await notifier.send(error_msg)
            return

        # ---- Extract post data -----------------------------------------------
        post_text = content.get("post_text", "")
        visual = content.get("visual")
        image_path = None
        if visual is not None:
            # Try file_path first, then fall back to url (may be a local path
            # when NanoBanana returns base64 and saves to disk)
            fp = (
                getattr(visual, "file_path", None)
                if hasattr(visual, "file_path")
                else visual.get("file_path") if isinstance(visual, dict) else None
            )
            if not fp:
                url = (
                    getattr(visual, "url", None)
                    if hasattr(visual, "url")
                    else visual.get("url") if isinstance(visual, dict) else None
                )
                if url and not url.startswith("http"):
                    fp = url  # local file path stored in url field
            image_path = fp
        qc_score = stats.get("qc_score")
        content_type = stats.get("content_type")

        logger.info(
            "Pipeline OK: run_id=%s qc=%.1f type=%s",
            run_id,
            qc_score or 0,
            content_type,
        )

        # ---- Send preview to Telegram ----------------------------------------
        await notifier.send_post_preview(
            run_id=run_id,
            post_text=post_text,
            image_path=image_path,
            qc_score=qc_score,
            content_type=content_type,
        )
        logger.info("Preview sent to Telegram. Waiting for approval...")

        # ---- Wait for human decision -----------------------------------------
        await decision_event.wait()

        if decision.get("action") == "approve":
            logger.info("Post APPROVED. Publishing to LinkedIn...")
            await notifier.send("Publishing to LinkedIn...")
            try:
                li = LinkedInClient()
                pub_result = await li.publish_post(post_text, image_path=image_path)
                post_id = pub_result.get("post_id", "unknown")
                await notifier.send(f"Published! post_id={post_id}")
                logger.info("Published: %s", post_id)
            except Exception:
                logger.exception("LinkedIn publish failed")
                await notifier.send("LinkedIn publish FAILED. Check logs.")
        else:
            logger.info("Post REJECTED by human.")
            await notifier.send("Post rejected. Discarding.")

    finally:
        await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
