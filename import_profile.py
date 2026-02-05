"""
Import LinkedIn posts and build an author voice profile.

Usage::

    # Import posts + build profile in one go:
    python import_profile.py https://www.linkedin.com/in/username "Author Name"

    # Import posts only (skip Claude profile analysis):
    python import_profile.py https://www.linkedin.com/in/username --posts-only

    # Import from a local JSON file instead of LinkedIn:
    python import_profile.py --json data/posts.json "Author Name"

    # Control how many posts to fetch (default 50):
    python import_profile.py https://www.linkedin.com/in/username "Author Name" --limit 30
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("import_profile")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import LinkedIn posts and build author voice profile"
    )
    parser.add_argument(
        "profile_url",
        nargs="?",
        help="LinkedIn profile URL (e.g. https://www.linkedin.com/in/username)",
    )
    parser.add_argument(
        "author_name",
        nargs="?",
        help="Author display name for the voice profile",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Import from a local JSON file instead of LinkedIn",
    )
    parser.add_argument(
        "--posts-only",
        action="store_true",
        help="Only import posts to DB, skip voice profile creation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max posts to fetch from LinkedIn (default: 50)",
    )
    args = parser.parse_args()

    if not args.json and not args.profile_url:
        parser.error("Provide a LinkedIn profile URL or --json FILE")

    if not args.posts_only and not args.author_name:
        parser.error("Provide author_name or use --posts-only")

    # --- Init database ------------------------------------------------
    from src.database import get_db

    db = await get_db()
    logger.info("Database connected")

    # --- Import posts -------------------------------------------------
    if args.json:
        from src.author.profile_importer import ProfileImporter

        importer = ProfileImporter(db=db)
        posts = await importer.import_from_json(args.json)
        logger.info("Imported %d posts from JSON file", len(posts))
    else:
        from src.tools.linkedin_client import LinkedInClient
        from src.author.profile_importer import ProfileImporter

        li = LinkedInClient()
        importer = ProfileImporter(linkedin_client=li, db=db)
        posts = await importer.import_from_linkedin(args.profile_url, limit=args.limit)
        logger.info("Imported %d posts from LinkedIn", len(posts))

    if not posts:
        logger.error("No posts imported. Nothing to do.")
        sys.exit(1)

    # --- Build voice profile ------------------------------------------
    if args.posts_only:
        logger.info("--posts-only: skipping voice profile creation")
        logger.info("Done. %d posts saved to database.", len(posts))
        return

    from src.author.author_profile_agent import AuthorProfileAgent

    agent = AuthorProfileAgent(db=db)
    profile = await agent.create_profile_from_posts(
        author_name=args.author_name,
        posts=posts,
    )

    # Save profile to database
    await agent.save_profile(profile)

    logger.info(
        "Voice profile created and saved for '%s' "
        "(%d posts analyzed, formality=%.2f, %d characteristic phrases)",
        profile.author_name,
        profile.posts_analyzed,
        profile.formality_level,
        len(profile.characteristic_phrases),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
