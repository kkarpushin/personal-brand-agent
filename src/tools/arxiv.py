"""
Async ArXiv paper search client.

Wraps the synchronous ``arxiv`` Python library with ``asyncio.to_thread``
so that it integrates seamlessly with the async agent pipeline.  The Trend
Scout agent uses this client to discover primary-source research papers in
AI/ML categories.

Default categories align with the architecture specification:
    cs.AI, cs.LG, cs.CL, cs.CV
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import arxiv

logger = logging.getLogger(__name__)


# =========================================================================
# Data model
# =========================================================================


@dataclass
class ArxivPaper:
    """Structured representation of an ArXiv paper.

    Attributes:
        id: ArXiv paper ID (e.g. ``"2401.12345"``).
        title: Paper title.
        summary: Abstract / summary text.
        authors: List of author names.
        published: ISO-format publication date string.
        pdf_url: Direct URL to the PDF.
        categories: List of ArXiv category tags.
    """

    id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    pdf_url: str
    categories: List[str] = field(default_factory=list)


# =========================================================================
# Client
# =========================================================================


class ArxivClient:
    """Async ArXiv paper search and retrieval client.

    Args:
        categories: ArXiv category filters for search.  Defaults to
            ``["cs.AI", "cs.LG", "cs.CL", "cs.CV"]``.

    Usage::

        client = ArxivClient()
        papers = await client.search_papers("transformer attention", max_results=5)
        paper  = await client.get_paper_details("2401.12345")
    """

    def __init__(self, categories: Optional[List[str]] = None) -> None:
        self.categories: List[str] = categories or [
            "cs.AI",
            "cs.LG",
            "cs.CL",
            "cs.CV",
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_papers(
        self,
        query: str = "artificial intelligence",
        max_results: int = 10,
        sort_by: str = "submittedDate",
    ) -> List[ArxivPaper]:
        """Search ArXiv for recent papers matching a query.

        Args:
            query: Free-text search query.
            max_results: Maximum number of results to return.
            sort_by: Sort criterion -- ``"submittedDate"`` (default),
                ``"relevance"``, or ``"lastUpdatedDate"``.

        Returns:
            List of :class:`ArxivPaper` results sorted according to
            ``sort_by``.
        """
        # Build category filter
        category_filter = " OR ".join(
            f"cat:{cat}" for cat in self.categories
        )
        full_query = f"({query}) AND ({category_filter})"

        # Map sort_by string to arxiv.SortCriterion enum
        sort_criterion_map = {
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_criterion_map.get(
            sort_by, arxiv.SortCriterion.SubmittedDate
        )

        def _search() -> List[ArxivPaper]:
            client = arxiv.Client()
            search = arxiv.Search(
                query=full_query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=arxiv.SortOrder.Descending,
            )
            papers: List[ArxivPaper] = []
            for result in client.results(search):
                papers.append(self._result_to_paper(result))
            return papers

        papers = await asyncio.to_thread(_search)

        logger.info(
            "ArXiv search: query=%r, results=%d",
            query,
            len(papers),
        )
        return papers

    # ------------------------------------------------------------------
    # Single paper lookup
    # ------------------------------------------------------------------

    async def get_paper_details(self, paper_id: str) -> Optional[ArxivPaper]:
        """Retrieve details for a specific ArXiv paper by its ID.

        Args:
            paper_id: ArXiv paper ID (e.g. ``"2401.12345"`` or the full
                URL ``"http://arxiv.org/abs/2401.12345v1"``).

        Returns:
            :class:`ArxivPaper` if found, or ``None`` if the ID does not
            resolve to a valid paper.
        """

        def _get_paper() -> Optional[ArxivPaper]:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            results = list(client.results(search))
            if not results:
                return None
            return self._result_to_paper(results[0])

        paper = await asyncio.to_thread(_get_paper)

        if paper is None:
            logger.warning("ArXiv paper not found: %s", paper_id)
        else:
            logger.debug("ArXiv paper retrieved: %s -- %s", paper_id, paper.title)

        return paper

    # ------------------------------------------------------------------
    # Trending / recent papers in configured categories
    # ------------------------------------------------------------------

    async def get_recent_papers(
        self,
        max_results: int = 20,
    ) -> List[ArxivPaper]:
        """Get the most recent papers across configured categories.

        This is a convenience method equivalent to calling
        :meth:`search_papers` with a broad AI query and sorted by
        submission date.

        Args:
            max_results: Maximum number of results.

        Returns:
            List of :class:`ArxivPaper` sorted newest-first.
        """
        return await self.search_papers(
            query="artificial intelligence OR machine learning OR large language model",
            max_results=max_results,
            sort_by="submittedDate",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_paper(result: Any) -> ArxivPaper:
        """Convert an ``arxiv.Result`` to an :class:`ArxivPaper`.

        Args:
            result: An ``arxiv.Result`` instance from the library.

        Returns:
            Corresponding :class:`ArxivPaper`.
        """
        return ArxivPaper(
            id=result.entry_id.split("/abs/")[-1] if "/abs/" in result.entry_id else result.entry_id,
            title=result.title.strip(),
            summary=result.summary.strip(),
            authors=[author.name for author in result.authors],
            published=result.published.isoformat() if result.published else "",
            pdf_url=result.pdf_url or "",
            categories=list(result.categories) if result.categories else [],
        )
