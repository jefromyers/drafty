"""Export command implementation."""

from pathlib import Path
from typing import List, Optional


def export_article(
    ctx,
    formats: List[str],
    output: Optional[str],
    template: Optional[str]
) -> List[Path]:
    """Export article to various formats."""
    # TODO: Implement export logic
    exported = []
    for fmt in formats:
        exported.append(Path(f"article.{fmt}"))
    return exported