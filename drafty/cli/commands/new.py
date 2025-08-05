"""Command to create a new article workspace."""

from pathlib import Path
from typing import Optional

from drafty.core.workspace import Workspace


def create_workspace(
    slug: str,
    template: Optional[str] = None,
    topic: Optional[str] = None,
    audience: Optional[str] = None,
) -> Path:
    """Create a new article workspace."""
    workspace = Workspace.create(
        slug=slug,
        template=template,
        topic=topic,
        audience=audience,
    )
    return workspace.base_path