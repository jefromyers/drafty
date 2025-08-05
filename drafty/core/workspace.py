"""Workspace management for Drafty articles."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from drafty.core.config import ArticleConfig, ArticleStatus, MetaConfig, ContentConfig


class Workspace:
    """Manages the file structure and state for an article workspace."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.config_file = self.base_path / "article.json"
        self.research_dir = self.base_path / "research"
        self.drafts_dir = self.base_path / "drafts"
        self.exports_dir = self.base_path / "exports"
        self.assets_dir = self.base_path / "assets"
        self.metadata_dir = self.base_path / ".drafty"

    @classmethod
    def create(
        cls,
        slug: str,
        base_dir: Path = Path.cwd(),
        template: Optional[str] = None,
        topic: Optional[str] = None,
        audience: Optional[str] = None,
    ) -> "Workspace":
        """Create a new workspace for an article."""
        workspace_path = base_dir / slug
        if workspace_path.exists():
            raise ValueError(f"Workspace already exists: {workspace_path}")

        workspace = cls(workspace_path)
        workspace._initialize_structure()

        # Create initial config
        meta = MetaConfig(
            slug=slug,
            created=datetime.now().isoformat(),
            status=ArticleStatus.DRAFT,
        )

        content = ContentConfig(
            topic=topic or f"Article about {slug.replace('-', ' ')}",
            audience=audience or "general audience",
        )

        config = ArticleConfig(meta=meta, content=content)

        # Apply template if provided
        if template:
            workspace._apply_template(template, config)

        workspace.save_config(config)
        workspace._create_readme()

        return workspace

    @classmethod
    def load(cls, path: Path) -> "Workspace":
        """Load an existing workspace."""
        workspace = cls(path)
        if not workspace.config_file.exists():
            raise ValueError(f"No article.json found in {path}")
        return workspace

    def _initialize_structure(self) -> None:
        """Create the directory structure for the workspace."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.research_dir.mkdir(exist_ok=True)
        self.drafts_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Create initial files
        (self.research_dir / "sources.json").write_text("[]")
        (self.research_dir / "notes.md").write_text("# Research Notes\n\n")
        (self.drafts_dir / "current.md").write_text("# Draft\n\n")
        (self.metadata_dir / "state.json").write_text(
            json.dumps({"version": "1.0", "history": []}, indent=2)
        )

    def _apply_template(self, template_name: str, config: ArticleConfig) -> None:
        """Apply a template to the configuration."""
        templates = {
            "blog": {
                "structure.outline_style": "guide",
                "content.tone": ["conversational", "friendly"],
                "quality.readability_target": 8,
            },
            "technical": {
                "structure.outline_style": "howto",
                "content.tone": ["technical", "professional"],
                "quality.readability_target": 12,
            },
            "comparison": {
                "structure.outline_style": "comparison",
                "content.intent": "inform",
                "structure.fixed_sections": [
                    "Introduction",
                    "Criteria",
                    "Comparison Table",
                    "Detailed Analysis",
                    "Recommendation",
                ],
            },
            "tutorial": {
                "structure.outline_style": "howto",
                "content.intent": "educate",
                "structure.fixed_sections": [
                    "Prerequisites",
                    "Overview",
                    "Step-by-Step Guide",
                    "Troubleshooting",
                    "Conclusion",
                ],
            },
        }

        if template_name in templates:
            template_data = templates[template_name]
            for key, value in template_data.items():
                parts = key.split(".")
                if len(parts) == 2:
                    section, field = parts
                    section_obj = getattr(config, section)
                    setattr(section_obj, field, value)

    def _create_readme(self) -> None:
        """Create a README file for the workspace."""
        readme_content = f"""# Article Workspace

## Structure

- `article.json` - Main configuration file
- `research/` - Research materials and sources
  - `sources.json` - Collected sources
  - `notes.md` - Manual research notes
- `drafts/` - Draft versions
  - `current.md` - Current working draft
  - `v*.md` - Version history
- `exports/` - Final exported files
- `assets/` - Images and other assets
- `.drafty/` - Internal metadata

## Commands

```bash
# Research phase
drafty research

# Generate outline
drafty outline

# Create draft
drafty draft

# Edit and refine
drafty edit

# Add links
drafty link

# Export final version
drafty export
```

## Configuration

Edit `article.json` to customize settings.
"""
        (self.base_path / "README.md").write_text(readme_content)

    def get_config(self) -> ArticleConfig:
        """Load the configuration from the workspace."""
        return ArticleConfig.from_file(self.config_file)

    def save_config(self, config: ArticleConfig) -> None:
        """Save the configuration to the workspace."""
        config.meta.updated = datetime.now().isoformat()
        config.to_file(self.config_file)
        self._update_history("config_updated")

    def get_current_draft(self) -> str:
        """Get the current draft content."""
        current_draft = self.drafts_dir / "current.md"
        if current_draft.exists():
            return current_draft.read_text()
        return ""

    def save_draft(self, content: str, create_version: bool = True) -> Path:
        """Save draft content."""
        current_draft = self.drafts_dir / "current.md"

        if create_version and current_draft.exists():
            # Create versioned backup
            version_num = len(list(self.drafts_dir.glob("v*.md"))) + 1
            version_file = self.drafts_dir / f"v{version_num:03d}.md"
            shutil.copy2(current_draft, version_file)

        current_draft.write_text(content)
        self._update_history("draft_saved")
        return current_draft

    def get_research_sources(self) -> List[Dict[str, Any]]:
        """Get all research sources."""
        sources_file = self.research_dir / "sources.json"
        if sources_file.exists():
            return json.loads(sources_file.read_text())
        return []

    def add_research_source(self, source: Dict[str, Any]) -> None:
        """Add a research source."""
        sources = self.get_research_sources()
        source["added_at"] = datetime.now().isoformat()
        sources.append(source)
        (self.research_dir / "sources.json").write_text(json.dumps(sources, indent=2))
        self._update_history("source_added")

    def save_export(self, format: str, content: str) -> Path:
        """Save an exported file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.base_path.name}_{timestamp}.{format}"
        export_path = self.exports_dir / filename
        export_path.write_text(content)
        self._update_history(f"exported_{format}")
        return export_path

    def get_state(self) -> Dict[str, Any]:
        """Get the workspace state."""
        state_file = self.metadata_dir / "state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {"version": "1.0", "history": []}

    def _update_history(self, action: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Update the workspace history."""
        state = self.get_state()
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data or {},
        }
        state["history"].append(history_entry)

        # Keep only last 100 history entries
        if len(state["history"]) > 100:
            state["history"] = state["history"][-100:]

        (self.metadata_dir / "state.json").write_text(json.dumps(state, indent=2))

    def get_draft_versions(self) -> List[Path]:
        """Get all draft version files."""
        return sorted(self.drafts_dir.glob("v*.md"))

    def restore_draft_version(self, version_path: Path) -> None:
        """Restore a specific draft version."""
        if not version_path.exists():
            raise ValueError(f"Version file does not exist: {version_path}")

        # Backup current before restoring
        self.save_draft(self.get_current_draft(), create_version=True)

        # Restore the version
        current_draft = self.drafts_dir / "current.md"
        shutil.copy2(version_path, current_draft)
        self._update_history("version_restored", {"version": version_path.name})

    def cleanup_old_versions(self, keep_last: int = 10) -> int:
        """Clean up old draft versions, keeping the most recent ones."""
        versions = self.get_draft_versions()
        if len(versions) <= keep_last:
            return 0

        to_delete = versions[:-keep_last]
        for version_file in to_delete:
            version_file.unlink()

        self._update_history("versions_cleaned", {"deleted": len(to_delete)})
        return len(to_delete)

    def get_workspace_stats(self) -> Dict[str, Any]:
        """Get statistics about the workspace."""
        config = self.get_config()
        current_draft = self.get_current_draft()
        word_count = len(current_draft.split())

        return {
            "slug": config.meta.slug,
            "status": config.meta.status,
            "created": config.meta.created,
            "updated": config.meta.updated,
            "word_count": word_count,
            "draft_versions": len(self.get_draft_versions()),
            "research_sources": len(self.get_research_sources()),
            "exports": len(list(self.exports_dir.glob("*"))),
        }

    def archive(self, archive_dir: Path) -> Path:
        """Archive the workspace to a specified directory."""
        archive_name = f"{self.base_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_path = archive_dir / f"{archive_name}.tar.gz"

        import tarfile

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.base_path, arcname=self.base_path.name)

        return archive_path