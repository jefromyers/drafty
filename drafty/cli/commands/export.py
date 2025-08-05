"""Export command implementation."""

import json
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console

from drafty.core.workspace import Workspace
from drafty.services.export import ExportService

console = Console()


def export_article(
    ctx,
    formats: List[str],
    output: Optional[str],
    template: Optional[str]
) -> List[Path]:
    """Export article to various formats."""
    # Load workspace and config
    workspace_path = Path.cwd()
    
    # Check if we're in a workspace
    if not (workspace_path / "article.json").exists():
        raise click.ClickException("Not in a Drafty workspace. Run 'drafty new' first.")
    
    workspace = Workspace.load(workspace_path)
    config = workspace.get_config()
    
    # Load current draft
    draft_content = workspace.get_current_draft()
    if not draft_content:
        raise click.ClickException("No draft found. Run 'drafty draft' first.")
    
    # Load draft data if available
    draft_data = None
    draft_json_file = workspace.drafts_dir / "draft.json"
    if draft_json_file.exists():
        with open(draft_json_file) as f:
            draft_data = json.load(f)
    
    # Create export service
    export_service = ExportService(config)
    
    # Determine output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = workspace.exports_dir
    
    exported_files = []
    
    for format_type in formats:
        format_lower = format_type.lower()
        
        try:
            if format_lower == "markdown" or format_lower == "md":
                # Export as clean markdown
                content = export_service.export_markdown(
                    draft_content,
                    include_metadata=True,
                    clean=True
                )
                filename = f"{config.meta.slug}.md"
                
            elif format_lower == "html":
                # Export as HTML
                content = export_service.export_html(
                    draft_content,
                    template_name=template,
                    include_styles=True
                )
                filename = f"{config.meta.slug}.html"
                
            elif format_lower == "text" or format_lower == "txt":
                # Export as plain text
                content = export_service.export_text(draft_content)
                filename = f"{config.meta.slug}.txt"
                
            elif format_lower == "json":
                # Export as JSON
                if draft_data:
                    content = export_service.export_json(
                        draft_data,
                        include_config=False
                    )
                else:
                    content = export_service.export_json(
                        {"content": draft_content},
                        include_config=False
                    )
                filename = f"{config.meta.slug}.json"
                
            else:
                console.print(f"[yellow]Warning: Unknown format '{format_type}'[/yellow]")
                continue
            
            # Save the exported file
            output_path = output_dir / filename
            output_path.write_text(content)
            exported_files.append(output_path)
            
            console.print(f"[green]✓[/green] Exported to: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error exporting {format_type}: {e}[/red]")
    
    # Update workspace status if successful
    if exported_files:
        config.meta.status = "published"
        workspace.save_config(config)
    
    return exported_files