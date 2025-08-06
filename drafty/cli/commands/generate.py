"""Generate command for automated article creation workflow."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from drafty.core.workspace import Workspace
from drafty.core.config import ArticleConfig
from drafty.services.research import ResearchService
from drafty.services.outline import OutlineService
from drafty.services.draft import DraftService
from drafty.services.edit import EditService, EditType
from drafty.services.export import ExportService

console = Console()


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in config file: {e}")


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration with CLI overrides."""
    config = base_config.copy()
    
    # Simple merge - CLI overrides take precedence
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    
    return config


def prepare_edit_types(edit_types: List[str]) -> List[Dict[str, Any]]:
    """Prepare edit configurations from edit type strings."""
    edits = []
    
    for edit_type in edit_types:
        if edit_type.lower() == "all":
            edits.extend([
                {"type": EditType.CLARITY},
                {"type": EditType.READABILITY, "target_grade_level": 10},
                {"type": EditType.GRAMMAR},
                {"type": EditType.STYLE}
            ])
        elif edit_type.lower() == "clarity":
            edits.append({"type": EditType.CLARITY})
        elif edit_type.lower() == "readability":
            edits.append({"type": EditType.READABILITY, "target_grade_level": 10})
        elif edit_type.lower() == "seo":
            edits.append({"type": EditType.SEO})
        elif edit_type.lower() == "grammar":
            edits.append({"type": EditType.GRAMMAR})
        elif edit_type.lower() == "style":
            edits.append({"type": EditType.STYLE})
    
    return edits


async def run_workflow(
    config: Dict[str, Any],
    workspace: Workspace,
    progress: Progress,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """Run the complete article generation workflow."""
    results = {
        "workspace": str(workspace.base_path),
        "research_sources": 0,
        "outline_sections": 0,
        "draft_words": 0,
        "exports": []
    }
    
    article_config = workspace.get_config()
    
    # Step 1: Research
    if not config.get("skip_research", False):
        task = progress.add_task("Researching...", total=100)
        try:
            research_service = ResearchService(article_config)
            research_results = await research_service.conduct_research(
                max_sources=config.get("max_sources", 10),
                provider_name=config.get("provider")
            )
            
            # Save research results
            if research_results.get("topic_analysis"):
                analysis_file = workspace.research_dir / "topic_analysis.json"
                analysis_file.write_text(json.dumps(research_results["topic_analysis"], indent=2))
            
            if research_results.get("search_results"):
                sources_file = workspace.research_dir / "sources.json"
                sources = []
                for result in research_results["search_results"]:
                    sources.append({
                        "url": result.get("url"),
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                        "query": result.get("query"),
                        "relevance": result.get("relevance", 0.5),
                        "added_at": result.get("added_at", "")
                    })
                sources_file.write_text(json.dumps(sources, indent=2))
                results["research_sources"] = len(sources)
            
            progress.update(task, completed=100)
            if verbose:
                console.print(f"[green]✓[/green] Research complete: {results['research_sources']} sources")
        except Exception as e:
            progress.update(task, completed=100, description=f"Research failed: {e}")
            if not config.get("continue_on_error", True):
                raise
    
    # Step 2: Outline
    task = progress.add_task("Generating outline...", total=100)
    try:
        outline_service = OutlineService(article_config)
        
        # Load research data
        research_data = {}
        topic_file = workspace.research_dir / "topic_analysis.json"
        if topic_file.exists():
            with open(topic_file) as f:
                research_data["topic_analysis"] = json.load(f)
        
        sources_file = workspace.research_dir / "sources.json"
        if sources_file.exists():
            with open(sources_file) as f:
                research_data["search_results"] = json.load(f)
        
        outline = await outline_service.generate_outline(
            research_data=research_data,
            provider_name=config.get("provider"),
            sections=config.get("sections", 5),
            style=config.get("style", "guide")
        )
        
        # Save outline
        outline_file = workspace.drafts_dir / "outline.json"
        outline_file.write_text(json.dumps(outline.to_dict(), indent=2))
        results["outline_sections"] = len(outline.sections)
        
        progress.update(task, completed=100)
        if verbose:
            console.print(f"[green]✓[/green] Outline generated: {results['outline_sections']} sections")
    except Exception as e:
        progress.update(task, completed=100, description=f"Outline failed: {e}")
        raise
    
    # Step 3: Draft
    task = progress.add_task("Writing draft...", total=100)
    try:
        draft_service = DraftService(article_config)
        
        draft = await draft_service.generate_draft(
            outline=outline,
            research_data=research_data,
            provider_name=config.get("provider")
        )
        
        # Save draft
        draft_content = draft.to_markdown()
        workspace.save_draft(draft_content, create_version=True)
        results["draft_words"] = draft.total_word_count
        
        # Also save JSON version
        draft_json_file = workspace.drafts_dir / "draft.json"
        draft_json_file.write_text(json.dumps(draft.to_dict(), indent=2))
        
        progress.update(task, completed=100)
        if verbose:
            console.print(f"[green]✓[/green] Draft created: {results['draft_words']} words")
    except Exception as e:
        progress.update(task, completed=100, description=f"Draft failed: {e}")
        raise
    
    # Step 4: Edit
    if not config.get("skip_edit", False):
        task = progress.add_task("Editing content...", total=100)
        try:
            edit_service = EditService(article_config)
            
            # Get current draft
            current_draft = workspace.get_current_draft()
            
            # Prepare edits
            edit_types = config.get("edit_types", ["all"])
            edits = prepare_edit_types(edit_types)
            
            if edits:
                edited_content = await edit_service.batch_edit(
                    current_draft,
                    edits,
                    provider_name=config.get("provider")
                )
                
                # Save edited version
                workspace.save_draft(edited_content, create_version=True)
                
                progress.update(task, completed=100)
                if verbose:
                    console.print(f"[green]✓[/green] Content edited with {len(edits)} improvements")
            else:
                progress.update(task, completed=100, description="No edits specified")
        except Exception as e:
            progress.update(task, completed=100, description=f"Edit failed: {e}")
            if not config.get("continue_on_error", True):
                raise
    
    # Step 5: Export
    task = progress.add_task("Exporting files...", total=100)
    try:
        export_service = ExportService(article_config)
        
        # Get final draft
        final_draft = workspace.get_current_draft()
        
        # Export to specified formats
        export_formats = config.get("export_formats", ["markdown", "html"])
        output_dir = Path(config.get("output_dir", workspace.exports_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for format_type in export_formats:
            if format_type.lower() in ["markdown", "md"]:
                content = export_service.export_markdown(final_draft, clean=True)
                filename = f"{article_config.meta.slug}.md"
            elif format_type.lower() == "html":
                content = export_service.export_html(final_draft, include_styles=True)
                filename = f"{article_config.meta.slug}.html"
            elif format_type.lower() == "json":
                draft_data = {"content": final_draft, "config": article_config.model_dump()}
                content = export_service.export_json(draft_data)
                filename = f"{article_config.meta.slug}.json"
            elif format_type.lower() in ["text", "txt"]:
                content = export_service.export_text(final_draft)
                filename = f"{article_config.meta.slug}.txt"
            else:
                continue
            
            output_path = output_dir / filename
            output_path.write_text(content)
            results["exports"].append(str(output_path))
        
        progress.update(task, completed=100)
        if verbose:
            console.print(f"[green]✓[/green] Exported {len(results['exports'])} files")
    except Exception as e:
        progress.update(task, completed=100, description=f"Export failed: {e}")
        raise
    
    return True, results


def generate_article(
    topic: Optional[str],
    config_file: Optional[Path],
    audience: Optional[str],
    keywords: Optional[List[str]],
    sections: Optional[int],
    word_count: Optional[int],
    provider: Optional[str],
    style: Optional[str],
    tone: Optional[str],
    edit_types: Optional[List[str]],
    export_formats: Optional[List[str]],
    output_dir: Optional[Path],
    skip_research: bool,
    skip_edit: bool,
    save_config: Optional[Path],
    dry_run: bool,
    verbose: bool,
    use_temp: bool = True
) -> Dict[str, Any]:
    """Generate a complete article with automated workflow."""
    
    # Load config from file if provided
    if config_file:
        config = load_config_from_file(config_file)
        if verbose:
            console.print(f"[cyan]Loaded config from {config_file}[/cyan]")
    else:
        config = {}
    
    # Override with CLI arguments
    cli_overrides = {
        "topic": topic,
        "audience": audience,
        "keywords": keywords,
        "sections": sections,
        "word_count": word_count,
        "provider": provider,
        "style": style,
        "tone": tone,
        "edit_types": edit_types,
        "export_formats": export_formats,
        "output_dir": str(output_dir) if output_dir else None,
        "skip_research": skip_research,
        "skip_edit": skip_edit,
    }
    
    # Merge configurations
    config = merge_configs(config, {k: v for k, v in cli_overrides.items() if v is not None})
    
    # Validate required fields
    if not config.get("topic"):
        raise click.ClickException("Topic is required (via --topic or config file)")
    
    # Set defaults
    config.setdefault("audience", "General audience")
    config.setdefault("sections", 5)
    config.setdefault("word_count", 2000)
    config.setdefault("style", "guide")
    config.setdefault("tone", "professional")
    config.setdefault("edit_types", ["all"])
    config.setdefault("export_formats", ["markdown", "html"])
    
    # Save config if requested
    if save_config:
        save_config.parent.mkdir(parents=True, exist_ok=True)
        with open(save_config, "w") as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]✓[/green] Config saved to {save_config}")
    
    # Dry run - just show what would be done
    if dry_run:
        console.print("\n[yellow]DRY RUN - No actions will be taken[/yellow]\n")
        
        table = Table(title="Workflow Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Topic", config["topic"])
        table.add_row("Audience", config["audience"])
        table.add_row("Keywords", ", ".join(config.get("keywords", [])) if config.get("keywords") else "None")
        table.add_row("Sections", str(config["sections"]))
        table.add_row("Word Count", str(config["word_count"]))
        table.add_row("Provider", config.get("provider", "default"))
        table.add_row("Style", config["style"])
        table.add_row("Tone", config["tone"])
        table.add_row("Edit Types", ", ".join(config["edit_types"]))
        table.add_row("Export Formats", ", ".join(config["export_formats"]))
        table.add_row("Skip Research", "Yes" if config.get("skip_research") else "No")
        table.add_row("Skip Edit", "Yes" if config.get("skip_edit") else "No")
        
        console.print(table)
        
        console.print("\n[cyan]Workflow steps that would be executed:[/cyan]")
        console.print("1. Create workspace")
        if not config.get("skip_research"):
            console.print("2. Conduct research")
        console.print("3. Generate outline")
        console.print("4. Create draft")
        if not config.get("skip_edit"):
            console.print("5. Edit content")
        console.print("6. Export files")
        
        return {"dry_run": True, "config": config}
    
    # Create workspace
    if use_temp:
        temp_dir = tempfile.mkdtemp(prefix="drafty-")
        workspace_path = Path(temp_dir) / "article"
    else:
        workspace_path = Path.cwd() / f"generated-{config['topic'][:30].replace(' ', '-').lower()}"
    
    workspace = Workspace.create(
        slug=workspace_path.name,
        topic=config["topic"],
        audience=config["audience"],
        base_path=workspace_path
    )
    
    # Update workspace config
    article_config = workspace.get_config()
    article_config.content.keywords = config.get("keywords", [])
    article_config.content.word_count.target = config["word_count"]
    article_config.content.tone = config["tone"]
    article_config.content.style = config["style"]
    
    # Set provider
    if config.get("provider"):
        article_config.llm.default = config["provider"]
        # Ensure provider config exists
        if config["provider"] not in article_config.llm.providers:
            article_config.llm.providers[config["provider"]] = {}
    
    workspace.save_config(article_config)
    
    # Run the workflow
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        try:
            success, results = asyncio.run(run_workflow(config, workspace, progress, verbose))
            
            if success:
                console.print("\n[green bold]✓ Article generation complete![/green bold]\n")
                
                # Show results
                table = Table(title="Generation Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Workspace", results["workspace"])
                table.add_row("Research Sources", str(results["research_sources"]))
                table.add_row("Outline Sections", str(results["outline_sections"]))
                table.add_row("Draft Words", str(results["draft_words"]))
                table.add_row("Exported Files", str(len(results["exports"])))
                
                console.print(table)
                
                if results["exports"]:
                    console.print("\n[cyan]Exported files:[/cyan]")
                    for export_path in results["exports"]:
                        console.print(f"  • {export_path}")
                
                return results
            else:
                raise click.ClickException("Workflow failed")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Workflow interrupted by user[/yellow]")
            raise
        except Exception as e:
            console.print(f"\n[red]Error during workflow: {e}[/red]")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise