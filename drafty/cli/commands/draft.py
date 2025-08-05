"""Draft generation command."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from drafty.core.workspace import Workspace
from drafty.services.draft import DraftService
from drafty.services.outline import ArticleOutline

console = Console()


def generate_draft(
    ctx,
    model: Optional[str],
    section: Optional[str],
    json_mode: bool,
    interactive: bool
) -> str:
    """Generate article draft."""
    # Load workspace and config
    workspace_path = Path.cwd()
    
    # Check if we're in a workspace
    if not (workspace_path / "article.json").exists():
        raise click.ClickException("Not in a Drafty workspace. Run 'drafty new' first.")
    
    workspace = Workspace.load(workspace_path)
    config = workspace.get_config()
    
    # Load outline
    outline_file = workspace.drafts_dir / "outline.json"
    if not outline_file.exists():
        raise click.ClickException("No outline found. Run 'drafty outline' first.")
    
    with open(outline_file) as f:
        outline_data = json.load(f)
    
    # Convert to ArticleOutline object
    from drafty.services.outline import OutlineSection
    
    # Parse outline data
    intro = OutlineSection(
        heading=outline_data["introduction"]["heading"],
        key_points=outline_data["introduction"]["key_points"],
        word_count=outline_data["introduction"]["word_count"]
    )
    
    sections = []
    for s in outline_data["sections"]:
        section_obj = OutlineSection(
            heading=s["heading"],
            key_points=s["key_points"],
            word_count=s["word_count"],
            special_elements=s.get("special_elements", [])
        )
        sections.append(section_obj)
    
    conclusion = OutlineSection(
        heading=outline_data["conclusion"]["heading"],
        key_points=outline_data["conclusion"]["key_points"],
        word_count=outline_data["conclusion"]["word_count"]
    )
    
    outline = ArticleOutline(
        title=outline_data["title"],
        meta_description=outline_data.get("meta_description", ""),
        introduction=intro,
        sections=sections,
        conclusion=conclusion,
        total_word_count=outline_data.get("total_word_count", 1500)
    )
    
    # Load research data if available
    research_data = None
    topic_analysis_file = workspace.research_dir / "topic_analysis.json"
    if topic_analysis_file.exists():
        with open(topic_analysis_file) as f:
            research_data = {"topic_analysis": json.load(f)}
        
        # Add scraped sources if available
        sources = workspace.get_research_sources()
        if sources:
            research_data["scraped_sources"] = sources
    
    # Override provider model if specified
    if model:
        config.llm.providers[config.llm.default].model = model
    
    # Enable JSON mode if requested (for OpenAI)
    if json_mode and config.llm.default == "openai":
        config.llm.providers["openai"].json_mode = True
    
    # Create draft service
    draft_service = DraftService(config)
    
    # Generate draft asynchronously with progress
    async def do_generate():
        # Determine which sections to generate
        sections_to_generate = None
        if section:
            sections_to_generate = [section]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating draft...", total=None)
            
            draft = await draft_service.generate_draft(
                outline=outline,
                research_data=research_data,
                sections_to_generate=sections_to_generate
            )
            
            progress.update(task, completed=True)
        
        return draft
    
    # Run the async function
    draft = asyncio.run(do_generate())
    
    # Save draft
    draft_content = draft.to_markdown()
    workspace.save_draft(draft_content, create_version=True)
    
    # Also save JSON version
    draft_json_file = workspace.drafts_dir / "draft.json"
    draft_json_file.write_text(json.dumps(draft.to_dict(), indent=2))
    
    # Interactive review if requested
    if interactive:
        console.print("\n[cyan]Draft generated![/cyan]")
        console.print(f"Saved to: drafts/current.md")
        console.print("\nFirst 500 characters:")
        console.print(draft_content[:500] + "...")
        
        if click.confirm("\nWould you like to regenerate any section?"):
            # Simple interactive regeneration
            section_name = click.prompt("Enter section heading")
            instructions = click.prompt("Enter regeneration instructions")
            
            async def regenerate():
                return await draft_service.regenerate_section(
                    section_heading=section_name,
                    outline=outline,
                    instructions=instructions,
                    current_content=None
                )
            
            new_section = asyncio.run(regenerate())
            console.print(f"\n[green]Section '{section_name}' regenerated![/green]")
    
    console.print(f"\n[green]✓[/green] Draft generated: {draft.total_word_count} words")
    
    return draft_content