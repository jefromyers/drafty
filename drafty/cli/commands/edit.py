"""Edit command implementation."""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from drafty.core.workspace import Workspace
from drafty.services.edit import EditService, EditType, ContentAnalyzer

console = Console()


def edit_article(
    ctx,
    edit_types: List[str],
    model: Optional[str],
    target_grade: Optional[int],
    target_words: Optional[int],
    keywords: Optional[List[str]],
    tone: Optional[str],
    analyze: bool,
    interactive: bool
) -> str:
    """Edit and refine article content."""
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
    
    # Analyze content if requested
    if analyze:
        console.print("\n[cyan]Content Analysis[/cyan]")
        console.print("=" * 60)
        
        # Readability analysis
        readability = ContentAnalyzer.analyze_readability(draft_content)
        
        table = Table(title="Readability Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Flesch Reading Ease", f"{readability['flesch_reading_ease']:.1f}")
        table.add_row("Grade Level", f"{readability['flesch_kincaid_grade']:.1f}")
        table.add_row("Avg Sentence Length", f"{readability['avg_sentence_length']:.1f} words")
        table.add_row("Difficult Words", str(readability['difficult_words']))
        
        console.print(table)
        
        # SEO analysis if keywords provided
        if keywords or config.content.keywords:
            target_keywords = keywords or config.content.keywords
            seo_analysis = ContentAnalyzer.analyze_seo(draft_content, target_keywords)
            
            seo_table = Table(title="SEO Analysis")
            seo_table.add_column("Keyword", style="cyan")
            seo_table.add_column("Count", style="yellow")
            seo_table.add_column("Density", style="green")
            seo_table.add_column("In First 100", style="magenta")
            
            for kw, stats in seo_analysis["keyword_analysis"].items():
                seo_table.add_row(
                    kw,
                    str(stats["count"]),
                    f"{stats['density']}%",
                    "✓" if stats["in_first_100_words"] else "✗"
                )
            
            console.print(seo_table)
        
        # Get improvement suggestions
        suggestions = ContentAnalyzer.suggest_improvements(draft_content, config)
        if suggestions:
            console.print("\n[yellow]Suggested Improvements:[/yellow]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"{i}. {suggestion['reason']}")
        
        if not edit_types:
            return draft_content
    
    # Override provider model if specified
    if model:
        config.llm.providers[config.llm.default].model = model
    
    # Create edit service
    edit_service = EditService(config)
    
    # Prepare edits
    edits = []
    
    for edit_type_str in edit_types:
        edit_type_lower = edit_type_str.lower()
        
        if edit_type_lower == "readability":
            edits.append({
                "type": EditType.READABILITY,
                "target_grade_level": target_grade or 10
            })
        elif edit_type_lower == "seo":
            edits.append({
                "type": EditType.SEO,
                "keywords": keywords or config.content.keywords or []
            })
        elif edit_type_lower == "tone":
            edits.append({
                "type": EditType.TONE,
                "target_tone": tone or config.content.tone
            })
        elif edit_type_lower == "length":
            edits.append({
                "type": EditType.LENGTH,
                "target_word_count": target_words or config.content.word_count.get("target", 1500)
            })
        elif edit_type_lower == "clarity":
            edits.append({"type": EditType.CLARITY})
        elif edit_type_lower == "grammar":
            edits.append({"type": EditType.GRAMMAR})
        elif edit_type_lower == "style":
            edits.append({"type": EditType.STYLE})
        elif edit_type_lower == "all":
            # Add all basic edit types
            edits.extend([
                {"type": EditType.CLARITY},
                {"type": EditType.READABILITY, "target_grade_level": target_grade or 10},
                {"type": EditType.GRAMMAR},
                {"type": EditType.STYLE}
            ])
            if config.content.keywords:
                edits.append({
                    "type": EditType.SEO,
                    "keywords": keywords or config.content.keywords
                })
    
    if not edits:
        console.print("[yellow]No edits specified. Use --help to see available edit types.[/yellow]")
        return draft_content
    
    # Perform edits asynchronously
    async def do_edit():
        console.print(f"\n[cyan]Performing {len(edits)} edit(s)...[/cyan]")
        
        edited_content = await edit_service.batch_edit(
            draft_content,
            edits,
            provider_name=config.llm.default
        )
        
        return edited_content
    
    # Run the async function
    edited_content = asyncio.run(do_edit())
    
    # Save edited draft
    workspace.save_draft(edited_content, create_version=True)
    
    # Show comparison if interactive
    if interactive:
        original_words = len(draft_content.split())
        edited_words = len(edited_content.split())
        
        console.print("\n[cyan]Edit Summary[/cyan]")
        console.print(f"Original: {original_words} words")
        console.print(f"Edited: {edited_words} words")
        console.print(f"Change: {edited_words - original_words:+d} words")
        
        # Show readability comparison
        original_readability = ContentAnalyzer.analyze_readability(draft_content)
        edited_readability = ContentAnalyzer.analyze_readability(edited_content)
        
        console.print(f"\nGrade Level: {original_readability['flesch_kincaid_grade']:.1f} → {edited_readability['flesch_kincaid_grade']:.1f}")
        console.print(f"Reading Ease: {original_readability['flesch_reading_ease']:.1f} → {edited_readability['flesch_reading_ease']:.1f}")
    
    console.print(f"\n[green]✓[/green] Content edited and saved")
    
    return edited_content