"""Outline generation command."""

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional

import click

from drafty.core.workspace import Workspace
from drafty.services.outline import OutlineService


def generate_outline(
    ctx,
    style: Optional[str],
    sections: Optional[int],
    interactive: bool
) -> Dict:
    """Generate article outline."""
    # Load workspace and config
    workspace_path = Path.cwd()
    
    # Check if we're in a workspace
    if not (workspace_path / "article.json").exists():
        raise click.ClickException("Not in a Drafty workspace. Run 'drafty new' first.")
    
    workspace = Workspace.load(workspace_path)
    config = workspace.get_config()
    
    # Load research data if available
    research_data = None
    topic_analysis_file = workspace.research_dir / "topic_analysis.json"
    if topic_analysis_file.exists():
        with open(topic_analysis_file) as f:
            topic_analysis = json.load(f)
        
        serp_analysis = None
        serp_file = workspace.research_dir / "serp_analysis.json"
        if serp_file.exists():
            with open(serp_file) as f:
                serp_analysis = json.load(f)
        
        research_data = {
            "topic_analysis": topic_analysis,
            "serp_analysis": serp_analysis,
            "search_results": workspace.get_research_sources()
        }
    
    # Create outline service
    outline_service = OutlineService(config)
    
    # Generate outline asynchronously
    async def do_generate():
        outline = await outline_service.generate_outline(
            research_data=research_data,
            style=style,
            sections=sections
        )
        
        # Enhance with fixed sections if configured
        outline = outline_service.enhance_outline_with_fixed_sections(outline)
        
        # Adjust for word count
        outline = outline_service.adjust_outline_for_word_count(outline)
        
        return outline
    
    # Run the async function
    outline = asyncio.run(do_generate())
    
    # Save outline to workspace
    outline_file = workspace.drafts_dir / "outline.json"
    outline_data = outline.to_dict()
    outline_file.write_text(json.dumps(outline_data, indent=2))
    
    # Create markdown version for easy viewing
    outline_md = generate_outline_markdown(outline_data)
    outline_md_file = workspace.drafts_dir / "outline.md"
    outline_md_file.write_text(outline_md)
    
    # Interactive editing if requested
    if interactive:
        click.echo("\nGenerated outline saved to: drafts/outline.md")
        click.echo("Edit the file and press Enter when done...")
        click.pause()
        
        # Reload the edited outline
        # This is a simple implementation - could be enhanced
        click.echo("Outline updated.")
    
    return outline_data


def generate_outline_markdown(outline_data: Dict) -> str:
    """Generate markdown representation of outline."""
    lines = []
    
    # Title and meta
    lines.append(f"# {outline_data['title']}")
    lines.append("")
    if outline_data.get('meta_description'):
        lines.append(f"*{outline_data['meta_description']}*")
        lines.append("")
    
    lines.append(f"**Target Word Count:** {outline_data.get('total_word_count', 'TBD')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Introduction
    intro = outline_data.get('introduction', {})
    lines.append(f"## {intro.get('heading', 'Introduction')} ({intro.get('word_count', 150)} words)")
    for point in intro.get('key_points', []):
        lines.append(f"- {point}")
    lines.append("")
    
    # Main sections
    for i, section in enumerate(outline_data.get('sections', []), 1):
        lines.append(f"## {i}. {section.get('heading', 'Section')} ({section.get('word_count', 300)} words)")
        
        # Key points
        for point in section.get('key_points', []):
            lines.append(f"- {point}")
        
        # Special elements
        if section.get('special_elements'):
            lines.append(f"**Special Elements:** {', '.join(section['special_elements'])}")
        
        # Subsections
        for subsection in section.get('subsections', []):
            lines.append(f"### {subsection.get('heading', 'Subsection')} ({subsection.get('word_count', 200)} words)")
            for point in subsection.get('key_points', []):
                lines.append(f"  - {point}")
        
        lines.append("")
    
    # Conclusion
    conclusion = outline_data.get('conclusion', {})
    lines.append(f"## {conclusion.get('heading', 'Conclusion')} ({conclusion.get('word_count', 150)} words)")
    for point in conclusion.get('key_points', []):
        lines.append(f"- {point}")
    lines.append("")
    
    return "\n".join(lines)