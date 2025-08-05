"""Research command implementation."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import click

from drafty.core.workspace import Workspace
from drafty.services.research import ResearchService


def run_research(
    ctx,
    provider: Optional[str],
    max_sources: int,
    queries: List[str]
) -> List[Dict]:
    """Run research phase."""
    # Load workspace and config
    workspace_path = Path.cwd()
    
    # Check if we're in a workspace
    if not (workspace_path / "article.json").exists():
        raise click.ClickException("Not in a Drafty workspace. Run 'drafty new' first.")
    
    workspace = Workspace.load(workspace_path)
    config = workspace.get_config()
    
    # Add seed queries if provided
    if queries:
        config.research.seed_queries.extend(queries)
    
    # Create research service
    research_service = ResearchService(config)
    
    # Run research asynchronously
    async def do_research():
        return await research_service.conduct_research(
            max_sources=max_sources,
            use_javascript=False,  # Could be a CLI option
            provider_name=provider
        )
    
    # Run the async function
    results = asyncio.run(do_research())
    
    # Save results to workspace
    if results.get("search_results"):
        for result in results["search_results"]:
            workspace.add_research_source({
                "url": result.get("url"),
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "query": result.get("query"),
                "relevance": result.get("relevance", 0.5)
            })
    
    # Save SERP analysis if available
    if results.get("serp_analysis"):
        serp_file = workspace.research_dir / "serp_analysis.json"
        serp_file.write_text(json.dumps(results["serp_analysis"], indent=2))
    
    # Save topic analysis
    if results.get("topic_analysis"):
        analysis_file = workspace.research_dir / "topic_analysis.json"
        analysis_file.write_text(json.dumps(results["topic_analysis"], indent=2))
    
    # Return search results for display
    return results.get("search_results", [])