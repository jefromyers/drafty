"""Main CLI entry point for Drafty."""

import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from drafty.cli.commands import draft, edit, export, generate, link, new, outline, research
from drafty.core.config import ArticleConfig

# Load environment variables from .env file
# Check multiple locations: current dir, parent dir, and drafty install dir
from dotenv import find_dotenv
env_file = find_dotenv(usecwd=True)
if not env_file:
    # Try the drafty installation directory
    drafty_dir = Path(__file__).parent.parent.parent
    env_file = drafty_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
else:
    load_dotenv(env_file)

console = Console()


class DraftyContext:
    """Context object to pass around CLI state."""

    def __init__(self):
        self.config: Optional[ArticleConfig] = None
        self.workspace: Optional[Path] = None
        self.verbose: bool = False
        self.debug: bool = False


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.version_option(version="0.1.0", prog_name="drafty")
@click.pass_context
def cli(ctx, verbose: bool, debug: bool, config: Optional[str]):
    """Drafty - AI-powered writing assistant for creating article drafts."""
    ctx.ensure_object(DraftyContext)
    ctx.obj.verbose = verbose
    ctx.obj.debug = debug

    if config:
        ctx.obj.config = ArticleConfig.from_file(Path(config))

    if ctx.invoked_subcommand is None:
        console.print(
            Panel.fit(
                "[bold cyan]Drafty[/bold cyan] - AI Writing Assistant\n\n"
                "Use [yellow]drafty --help[/yellow] to see available commands.",
                border_style="cyan",
            )
        )


@cli.command()
@click.argument("topic", required=False)
@click.option("--config", "-c", type=click.Path(exists=True), help="Load config from JSON file")
@click.option("--audience", "-a", help="Target audience")
@click.option("--keywords", "-k", help="SEO keywords (comma-separated)")
@click.option("--sections", "-s", type=int, help="Number of sections")
@click.option("--word-count", "-w", type=int, help="Target word count")
@click.option("--provider", "-p", help="LLM provider (openai/gemini)")
@click.option("--style", help="Article style (guide/howto/listicle)")
@click.option("--tone", help="Writing tone (professional/casual/friendly)")
@click.option("--edit-types", help="Edit types (all/clarity/seo/readability)")
@click.option("--export-formats", "-f", help="Export formats (markdown,html,json)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--skip-research", is_flag=True, help="Skip research phase")
@click.option("--skip-edit", is_flag=True, help="Skip editing phase")
@click.option("--save-config", type=click.Path(), help="Save config to JSON file")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.pass_obj
def generate(ctx: DraftyContext, topic: Optional[str], config: Optional[str], audience: Optional[str],
             keywords: Optional[str], sections: Optional[int], word_count: Optional[int],
             provider: Optional[str], style: Optional[str], tone: Optional[str],
             edit_types: Optional[str], export_formats: Optional[str], output_dir: Optional[str],
             skip_research: bool, skip_edit: bool, save_config: Optional[str],
             dry_run: bool, verbose: bool):
    """Generate a complete article with automated workflow.
    
    Examples:
        drafty generate "My Topic" --audience "Developers"
        drafty generate --config article.json
        drafty generate --config base.json --topic "Override Topic"
    """
    from drafty.cli.commands.generate import generate_article
    from pathlib import Path
    
    # Parse comma-separated values
    keyword_list = keywords.split(",") if keywords else None
    edit_type_list = edit_types.split(",") if edit_types else None
    format_list = export_formats.split(",") if export_formats else None
    
    # Convert paths
    config_path = Path(config) if config else None
    save_path = Path(save_config) if save_config else None
    output_path = Path(output_dir) if output_dir else None
    
    try:
        results = generate_article(
            topic=topic,
            config_file=config_path,
            audience=audience,
            keywords=keyword_list,
            sections=sections,
            word_count=word_count,
            provider=provider,
            style=style,
            tone=tone,
            edit_types=edit_type_list,
            export_formats=format_list,
            output_dir=output_path,
            skip_research=skip_research,
            skip_edit=skip_edit,
            save_config=save_path,
            dry_run=dry_run,
            verbose=verbose
        )
        
        if not dry_run:
            console.print("\n[green bold]✨ Article generated successfully![/green bold]")
    except Exception as e:
        console.print(f"[red]Generation failed: {e}[/red]")
        if ctx.debug:
            import traceback
            console.print(traceback.format_exc())
        raise


@cli.command()
@click.argument("slug")
@click.option("--template", "-t", help="Template to use for initialization")
@click.option("--topic", help="Article topic")
@click.option("--audience", help="Target audience")
@click.pass_obj
def new(ctx: DraftyContext, slug: str, template: Optional[str], topic: Optional[str], audience: Optional[str]):
    """Initialize a new article workspace."""
    from drafty.cli.commands.new import create_workspace

    console.print(f"[cyan]Creating new article workspace:[/cyan] {slug}")
    workspace = create_workspace(slug, template, topic, audience)
    console.print(f"[green]✓[/green] Workspace created at: {workspace}")


@cli.command()
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--max-sources", type=int, default=10, help="Maximum sources to gather")
@click.option("--queries", "-q", multiple=True, help="Seed queries for research")
@click.pass_obj
def research(ctx: DraftyContext, provider: Optional[str], max_sources: int, queries: tuple):
    """Gather and analyze research sources."""
    from drafty.cli.commands.research import run_research

    console.print("[cyan]Starting research phase...[/cyan]")
    with console.status("[yellow]Gathering sources...[/yellow]"):
        results = run_research(ctx, provider, max_sources, list(queries))
    console.print(f"[green]✓[/green] Research complete: {len(results)} sources gathered")


@cli.command()
@click.option("--style", "-s", help="Outline style (hub, howto, comparison, listicle, guide)")
@click.option("--sections", type=int, help="Number of sections")
@click.option("--interactive", "-i", is_flag=True, help="Interactive outline editing")
@click.pass_obj
def outline(ctx: DraftyContext, style: Optional[str], sections: Optional[int], interactive: bool):
    """Generate or refine article outline."""
    from drafty.cli.commands.outline import generate_outline

    console.print("[cyan]Generating outline...[/cyan]")
    outline_data = generate_outline(ctx, style, sections, interactive)
    console.print("[green]✓[/green] Outline generated successfully")


@cli.command()
@click.option("--model", "-m", help="LLM model to use")
@click.option("--section", help="Draft specific section only")
@click.option("--json-mode", is_flag=True, help="Use JSON mode for OpenAI")
@click.option("--interactive", "-i", is_flag=True, help="Review each section")
@click.pass_obj
def draft(ctx: DraftyContext, model: Optional[str], section: Optional[str], json_mode: bool, interactive: bool):
    """Generate article draft content."""
    from drafty.cli.commands.draft import generate_draft

    console.print("[cyan]Generating draft...[/cyan]")
    with console.status("[yellow]Writing content...[/yellow]"):
        draft_content = generate_draft(ctx, model, section, json_mode, interactive)
    console.print("[green]✓[/green] Draft generated successfully")


@cli.command()
@click.argument("edit_types", nargs=-1)
@click.option("--model", "-m", help="LLM model to use")
@click.option("--target-grade", type=int, help="Target readability grade level")
@click.option("--target-words", type=int, help="Target word count")
@click.option("--keywords", help="Keywords for SEO (comma-separated)")
@click.option("--tone", help="Target tone (professional, casual, friendly, etc.)")
@click.option("--analyze", is_flag=True, help="Analyze content without editing")
@click.option("--interactive", "-i", is_flag=True, help="Interactive editing mode")
@click.pass_obj
def edit(ctx: DraftyContext, edit_types: tuple, model: Optional[str], target_grade: Optional[int], 
         target_words: Optional[int], keywords: Optional[str], tone: Optional[str], 
         analyze: bool, interactive: bool):
    """Edit and refine draft content.
    
    Examples:
        drafty edit --analyze
        drafty edit all
        drafty edit readability seo
        drafty edit tone --tone friendly
    """
    from drafty.cli.commands.edit import edit_article

    # Parse keywords if provided
    keyword_list = keywords.split(",") if keywords else None
    
    # Convert edit_types tuple to list
    edit_type_list = list(edit_types) if edit_types else []
    
    console.print("[cyan]Processing edits...[/cyan]")
    result = edit_article(
        ctx, 
        edit_type_list,
        model,
        target_grade,
        target_words,
        keyword_list,
        tone,
        analyze,
        interactive
    )
    console.print("[green]✓[/green] Edit complete")


@cli.command()
@click.option("--suggest", is_flag=True, help="Suggest link placements")
@click.option("--use-ner", is_flag=True, help="Use NER for smart linking")
@click.option("--validate", is_flag=True, help="Validate existing links")
@click.pass_obj
def link(ctx: DraftyContext, suggest: bool, use_ner: bool, validate: bool):
    """Insert and manage article links."""
    from drafty.cli.commands.link import manage_links

    console.print("[cyan]Managing links...[/cyan]")
    link_results = manage_links(ctx, suggest, use_ner, validate)
    console.print(f"[green]✓[/green] Links processed: {link_results['total']} links")


@cli.command()
@click.option("--format", "-f", multiple=True, default=["markdown"], help="Export format(s)")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--template", help="Custom template to use")
@click.pass_obj
def export(ctx: DraftyContext, format: tuple, output: Optional[str], template: Optional[str]):
    """Export article to various formats."""
    from drafty.cli.commands.export import export_article

    console.print(f"[cyan]Exporting to {', '.join(format)}...[/cyan]")
    exported_files = export_article(ctx, list(format), output, template)
    for file in exported_files:
        console.print(f"[green]✓[/green] Exported: {file}")


@cli.command()
@click.pass_obj
def chat(ctx: DraftyContext):
    """Interactive chat mode for refinements."""
    console.print("[cyan]Entering interactive chat mode...[/cyan]")
    console.print("[yellow]Type 'exit' to quit, 'help' for commands[/yellow]\n")

    while True:
        try:
            user_input = console.input("[bold]You:[/bold] ")
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "help":
                console.print(
                    "[yellow]Available commands:[/yellow]\n"
                    "  rewrite <section> - Rewrite a section\n"
                    "  expand <section> - Expand a section\n"
                    "  shorten <section> - Shorten a section\n"
                    "  improve <text> - Improve specific text\n"
                    "  exit/quit - Exit chat mode"
                )
            else:
                console.print(f"[bold cyan]Assistant:[/bold cyan] Processing: {user_input}")
        except KeyboardInterrupt:
            break

    console.print("\n[cyan]Exiting chat mode.[/cyan]")


@cli.command()
@click.argument("action", type=click.Choice(["analyze", "entities", "readability"]))
@click.option("--file", "-f", type=click.Path(exists=True), help="File to analyze")
@click.pass_obj
def nlp(ctx: DraftyContext, action: str, file: Optional[str]):
    """NLP analysis tools."""
    from drafty.cli.commands.nlp import run_nlp_analysis

    console.print(f"[cyan]Running NLP {action} analysis...[/cyan]")
    results = run_nlp_analysis(ctx, action, file)
    console.print("[green]✓[/green] Analysis complete")


@cli.command()
@click.option("--all", is_flag=True, help="Show all configuration")
@click.option("--section", help="Show specific section")
@click.option("--status", is_flag=True, help="Show configuration status")
@click.pass_obj
def config(ctx: DraftyContext, all: bool, section: Optional[str], status: bool):
    """View and manage configuration."""
    if status:
        # Show environment configuration status
        from drafty.core.settings import settings
        settings.print_status()
        return
    
    if not ctx.config:
        console.print("[red]No configuration loaded[/red]")
        console.print("[yellow]Tip: Use --status to check environment configuration[/yellow]")
        return

    if all:
        import json

        console.print(json.dumps(ctx.config.model_dump(), indent=2))
    elif section:
        section_data = getattr(ctx.config, section, None)
        if section_data:
            import json

            console.print(json.dumps(section_data.model_dump(), indent=2))
        else:
            console.print(f"[red]Section '{section}' not found[/red]")
    else:
        console.print(f"Slug: {ctx.config.meta.slug}")
        console.print(f"Topic: {ctx.config.content.topic}")
        console.print(f"Status: {ctx.config.meta.status}")


def main():
    """Main entry point."""
    try:
        cli(obj=DraftyContext())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if "--debug" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()