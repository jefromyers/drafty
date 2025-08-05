"""Main CLI entry point for Drafty."""

import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from drafty.cli.commands import draft, edit, export, link, new, outline, research
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
@click.option("--readability", type=int, help="Target readability score")
@click.option("--tone", help="Adjust tone")
@click.option("--length", help="Adjust length (shorter/longer)")
@click.option("--seo", is_flag=True, help="Optimize for SEO")
@click.pass_obj
def edit(ctx: DraftyContext, readability: Optional[int], tone: Optional[str], length: Optional[str], seo: bool):
    """Edit and refine draft content."""
    from drafty.cli.commands.edit import refine_draft

    console.print("[cyan]Refining draft...[/cyan]")
    refined = refine_draft(ctx, readability, tone, length, seo)
    console.print("[green]✓[/green] Draft refined successfully")


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