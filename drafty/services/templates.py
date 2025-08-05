"""Template management using Jinja2."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape


class TemplateManager:
    """Manages Jinja2 templates for prompts and outputs."""

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize template manager.
        
        Args:
            template_dir: Directory containing templates. Defaults to built-in templates.
        """
        if template_dir is None:
            # Use default templates directory
            template_dir = Path(__file__).parent.parent / "templates"
        
        self.template_dir = Path(template_dir)
        
        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader([self.template_dir]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Add custom filters
        self.env.filters['json'] = lambda x: json.dumps(x, indent=2)
        self.env.filters['markdown_escape'] = self._markdown_escape
        self.env.filters['truncate_words'] = self._truncate_words

    @staticmethod
    def _markdown_escape(text: str) -> str:
        """Escape special markdown characters."""
        chars_to_escape = ['*', '_', '`', '[', ']', '#', '-', '+', '!']
        for char in chars_to_escape:
            text = text.replace(char, f'\\{char}')
        return text

    @staticmethod
    def _truncate_words(text: str, length: int = 100) -> str:
        """Truncate text to specified number of words."""
        words = text.split()
        if len(words) <= length:
            return text
        return ' '.join(words[:length]) + '...'

    def get_template(self, name: str) -> Template:
        """Get a template by name.
        
        Args:
            name: Template name (e.g., 'prompts/research.j2')
        """
        return self.env.get_template(name)

    def render(self, template_name: str, **context) -> str:
        """Render a template with context.
        
        Args:
            template_name: Name of the template
            **context: Variables to pass to the template
        """
        template = self.get_template(template_name)
        return template.render(**context)

    def render_prompt(self, prompt_type: str, **context) -> str:
        """Render a prompt template.
        
        Args:
            prompt_type: Type of prompt (research, outline, draft, etc.)
            **context: Variables for the prompt
        """
        template_name = f"prompts/{prompt_type}.j2"
        return self.render(template_name, **context)

    def render_output(self, output_type: str, **context) -> str:
        """Render an output template.
        
        Args:
            output_type: Type of output (markdown, html, etc.)
            **context: Variables for the output
        """
        template_name = f"outputs/{output_type}.j2"
        return self.render(template_name, **context)

    def list_templates(self, subdirectory: Optional[str] = None) -> list[str]:
        """List available templates.
        
        Args:
            subdirectory: Optional subdirectory to list (e.g., 'prompts')
        """
        search_dir = self.template_dir
        if subdirectory:
            search_dir = search_dir / subdirectory
        
        templates = []
        if search_dir.exists():
            for template_file in search_dir.rglob("*.j2"):
                relative_path = template_file.relative_to(self.template_dir)
                templates.append(str(relative_path))
        
        return sorted(templates)

    def add_custom_template(self, name: str, content: str) -> None:
        """Add a custom template.
        
        Args:
            name: Template name (with .j2 extension)
            content: Template content
        """
        template_path = self.template_dir / name
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(content)

    def create_from_string(self, template_string: str) -> Template:
        """Create a template from a string.
        
        Args:
            template_string: Template content as string
        """
        return self.env.from_string(template_string)


class PromptBuilder:
    """Helper class for building structured prompts."""

    def __init__(self, template_manager: Optional[TemplateManager] = None):
        """Initialize prompt builder."""
        self.template_manager = template_manager or TemplateManager()

    def build_research_prompt(
        self,
        topic: str,
        audience: str,
        queries: list[str],
        sources: Optional[list[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Build a research prompt."""
        return self.template_manager.render_prompt(
            "research",
            topic=topic,
            audience=audience,
            queries=queries,
            sources=sources or [],
            **kwargs
        )

    def build_outline_prompt(
        self,
        topic: str,
        audience: str,
        style: str,
        research: Optional[str] = None,
        sections: Optional[int] = None,
        **kwargs
    ) -> str:
        """Build an outline generation prompt."""
        return self.template_manager.render_prompt(
            "outline",
            topic=topic,
            audience=audience,
            style=style,
            research=research,
            sections=sections,
            **kwargs
        )

    def build_draft_prompt(
        self,
        section_title: str,
        section_context: str,
        topic: str,
        audience: str,
        tone: list[str],
        word_count: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> str:
        """Build a draft section prompt."""
        return self.template_manager.render_prompt(
            "draft_section",
            section_title=section_title,
            section_context=section_context,
            topic=topic,
            audience=audience,
            tone=tone,
            word_count=word_count,
            **kwargs
        )

    def build_edit_prompt(
        self,
        content: str,
        edit_type: str,
        instructions: str,
        **kwargs
    ) -> str:
        """Build an editing prompt."""
        return self.template_manager.render_prompt(
            "edit",
            content=content,
            edit_type=edit_type,
            instructions=instructions,
            **kwargs
        )

    def build_link_prompt(
        self,
        content: str,
        links: list[Dict[str, str]],
        use_ner: bool = False,
        **kwargs
    ) -> str:
        """Build a link insertion prompt."""
        return self.template_manager.render_prompt(
            "link",
            content=content,
            links=links,
            use_ner=use_ner,
            **kwargs
        )