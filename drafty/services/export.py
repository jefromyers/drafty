"""Export service for generating final article outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from drafty.core.config import ArticleConfig
from drafty.services.templates import TemplateManager


class ExportService:
    """Service for exporting articles to various formats."""
    
    def __init__(self, config: ArticleConfig):
        """Initialize export service."""
        self.config = config
        self.template_manager = TemplateManager()
    
    def export_markdown(
        self,
        draft_content: str,
        include_metadata: bool = True,
        clean: bool = True
    ) -> str:
        """Export article as clean markdown.
        
        Args:
            draft_content: Raw draft content
            include_metadata: Whether to include frontmatter
            clean: Whether to clean up formatting
        """
        # Check if it's a file path or actual content
        content = draft_content
        try:
            # Only check if it looks like a path (not multi-line content)
            if '\n' not in draft_content and Path(draft_content).exists():
                with open(draft_content) as f:
                    content = f.read()
        except (OSError, ValueError):
            # If it fails, assume it's content
            pass
        
        # Clean up if requested
        if clean:
            content = self._clean_markdown(content)
        
        # Add frontmatter if requested
        if include_metadata:
            frontmatter = self._generate_frontmatter()
            content = frontmatter + "\n" + content
        
        return content
    
    def export_html(
        self,
        draft_content: str,
        template_name: Optional[str] = None,
        include_styles: bool = True
    ) -> str:
        """Export article as HTML.
        
        Args:
            draft_content: Markdown draft content
            template_name: Custom HTML template to use
            include_styles: Whether to include CSS styles
        """
        import markdown
        from markdown.extensions import tables, fenced_code, codehilite, toc
        
        # Convert markdown to HTML
        md = markdown.Markdown(extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
            'sane_lists',
            'meta'
        ])
        
        # Parse markdown
        html_content = md.convert(draft_content)
        
        # Get metadata if available
        metadata = {
            "title": self.config.content.topic,
            "author": self.config.meta.extra.get("author", "Drafty AI"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "description": "",
        }
        
        if hasattr(md, 'Meta'):
            for key, value in md.Meta.items():
                metadata[key] = ' '.join(value) if isinstance(value, list) else value
        
        # Use template if specified
        if template_name:
            template = self.template_manager.get_template(f"outputs/{template_name}.j2")
        else:
            # Use default HTML template or create basic one
            try:
                template = self.template_manager.get_template("outputs/html.j2")
            except:
                # Fallback to basic HTML
                return self._generate_basic_html(html_content, metadata, include_styles)
        
        # Render with template
        return template.render(
            content=html_content,
            metadata=metadata,
            config=self.config,
            include_styles=include_styles
        )
    
    def export_json(
        self,
        draft_data: Dict[str, Any],
        include_config: bool = False
    ) -> str:
        """Export article data as JSON.
        
        Args:
            draft_data: Draft data dictionary
            include_config: Whether to include configuration
        """
        export_data = {
            "article": draft_data,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
            }
        }
        
        if include_config:
            export_data["config"] = self.config.model_dump()
        
        return json.dumps(export_data, indent=2, default=str)
    
    def export_text(
        self,
        draft_content: str,
        width: int = 80
    ) -> str:
        """Export article as plain text.
        
        Args:
            draft_content: Markdown draft content
            width: Line width for wrapping
        """
        import re
        import textwrap
        
        # Remove markdown formatting
        text = draft_content
        
        # Remove headers markdown
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.MULTILINE)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Clean up bullets
        text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
        
        # Wrap text
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            if line.strip():
                wrapped = textwrap.fill(line, width=width)
                wrapped_lines.append(wrapped)
            else:
                wrapped_lines.append('')
        
        return '\n'.join(wrapped_lines)
    
    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content."""
        import re
        
        # Remove multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Ensure headers have blank lines around them
        content = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', content)
        content = re.sub(r'(#{1,6}[^\n]+)\n([^\n])', r'\1\n\n\2', content)
        
        # Fix list formatting
        content = re.sub(r'\n\s*[-*+]\s+', '\n- ', content)
        
        # Remove trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Remove word count footer if present
        content = re.sub(r'\n---\n\*Total word count:.*\*$', '', content)
        
        return content.strip()
    
    def _generate_frontmatter(self) -> str:
        """Generate YAML frontmatter for markdown."""
        frontmatter = [
            "---",
            f"title: {self.config.content.topic}",
            f"date: {datetime.now().strftime('%Y-%m-%d')}",
            f"author: {self.config.meta.extra.get('author', 'Drafty AI')}",
        ]
        
        if self.config.content.keywords:
            keywords = ", ".join(self.config.content.keywords)
            frontmatter.append(f"keywords: {keywords}")
        
        if self.config.meta.tags:
            tags = ", ".join(self.config.meta.tags)
            frontmatter.append(f"tags: {tags}")
        
        frontmatter.append("---\n")
        
        return "\n".join(frontmatter)
    
    def _generate_basic_html(
        self,
        content: str,
        metadata: Dict[str, Any],
        include_styles: bool
    ) -> str:
        """Generate basic HTML without template."""
        styles = ""
        if include_styles:
            styles = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }
        h1 { font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.25em; }
        code {
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 85%;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
        }
        blockquote {
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0;
        }
        ul, ol {
            padding-left: 2em;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        table th, table td {
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
        }
        table th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
    </style>
"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.get('title', 'Article')}</title>
    <meta name="description" content="{metadata.get('description', '')}">
    <meta name="author" content="{metadata.get('author', '')}">
    {styles}
</head>
<body>
    <article>
        {content}
    </article>
    <footer>
        <p><small>Generated on {metadata.get('date', '')} by Drafty AI</small></p>
    </footer>
</body>
</html>"""
        
        return html