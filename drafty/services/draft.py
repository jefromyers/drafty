"""Draft generation service for writing article content."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from drafty.core.config import ArticleConfig
from drafty.providers.base import LLMProviderFactory, LLMMessage
from drafty.services.outline import ArticleOutline, OutlineSection
from drafty.services.templates import PromptBuilder


class DraftSection:
    """Represents a drafted section of content."""
    
    def __init__(
        self,
        heading: str,
        content: str,
        word_count: int,
        outline_section: OutlineSection
    ):
        self.heading = heading
        self.content = content
        self.word_count = word_count
        self.outline_section = outline_section
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "heading": self.heading,
            "content": self.content,
            "word_count": self.word_count,
            "key_points_covered": self.outline_section.key_points
        }


class ArticleDraft:
    """Complete article draft."""
    
    def __init__(
        self,
        title: str,
        meta_description: str,
        sections: List[DraftSection],
        total_word_count: int
    ):
        self.title = title
        self.meta_description = meta_description
        self.sections = sections
        self.total_word_count = total_word_count
    
    def to_markdown(self) -> str:
        """Convert draft to markdown format."""
        lines = []
        
        # Title
        lines.append(f"# {self.title}")
        lines.append("")
        
        # Meta description
        if self.meta_description:
            lines.append(f"*{self.meta_description}*")
            lines.append("")
        
        # Sections
        for section in self.sections:
            # Add appropriate heading level
            heading_prefix = "#" * section.outline_section.level
            lines.append(f"{heading_prefix} {section.heading}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        
        # Footer with word count
        lines.append("---")
        lines.append(f"*Total word count: {self.total_word_count}*")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "meta_description": self.meta_description,
            "sections": [s.to_dict() for s in self.sections],
            "total_word_count": self.total_word_count
        }


class DraftService:
    """Service for generating article drafts."""
    
    def __init__(self, config: ArticleConfig):
        """Initialize draft service."""
        self.config = config
        self.prompt_builder = PromptBuilder()
    
    async def generate_draft(
        self,
        outline: ArticleOutline,
        research_data: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        sections_to_generate: Optional[List[str]] = None
    ) -> ArticleDraft:
        """Generate complete article draft from outline.
        
        Args:
            outline: Article outline to follow
            research_data: Research data to reference
            provider_name: LLM provider to use
            sections_to_generate: Specific sections to generate (None = all)
        """
        # Get provider
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        # Convert Pydantic model to dict if needed
        if hasattr(provider_config, 'model_dump'):
            provider_config = provider_config.model_dump()
        elif not isinstance(provider_config, dict):
            provider_config = {}
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        # Prepare research context
        research_context = self._prepare_research_context(research_data)
        
        # Generate each section
        draft_sections = []
        
        # Get all sections to generate
        all_sections = outline.get_all_sections()
        
        for section in all_sections:
            # Skip if not in sections_to_generate (if specified)
            if sections_to_generate and section.heading not in sections_to_generate:
                continue
            
            # Generate section content
            section_content = await self._generate_section(
                section=section,
                outline=outline,
                research_context=research_context,
                previous_sections=draft_sections,
                provider=provider
            )
            
            # Count words
            word_count = len(section_content.split())
            
            # Create draft section
            draft_section = DraftSection(
                heading=section.heading,
                content=section_content,
                word_count=word_count,
                outline_section=section
            )
            
            draft_sections.append(draft_section)
            
            # Optional: Add delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        # Calculate total word count
        total_words = sum(s.word_count for s in draft_sections)
        
        # Create article draft
        draft = ArticleDraft(
            title=outline.title,
            meta_description=outline.meta_description,
            sections=draft_sections,
            total_word_count=total_words
        )
        
        return draft
    
    async def _generate_section(
        self,
        section: OutlineSection,
        outline: ArticleOutline,
        research_context: str,
        previous_sections: List[DraftSection],
        provider: LLMProviderFactory
    ) -> str:
        """Generate content for a single section."""
        # Build context from previous sections
        previous_content = ""
        if previous_sections:
            previous_content = "Previously written sections:\n"
            for prev in previous_sections[-2:]:  # Last 2 sections for context
                previous_content += f"\n{prev.heading}:\n{prev.content[:500]}...\n"
        
        # Build section context
        section_context = f"""Article Title: {outline.title}
Topic: {self.config.content.topic}
Target Audience: {self.config.content.audience}

Outline for this section:
- Heading: {section.heading}
- Key Points: {', '.join(section.key_points)}
- Target Word Count: {section.word_count}
{f'- Special Elements: {", ".join(section.special_elements)}' if section.special_elements else ''}

{previous_content}

{f'Research Context: {research_context[:1000]}' if research_context else ''}"""
        
        # Build prompt
        prompt = self.prompt_builder.build_draft_prompt(
            section_title=section.heading,
            section_context=section_context,
            topic=self.config.content.topic,
            audience=self.config.content.audience,
            tone=self.config.content.tone,
            word_count={"min": int(section.word_count * 0.9), "max": int(section.word_count * 1.1)},
            key_points=section.key_points
        )
        
        # Generate content
        messages = [
            LLMMessage(
                role="system",
                content="You are a professional content writer creating high-quality article sections."
            ),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        
        # Parse JSON response if needed
        content = response.content
        if content.startswith('{') and content.endswith('}'):
            try:
                import json
                data = json.loads(content)
                # Extract the actual content from JSON response
                if 'section' in data and 'content' in data['section']:
                    content = data['section']['content']
                elif 'content' in data:
                    content = data['content']
            except json.JSONDecodeError:
                pass  # Use raw content if not valid JSON
        
        # Remove duplicate heading if it exists
        if content.startswith(f"## {section.heading}\n"):
            content = content[len(f"## {section.heading}\n"):].lstrip()
        elif content.startswith(f"### {section.heading}\n"):
            content = content[len(f"### {section.heading}\n"):].lstrip()
        
        return content
    
    def _prepare_research_context(self, research_data: Optional[Dict[str, Any]]) -> str:
        """Prepare research context for draft generation."""
        if not research_data:
            return ""
        
        context_parts = []
        
        # Add key facts from research
        if research_data.get("topic_analysis"):
            analysis = research_data["topic_analysis"]
            if analysis.get("key_concepts"):
                context_parts.append("Key Concepts: " + ", ".join(analysis["key_concepts"][:5]))
        
        # Add relevant source snippets
        if research_data.get("scraped_sources"):
            context_parts.append("\nRelevant Information from Sources:")
            for source in research_data["scraped_sources"][:3]:
                if source.get("content"):
                    snippet = source["content"][:500]
                    context_parts.append(f"- {snippet}...")
        
        # Add SERP insights
        if research_data.get("serp_analysis", {}).get("featured_snippet"):
            snippet = research_data["serp_analysis"]["featured_snippet"]
            context_parts.append(f"\nFeatured Snippet: {snippet}")
        
        return "\n".join(context_parts)
    
    async def regenerate_section(
        self,
        section_heading: str,
        outline: ArticleOutline,
        instructions: str,
        current_content: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> DraftSection:
        """Regenerate a specific section with instructions.
        
        Args:
            section_heading: Heading of section to regenerate
            outline: Article outline
            instructions: Specific instructions for regeneration
            current_content: Current content to improve
            provider_name: LLM provider to use
        """
        # Find the outline section
        outline_section = None
        for section in outline.get_all_sections():
            if section.heading == section_heading:
                outline_section = section
                break
        
        if not outline_section:
            raise ValueError(f"Section '{section_heading}' not found in outline")
        
        # Get provider
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        # Convert Pydantic model to dict if needed
        if hasattr(provider_config, 'model_dump'):
            provider_config = provider_config.model_dump()
        elif not isinstance(provider_config, dict):
            provider_config = {}
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        # Build regeneration prompt
        current_content_section = ''
        if current_content:
            current_content_section = f'Current Content:\n{current_content}\n\n'
        
        prompt = f"""Please regenerate the following section with these instructions: {instructions}

Section: {section_heading}
Key Points: {', '.join(outline_section.key_points)}
Target Word Count: {outline_section.word_count}

{current_content_section}Write an improved version following the instructions."""
        
        messages = [
            LLMMessage(
                role="system",
                content="You are improving article content based on specific feedback."
            ),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        
        # Create draft section
        return DraftSection(
            heading=section_heading,
            content=response.content,
            word_count=len(response.content.split()),
            outline_section=outline_section
        )