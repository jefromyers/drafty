"""Outline generation service for article structure."""

import json
from typing import Any, Dict, List, Optional

from drafty.core.config import ArticleConfig
from drafty.providers.base import LLMProviderFactory, LLMMessage
from drafty.services.templates import PromptBuilder


class OutlineSection:
    """Represents a section in the article outline."""
    
    def __init__(
        self,
        heading: str,
        key_points: List[str],
        word_count: int = 300,
        level: int = 2,
        special_elements: Optional[List[str]] = None
    ):
        self.heading = heading
        self.key_points = key_points
        self.word_count = word_count
        self.level = level  # Heading level (1-6)
        self.special_elements = special_elements or []
        self.subsections: List[OutlineSection] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            "heading": self.heading,
            "key_points": self.key_points,
            "word_count": self.word_count,
            "level": self.level,
        }
        if self.special_elements:
            data["special_elements"] = self.special_elements
        if self.subsections:
            data["subsections"] = [s.to_dict() for s in self.subsections]
        return data


class ArticleOutline:
    """Complete article outline structure."""
    
    def __init__(
        self,
        title: str,
        meta_description: str,
        introduction: OutlineSection,
        sections: List[OutlineSection],
        conclusion: OutlineSection,
        total_word_count: int
    ):
        self.title = title
        self.meta_description = meta_description
        self.introduction = introduction
        self.sections = sections
        self.conclusion = conclusion
        self.total_word_count = total_word_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "meta_description": self.meta_description,
            "introduction": self.introduction.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "conclusion": self.conclusion.to_dict(),
            "total_word_count": self.total_word_count,
        }
    
    def get_all_sections(self) -> List[OutlineSection]:
        """Get all sections in order."""
        all_sections = [self.introduction]
        all_sections.extend(self.sections)
        all_sections.append(self.conclusion)
        return all_sections
    
    def calculate_total_words(self) -> int:
        """Calculate total word count from all sections."""
        total = self.introduction.word_count + self.conclusion.word_count
        for section in self.sections:
            total += section.word_count
            for subsection in section.subsections:
                total += subsection.word_count
        return total


class OutlineService:
    """Service for generating article outlines."""
    
    def __init__(self, config: ArticleConfig):
        """Initialize outline service."""
        self.config = config
        self.prompt_builder = PromptBuilder()
    
    async def generate_outline(
        self,
        research_data: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        style: Optional[str] = None,
        sections: Optional[int] = None
    ) -> ArticleOutline:
        """Generate article outline based on research and configuration.
        
        Args:
            research_data: Research results to inform outline
            provider_name: LLM provider to use
            style: Outline style (overrides config)
            sections: Number of sections (overrides config)
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
        
        # Build outline prompt
        outline_style = style or self.config.structure.outline_style
        num_sections = sections or len(self.config.structure.fixed_sections) or 5
        
        prompt = self.prompt_builder.build_outline_prompt(
            topic=self.config.content.topic,
            audience=self.config.content.audience,
            style=outline_style,
            research=research_context,
            sections=num_sections
        )
        
        # Generate outline
        messages = [
            LLMMessage(
                role="system",
                content="You are an expert content strategist creating article outlines."
            ),
            LLMMessage(role="user", content=prompt)
        ]
        
        # Use JSON mode if available
        kwargs = {}
        if hasattr(provider, 'supports_json_mode') and provider.supports_json_mode():
            kwargs['json_mode'] = True
        
        response = await provider.generate(messages, **kwargs)
        
        # Parse response
        try:
            if response.as_json():
                outline_data = response.as_json()
            else:
                outline_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: create basic outline
            outline_data = self._create_fallback_outline()
        
        # Convert to ArticleOutline object
        return self._parse_outline(outline_data)
    
    def _prepare_research_context(self, research_data: Optional[Dict[str, Any]]) -> str:
        """Prepare research context for outline generation."""
        if not research_data:
            return ""
        
        context_parts = []
        
        # Add topic analysis
        if research_data.get("topic_analysis"):
            analysis = research_data["topic_analysis"]
            context_parts.append("Key Concepts: " + ", ".join(analysis.get("key_concepts", [])))
            context_parts.append("Important Questions: " + ", ".join(analysis.get("important_questions", [])))
        
        # Add SERP insights
        if research_data.get("serp_analysis"):
            serp = research_data["serp_analysis"]
            if serp.get("people_also_ask"):
                questions = [q.get("question", "") for q in serp["people_also_ask"][:5] if q.get("question")]
                if questions:
                    context_parts.append("People Also Ask: " + " | ".join(questions))
        
        # Add top sources
        if research_data.get("search_results"):
            sources = research_data["search_results"][:5]
            titles = [s.get("title", "") for s in sources if s.get("title")]
            if titles:
                context_parts.append("Competitor Articles: " + " | ".join(titles))
        
        return "\n".join(context_parts)
    
    def _parse_outline(self, data: Dict[str, Any]) -> ArticleOutline:
        """Parse outline data into ArticleOutline object."""
        # Parse introduction
        intro_data = data.get("introduction", {})
        introduction = OutlineSection(
            heading=intro_data.get("heading", "Introduction"),
            key_points=intro_data.get("key_points", ["Overview of topic"]),
            word_count=intro_data.get("word_count", 150),
            level=2
        )
        
        # Parse main sections
        sections = []
        for section_data in data.get("sections", []):
            section = OutlineSection(
                heading=section_data.get("heading", "Section"),
                key_points=section_data.get("key_points", []),
                word_count=section_data.get("word_count", 300),
                level=2,
                special_elements=section_data.get("special_elements", [])
            )
            
            # Add subsections if present
            for subsection_data in section_data.get("subsections", []):
                subsection = OutlineSection(
                    heading=subsection_data.get("heading", "Subsection"),
                    key_points=subsection_data.get("key_points", []),
                    word_count=subsection_data.get("word_count", 200),
                    level=3
                )
                section.subsections.append(subsection)
            
            sections.append(section)
        
        # Parse conclusion
        conclusion_data = data.get("conclusion", {})
        conclusion = OutlineSection(
            heading=conclusion_data.get("heading", "Conclusion"),
            key_points=conclusion_data.get("key_points", ["Summary", "Next steps"]),
            word_count=conclusion_data.get("word_count", 150),
            level=2
        )
        
        # Create outline
        outline = ArticleOutline(
            title=data.get("title", self.config.content.topic),
            meta_description=data.get("meta_description", ""),
            introduction=introduction,
            sections=sections,
            conclusion=conclusion,
            total_word_count=data.get("total_word_count", 1500)
        )
        
        return outline
    
    def _create_fallback_outline(self) -> Dict[str, Any]:
        """Create a basic fallback outline."""
        return {
            "title": self.config.content.topic,
            "meta_description": f"Comprehensive guide to {self.config.content.topic}",
            "introduction": {
                "heading": "Introduction",
                "key_points": ["Topic overview", "Why it matters"],
                "word_count": 150
            },
            "sections": [
                {
                    "heading": "Understanding the Basics",
                    "key_points": ["Core concepts", "Key terminology"],
                    "word_count": 400
                },
                {
                    "heading": "Detailed Analysis",
                    "key_points": ["In-depth exploration", "Examples"],
                    "word_count": 500
                },
                {
                    "heading": "Practical Applications",
                    "key_points": ["Real-world use cases", "Best practices"],
                    "word_count": 400
                }
            ],
            "conclusion": {
                "heading": "Conclusion",
                "key_points": ["Summary", "Key takeaways", "Next steps"],
                "word_count": 150
            },
            "total_word_count": 1600
        }
    
    def enhance_outline_with_fixed_sections(
        self,
        outline: ArticleOutline
    ) -> ArticleOutline:
        """Enhance outline with fixed sections from config."""
        if not self.config.structure.fixed_sections:
            return outline
        
        # Map fixed sections to outline sections
        for fixed_heading in self.config.structure.fixed_sections:
            # Check if section already exists
            exists = any(
                s.heading.lower() == fixed_heading.lower()
                for s in outline.sections
            )
            
            if not exists:
                # Add missing fixed section
                new_section = OutlineSection(
                    heading=fixed_heading,
                    key_points=["Content for " + fixed_heading],
                    word_count=300,
                    level=2
                )
                outline.sections.append(new_section)
        
        # Recalculate total word count
        outline.total_word_count = outline.calculate_total_words()
        
        return outline
    
    def adjust_outline_for_word_count(
        self,
        outline: ArticleOutline,
        target_min: Optional[int] = None,
        target_max: Optional[int] = None
    ) -> ArticleOutline:
        """Adjust outline section word counts to meet target."""
        target_min = target_min or self.config.content.word_count.min
        target_max = target_max or self.config.content.word_count.max
        
        current_total = outline.calculate_total_words()
        
        # If within range, no adjustment needed
        if target_min <= current_total <= target_max:
            return outline
        
        # Calculate adjustment ratio
        target = (target_min + target_max) // 2
        ratio = target / current_total
        
        # Adjust all section word counts
        outline.introduction.word_count = int(outline.introduction.word_count * ratio)
        outline.conclusion.word_count = int(outline.conclusion.word_count * ratio)
        
        for section in outline.sections:
            section.word_count = int(section.word_count * ratio)
            for subsection in section.subsections:
                subsection.word_count = int(subsection.word_count * ratio)
        
        # Update total
        outline.total_word_count = outline.calculate_total_words()
        
        return outline