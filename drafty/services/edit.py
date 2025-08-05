"""Edit and refinement service for improving generated content."""

import asyncio
import re
from typing import Any, Dict, List, Optional
from enum import Enum

from drafty.core.config import ArticleConfig
from drafty.providers.base import LLMMessage, LLMProviderFactory
from drafty.services.templates import PromptBuilder


class EditType(Enum):
    """Types of edits that can be performed."""
    CLARITY = "clarity"
    READABILITY = "readability"
    SEO = "seo"
    TONE = "tone"
    LENGTH = "length"
    FACT_CHECK = "fact_check"
    GRAMMAR = "grammar"
    STYLE = "style"


class EditService:
    """Service for editing and refining article content."""
    
    def __init__(self, config: ArticleConfig):
        """Initialize edit service."""
        self.config = config
        self.prompt_builder = PromptBuilder()
    
    async def edit_content(
        self,
        content: str,
        edit_types: List[EditType],
        provider_name: Optional[str] = None,
        instructions: Optional[str] = None,
        target_word_count: Optional[int] = None
    ) -> str:
        """Edit content with specified improvements.
        
        Args:
            content: Content to edit
            edit_types: Types of edits to perform
            provider_name: LLM provider to use
            instructions: Additional editing instructions
            target_word_count: Target word count if adjusting length
        """
        # Get provider
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        # Build editing prompt
        prompt = self._build_edit_prompt(
            content=content,
            edit_types=edit_types,
            instructions=instructions,
            target_word_count=target_word_count
        )
        
        messages = [
            LLMMessage(
                role="system",
                content="You are a professional editor improving article content while maintaining its core message and structure."
            ),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        
        # Parse JSON response if needed
        edited_content = response.content
        if edited_content.startswith('{') and edited_content.endswith('}'):
            try:
                import json
                data = json.loads(edited_content)
                if 'content' in data:
                    edited_content = data['content']
                elif 'edited' in data:
                    edited_content = data['edited']
            except json.JSONDecodeError:
                pass
        
        return edited_content
    
    async def improve_readability(
        self,
        content: str,
        target_grade_level: int = 8,
        provider_name: Optional[str] = None
    ) -> str:
        """Improve content readability for target audience.
        
        Args:
            content: Content to improve
            target_grade_level: Target reading grade level (default 8th grade)
            provider_name: LLM provider to use
        """
        prompt = f"""Improve the readability of this content for a {target_grade_level}th grade reading level:

{content}

Guidelines:
- Use shorter sentences (15-20 words average)
- Use simpler vocabulary where appropriate
- Break up long paragraphs
- Add transitions between ideas
- Maintain professional tone
- Keep technical terms but explain them

Return the improved content only."""
        
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        messages = [
            LLMMessage(role="system", content="You are a readability expert."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        return response.content
    
    async def optimize_seo(
        self,
        content: str,
        keywords: List[str],
        provider_name: Optional[str] = None
    ) -> str:
        """Optimize content for SEO.
        
        Args:
            content: Content to optimize
            keywords: Target keywords to incorporate
            provider_name: LLM provider to use
        """
        prompt = f"""Optimize this content for SEO with these target keywords: {', '.join(keywords)}

Content:
{content}

Guidelines:
- Naturally incorporate keywords (2-3% density)
- Add semantic variations of keywords
- Ensure keywords appear in first 100 words
- Use keywords in subheadings where appropriate
- Maintain natural, readable flow
- Don't keyword stuff

Return the SEO-optimized content only."""
        
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        messages = [
            LLMMessage(role="system", content="You are an SEO content optimization expert."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        return response.content
    
    async def adjust_tone(
        self,
        content: str,
        target_tone: str,
        provider_name: Optional[str] = None
    ) -> str:
        """Adjust content tone.
        
        Args:
            content: Content to adjust
            target_tone: Target tone (e.g., professional, casual, authoritative)
            provider_name: LLM provider to use
        """
        prompt = f"""Adjust the tone of this content to be more {target_tone}:

{content}

Guidelines for {target_tone} tone:
{self._get_tone_guidelines(target_tone)}

Maintain all facts and key information while adjusting the tone.

Return the adjusted content only."""
        
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        messages = [
            LLMMessage(role="system", content="You are a tone and style expert."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        return response.content
    
    async def adjust_length(
        self,
        content: str,
        target_word_count: int,
        provider_name: Optional[str] = None
    ) -> str:
        """Adjust content length to target word count.
        
        Args:
            content: Content to adjust
            target_word_count: Target word count
            provider_name: LLM provider to use
        """
        current_word_count = len(content.split())
        
        if current_word_count < target_word_count:
            action = "expand"
            instruction = f"Expand this content from {current_word_count} to approximately {target_word_count} words"
        else:
            action = "condense"
            instruction = f"Condense this content from {current_word_count} to approximately {target_word_count} words"
        
        prompt = f"""{instruction}:

{content}

Guidelines:
- {"Add more details, examples, and explanations" if action == "expand" else "Remove redundancy and less important details"}
- Maintain all key points and main message
- Keep the structure and flow intact
- Target: {target_word_count} words (±10%)

Return the adjusted content only."""
        
        provider_name = provider_name or self.config.llm.default
        provider_config = self.config.llm.providers.get(provider_name, {})
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        messages = [
            LLMMessage(role="system", content="You are a content editor expert at adjusting length."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await provider.generate(messages)
        return response.content
    
    async def batch_edit(
        self,
        content: str,
        edits: List[Dict[str, Any]],
        provider_name: Optional[str] = None
    ) -> str:
        """Perform multiple edits in sequence.
        
        Args:
            content: Content to edit
            edits: List of edit configurations
            provider_name: LLM provider to use
        """
        edited_content = content
        
        for edit in edits:
            edit_type = EditType(edit.get("type"))
            
            if edit_type == EditType.READABILITY:
                edited_content = await self.improve_readability(
                    edited_content,
                    target_grade_level=edit.get("target_grade_level", 8),
                    provider_name=provider_name
                )
            elif edit_type == EditType.SEO:
                edited_content = await self.optimize_seo(
                    edited_content,
                    keywords=edit.get("keywords", self.config.content.keywords or []),
                    provider_name=provider_name
                )
            elif edit_type == EditType.TONE:
                edited_content = await self.adjust_tone(
                    edited_content,
                    target_tone=edit.get("target_tone", self.config.content.tone),
                    provider_name=provider_name
                )
            elif edit_type == EditType.LENGTH:
                edited_content = await self.adjust_length(
                    edited_content,
                    target_word_count=edit.get("target_word_count"),
                    provider_name=provider_name
                )
            else:
                # General edit
                edited_content = await self.edit_content(
                    edited_content,
                    edit_types=[edit_type],
                    instructions=edit.get("instructions"),
                    provider_name=provider_name
                )
        
        return edited_content
    
    def _build_edit_prompt(
        self,
        content: str,
        edit_types: List[EditType],
        instructions: Optional[str] = None,
        target_word_count: Optional[int] = None
    ) -> str:
        """Build prompt for editing content."""
        edit_instructions = []
        
        for edit_type in edit_types:
            if edit_type == EditType.CLARITY:
                edit_instructions.append("- Improve clarity and remove ambiguity")
            elif edit_type == EditType.READABILITY:
                edit_instructions.append("- Simplify complex sentences and improve flow")
            elif edit_type == EditType.SEO:
                keywords = self.config.content.keywords or []
                if keywords:
                    edit_instructions.append(f"- Optimize for SEO with keywords: {', '.join(keywords[:5])}")
            elif edit_type == EditType.TONE:
                edit_instructions.append(f"- Adjust tone to be more {self.config.content.tone}")
            elif edit_type == EditType.LENGTH:
                if target_word_count:
                    current_count = len(content.split())
                    if current_count < target_word_count:
                        edit_instructions.append(f"- Expand content to approximately {target_word_count} words")
                    else:
                        edit_instructions.append(f"- Condense content to approximately {target_word_count} words")
            elif edit_type == EditType.FACT_CHECK:
                edit_instructions.append("- Verify and correct any factual inaccuracies")
            elif edit_type == EditType.GRAMMAR:
                edit_instructions.append("- Fix grammar, punctuation, and spelling errors")
            elif edit_type == EditType.STYLE:
                edit_instructions.append("- Improve style consistency and professional presentation")
        
        if instructions:
            edit_instructions.append(f"- {instructions}")
        
        prompt = f"""Edit and improve the following content:

{content}

Editing requirements:
{chr(10).join(edit_instructions)}

Maintain the core message and structure while making improvements.

Return the edited content only."""
        
        return prompt
    
    def _get_tone_guidelines(self, tone: str) -> str:
        """Get guidelines for specific tone adjustments."""
        tone_guidelines = {
            "professional": """- Use formal language and industry terminology
- Remove colloquialisms and casual expressions
- Maintain objective, fact-based presentation
- Use third person where appropriate""",
            
            "casual": """- Use conversational language
- Include relatable examples and analogies
- Use contractions and everyday expressions
- Address reader directly (you, your)""",
            
            "authoritative": """- Use confident, declarative statements
- Include data, statistics, and expert references
- Remove hedging language (might, perhaps, maybe)
- Demonstrate expertise and command of subject""",
            
            "friendly": """- Use warm, approachable language
- Include encouraging phrases
- Use inclusive pronouns (we, our)
- Add personable touches without being unprofessional""",
            
            "persuasive": """- Use compelling language and strong verbs
- Include calls to action
- Emphasize benefits and value
- Build logical arguments with evidence"""
        }
        
        return tone_guidelines.get(tone.lower(), 
            "- Adjust language and style appropriately\n- Maintain consistency throughout")


class ContentAnalyzer:
    """Analyze content for improvement opportunities."""
    
    @staticmethod
    def analyze_readability(content: str) -> Dict[str, Any]:
        """Analyze content readability metrics."""
        import textstat
        
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(content),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(content),
            "gunning_fog": textstat.gunning_fog(content),
            "automated_readability_index": textstat.automated_readability_index(content),
            "coleman_liau_index": textstat.coleman_liau_index(content),
            "difficult_words": textstat.difficult_words(content),
            "avg_sentence_length": textstat.avg_sentence_length(content),
            "avg_syllables_per_word": textstat.avg_syllables_per_word(content)
        }
    
    @staticmethod
    def analyze_seo(content: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze content for SEO optimization."""
        content_lower = content.lower()
        word_count = len(content.split())
        
        keyword_analysis = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            
            keyword_analysis[keyword] = {
                "count": count,
                "density": round(density, 2),
                "in_first_100_words": keyword_lower in ' '.join(content.split()[:100]).lower()
            }
        
        # Check for headers
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        
        return {
            "word_count": word_count,
            "keyword_analysis": keyword_analysis,
            "headers_count": len(headers),
            "headers": headers[:5],  # First 5 headers
            "meta_description_length": len(content.split('\n')[0]) if content else 0
        }
    
    @staticmethod
    def suggest_improvements(content: str, config: ArticleConfig) -> List[Dict[str, Any]]:
        """Suggest improvements based on content analysis."""
        suggestions = []
        
        # Analyze readability
        readability = ContentAnalyzer.analyze_readability(content)
        
        if readability["flesch_kincaid_grade"] > 12:
            suggestions.append({
                "type": EditType.READABILITY,
                "reason": f"Content grade level is {readability['flesch_kincaid_grade']:.1f}, which may be too complex",
                "target_grade_level": 10
            })
        
        if readability["avg_sentence_length"] > 25:
            suggestions.append({
                "type": EditType.CLARITY,
                "reason": f"Average sentence length is {readability['avg_sentence_length']:.1f} words, which is quite long"
            })
        
        # Check SEO if keywords provided
        if config.content.keywords:
            seo_analysis = ContentAnalyzer.analyze_seo(content, config.content.keywords)
            
            for keyword, stats in seo_analysis["keyword_analysis"].items():
                if stats["density"] < 0.5:
                    suggestions.append({
                        "type": EditType.SEO,
                        "reason": f"Keyword '{keyword}' has low density ({stats['density']}%)",
                        "keywords": [keyword]
                    })
        
        # Check word count against target
        word_count = len(content.split())
        target_count = config.content.word_count.get("target", 1500)
        
        if abs(word_count - target_count) > target_count * 0.2:
            suggestions.append({
                "type": EditType.LENGTH,
                "reason": f"Content is {word_count} words, target is {target_count}",
                "target_word_count": target_count
            })
        
        return suggestions