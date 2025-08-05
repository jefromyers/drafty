"""Configuration schemas for Drafty using Pydantic v2."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ToneEnum(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"
    TECHNICAL = "technical"


class IntentEnum(str, Enum):
    INFORM = "inform"
    EDUCATE = "educate"
    CONVERT = "convert"
    ENTERTAIN = "entertain"
    PERSUADE = "persuade"


class OutlineStyleEnum(str, Enum):
    HUB = "hub"
    HOWTO = "howto"
    COMPARISON = "comparison"
    LISTICLE = "listicle"
    GUIDE = "guide"
    REVIEW = "review"


class LLMProviderEnum(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class ArticleStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class WordCount(BaseModel):
    min: int = Field(default=1000, ge=100)
    max: int = Field(default=3000, le=10000)

    @field_validator("max")
    @classmethod
    def validate_max(cls, v: int, info) -> int:
        if "min" in info.data and v < info.data["min"]:
            raise ValueError("max must be greater than min")
        return v


class MetaConfig(BaseModel):
    slug: str = Field(..., pattern=r"^[a-z0-9-]+$")
    version: str = Field(default="1.0")
    status: ArticleStatus = Field(default=ArticleStatus.DRAFT)
    created: Optional[str] = None
    updated: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ContentConfig(BaseModel):
    topic: str = Field(..., min_length=3)
    audience: str = Field(..., min_length=3)
    tone: List[ToneEnum] = Field(default_factory=lambda: [ToneEnum.PROFESSIONAL])
    intent: IntentEnum = Field(default=IntentEnum.INFORM)
    word_count: WordCount = Field(default_factory=WordCount)
    keywords: List[str] = Field(default_factory=list)
    forbidden_phrases: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ResearchConfig(BaseModel):
    strategies: List[str] = Field(
        default_factory=lambda: ["web_search", "competitor_analysis"]
    )
    seed_queries: List[str] = Field(default_factory=list)
    required_sources: List[Dict[str, str]] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=list)
    max_sources: int = Field(default=10, ge=1, le=50)
    enable_serp_analysis: bool = Field(default=True)
    extra: Dict[str, Any] = Field(default_factory=dict)


class StructureConfig(BaseModel):
    outline_style: OutlineStyleEnum = Field(default=OutlineStyleEnum.GUIDE)
    fixed_sections: List[str] = Field(default_factory=list)
    dynamic_sections: bool = Field(default=True)
    max_heading_depth: int = Field(default=3, ge=1, le=6)
    include_toc: bool = Field(default=True)
    section_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class LinkConfig(BaseModel):
    internal: List[Dict[str, str]] = Field(default_factory=list)
    external: List[Dict[str, str]] = Field(default_factory=list)
    auto_link_keywords: Dict[str, str] = Field(default_factory=dict)
    link_density: Dict[str, float] = Field(
        default_factory=lambda: {"min": 2.0, "max": 5.0}
    )
    nofollow_external: bool = Field(default=False)
    link_attributes: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class QualityConfig(BaseModel):
    readability_target: int = Field(default=8, ge=1, le=18)
    seo_score_min: int = Field(default=80, ge=0, le=100)
    plagiarism_threshold: int = Field(default=10, ge=0, le=100)
    fact_check: bool = Field(default=True)
    grammar_check: bool = Field(default=True)
    inclusive_language: bool = Field(default=True)
    custom_checks: List[Dict[str, Any]] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class LLMProviderConfig(BaseModel):
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    json_mode: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None, exclude=True)
    base_url: Optional[str] = None
    timeout: int = Field(default=60, ge=1)
    extra: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)
    default: LLMProviderEnum = Field(default=LLMProviderEnum.OPENAI)
    fallback_order: List[LLMProviderEnum] = Field(default_factory=list)
    prompt_templates: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class NLPConfig(BaseModel):
    spacy_model: str = Field(default="en_core_web_sm")
    enable_ner: bool = Field(default=True)
    enable_pos: bool = Field(default=True)
    enable_dependency: bool = Field(default=True)
    custom_entities: List[str] = Field(default_factory=list)
    pipelines: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ScrapingConfig(BaseModel):
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0)
    user_agent: str = Field(default="Drafty/1.0 (AI Writing Assistant)")
    respect_robots_txt: bool = Field(default=True)
    max_concurrent: int = Field(default=5, ge=1)
    headers: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExportConfig(BaseModel):
    formats: List[str] = Field(default_factory=lambda: ["markdown", "html"])
    include_metadata: bool = Field(default=True)
    include_toc: bool = Field(default=True)
    syntax_highlighting: bool = Field(default=True)
    custom_templates: Dict[str, str] = Field(default_factory=dict)
    output_dir: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ArticleConfig(BaseModel):
    """Main configuration model for an article with extensible extra fields."""

    meta: MetaConfig
    content: ContentConfig
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    linking: LinkConfig = Field(default_factory=LinkConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    plugins: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "ArticleConfig":
        """Load configuration from a JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: Path) -> None:
        """Save configuration to a JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    def get_extra(self, section: str, key: str, default: Any = None) -> Any:
        """Get a value from the extra field of a section."""
        section_obj = getattr(self, section, None)
        if section_obj and hasattr(section_obj, "extra"):
            return section_obj.extra.get(key, default)
        return default

    def set_extra(self, section: str, key: str, value: Any) -> None:
        """Set a value in the extra field of a section."""
        section_obj = getattr(self, section, None)
        if section_obj and hasattr(section_obj, "extra"):
            section_obj.extra[key] = value

    class Config:
        use_enum_values = True
        extra = "allow"  # Allow additional fields at the root level