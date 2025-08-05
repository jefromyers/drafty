# Drafty - AI Writing Assistant CLI

A powerful, modular CLI tool for AI-assisted article drafting with support for multiple LLM providers, intelligent web scraping, and structured content generation.

## Features

- 🤖 **Multiple LLM Providers**: OpenAI (with structured outputs & JSON mode), Google Gemini, Anthropic Claude (coming soon)
- 🔍 **Smart Web Scraping**: Using trafilatura and selectolax for fast, accurate content extraction
- 📈 **Real SERP Data**: Data4SEO integration for Google search results, People Also Ask, and related searches
- 🚀 **JavaScript Rendering**: Browserless integration for dynamic content
- 📝 **Complete Workflow**: Research → Outline → Draft → Edit → Export
- ✏️ **Advanced Editing**: Multi-mode content refinement (readability, SEO, tone, length, clarity)
- 📊 **Content Analysis**: Readability metrics, SEO analysis, and improvement suggestions
- 🎨 **Template System**: Jinja2-based templates for customizable outputs
- 📁 **Version Control**: Automatic draft versioning and workspace management
- 🔧 **Extensible Config**: Pydantic v2 schemas with extra fields support

## Requirements

- Python 3.11+
- API keys for at least one LLM provider (OpenAI, Gemini, or Anthropic)
- Optional: Data4SEO account for real SERP data
- Optional: Docker for JavaScript rendering support

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd drafty

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

# Download spaCy model (optional, for NLP features)
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Set up environment variables

Create a `.env` file:

```bash
# LLM Providers
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key  # or GOOGLE_API_KEY
ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Custom model selection
GEMINI_MODEL=gemini-2.0-flash-exp  # Default Gemini model

# Optional: Data4SEO (for real SERP data)
DATA4SEO_USERNAME=your_data4seo_username
DATA4SEO_PASSWORD=your_data4seo_password

# Optional: Browserless (for JavaScript rendering)
BROWSERLESS_URL=http://localhost:3000
```

### 2. Create a new article

```bash
# Initialize a new article workspace
drafty new my-article --topic "AI Writing Tools" --audience "Content creators"

# Navigate to the workspace
cd my-article/
```

### 3. Complete workflow

```bash
# Check configuration status
drafty config status

# Gather research sources
drafty research --max-sources 10

# Generate outline (3-5 sections by default)
drafty outline --sections 5 --style guide

# Create draft
drafty draft --model gpt-4o-mini

# Edit and refine content
drafty edit all --analyze  # Analyze and improve all aspects
drafty edit readability --target-grade 10  # Adjust readability
drafty edit seo --keywords "AI writing, content automation"
drafty edit length --target-words 2000  # Adjust word count

# Export final version
drafty export --format markdown html json
```

## Project Structure

```
my-article/
├── article.json       # Configuration file
├── research/         
│   ├── sources.json   # Collected sources
│   └── notes.md      # Manual notes
├── drafts/           
│   ├── current.md    # Working draft
│   └── v*.md         # Version history
├── exports/          # Final outputs
└── .drafty/          # Internal metadata
```

## Configuration

Articles are configured via `article.json`:

```json
{
  "meta": {
    "slug": "my-article",
    "status": "draft",
    "tags": ["AI", "automation"],
    "extra": {
      "author": "Your Name",
      "client": "Optional Client Name"
    }
  },
  "content": {
    "topic": "AI Writing Tools",
    "audience": "Content creators",
    "tone": "professional",
    "word_count": {
      "min": 1500,
      "max": 2500,
      "target": 2000
    },
    "keywords": ["AI writing", "content automation", "GPT"],
    "style": "guide"
  },
  "llm": {
    "providers": {
      "openai": {
        "model": "gpt-4o-mini",
        "json_mode": true
      },
      "gemini": {
        "model": "gemini-2.0-flash-exp"
      }
    },
    "default": "openai"
  },
  "research": {
    "max_sources": 10,
    "use_serp": true,
    "seed_queries": ["AI writing tools 2024", "content automation best practices"]
  }
}
```

## Advanced Features

### JavaScript Rendering

For sites with dynamic content, use the Browserless container:

```bash
# Start Browserless container
docker-compose up -d browserless

# Scrape with JS rendering
drafty research --use-javascript
```

### Structured Outputs (OpenAI)

The OpenAI provider supports structured outputs for reliable JSON responses:

```python
from drafty.providers.openai import OpenAIProvider
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    sections: List[str]
    keywords: List[str]

provider = OpenAIProvider({"model": "gpt-4o-mini"})
result = await provider.generate_with_schema(messages, Article)
```

### Content Analysis & Editing

```bash
# Analyze content quality
drafty edit --analyze  # Shows readability, SEO, and improvement suggestions

# Batch editing with multiple improvements
drafty edit all  # Apply clarity, readability, grammar, and style fixes

# Specific editing modes
drafty edit readability --target-grade 8  # Target 8th grade reading level
drafty edit seo --keywords "keyword1,keyword2"  # SEO optimization
drafty edit tone --tone casual  # Adjust tone (professional, casual, friendly, etc.)
drafty edit length --target-words 1500  # Expand or condense to target length
drafty edit clarity  # Improve clarity and remove ambiguity
drafty edit grammar  # Fix grammar and punctuation
drafty edit style  # Improve consistency and professional presentation
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `drafty new <slug>` | Create new article workspace |
| `drafty config status` | Check API keys and configuration |
| `drafty research` | Gather and analyze sources |
| `drafty outline` | Generate article structure |
| `drafty draft` | Create content sections |
| `drafty edit <type>` | Edit and refine content (see editing modes above) |
| `drafty export` | Generate final outputs (markdown, HTML, JSON) |
| `drafty status` | View workspace status and draft versions |

## Development

```bash
# Install in development mode
pip install -e .

# Run workflow test
python test_workflow.py

# Install dev dependencies (when available)
pip install -e ".[dev]"

# Run tests (when available)
pytest

# Format code
black drafty/
ruff check drafty/

# Type checking
mypy drafty/
```

## Docker Support

```yaml
# docker-compose.yml
services:
  browserless:
    image: ghcr.io/browserless/chromium
    environment:
      CONCURRENT: 5
      QUEUED: 10
      TIMEOUT: 45000
    ports:
      - 3000:3000
```

## Architecture

- **Modular Design**: Each phase (research, outline, draft) is independent
- **File-Based**: All data stored in workspace for version control
- **Extensible**: Plugin system for custom processors
- **Async First**: Built on httpx for concurrent operations
- **Type Safe**: Pydantic v2 models throughout

## Key Technologies

- **CLI Framework**: Click for command-line interface
- **LLM Integration**: OpenAI, Google Generative AI SDKs
- **HTTP Client**: httpx for async requests
- **Web Scraping**: trafilatura + selectolax for content extraction
- **Data Models**: Pydantic v2 for validation and configuration
- **Templates**: Jinja2 for prompt and output templates
- **Text Analysis**: textstat for readability metrics
- **Markdown Processing**: python-markdown for format conversion
- **Environment Management**: python-dotenv for configuration
- **Output Formatting**: Rich for beautiful terminal output

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT

## Current Status

### ✅ Completed Features
- Multiple LLM provider support (OpenAI, Gemini)
- Complete article generation pipeline
- Advanced editing and refinement tools
- Content analysis and SEO optimization
- Multi-format export (Markdown, HTML, JSON)
- Automatic draft versioning
- Configuration management with .env support

### 🚧 In Progress
- Anthropic Claude integration
- spaCy NLP integration for entity recognition
- Link management with NER

### 📋 Roadmap
- [ ] Ollama integration for local models
- [ ] Advanced fact-checking
- [ ] Multi-language support
- [ ] Web UI for review
- [ ] GitHub Actions integration
- [ ] Plugin marketplace
- [ ] Real-time collaboration features
- [ ] Content scheduling and publishing