# Drafty - AI Writing Assistant CLI

A powerful, modular CLI tool for AI-assisted article drafting with support for multiple LLM providers, intelligent web scraping, and structured content generation.

## Features

- 🤖 **Multiple LLM Providers**: OpenAI (with structured outputs), Google Gemini, Anthropic Claude
- 🔍 **Smart Web Scraping**: Using trafilatura and selectolax for fast, accurate content extraction
- 🚀 **JavaScript Rendering**: Browserless integration for dynamic content
- 📝 **Flexible Workflows**: Research → Outline → Draft → Edit → Link → Export
- 🧠 **NLP Integration**: spaCy for entity recognition and text analysis
- 📊 **Structured Outputs**: JSON mode and schema validation for reliable responses
- 🎨 **Template System**: Jinja2-based templates for customizable outputs

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
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 2. Create a new article

```bash
# Initialize a new article workspace
drafty new my-article --topic "AI Writing Tools" --audience "Content creators"

# Navigate to the workspace
cd my-article/
```

### 3. Research and generate content

```bash
# Gather research sources
drafty research --max-sources 10

# Generate outline
drafty outline --style guide

# Create draft
drafty draft --model gpt-4o-mini

# Refine and optimize
drafty edit --readability 8 --seo

# Add smart links
drafty link --suggest --use-ner

# Export final version
drafty export --format markdown html
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
    "status": "draft"
  },
  "content": {
    "topic": "AI Writing Tools",
    "audience": "Content creators",
    "tone": ["professional", "approachable"],
    "word_count": {"min": 1500, "max": 2500}
  },
  "llm": {
    "providers": {
      "openai": {
        "model": "gpt-4o-mini",
        "json_mode": true
      },
      "gemini": {
        "model": "gemini-1.5-flash"
      }
    },
    "default": "openai"
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

### NLP Analysis

```bash
# Analyze content with spaCy
drafty nlp analyze --file drafts/current.md

# Extract entities
drafty nlp entities

# Check readability
drafty nlp readability
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `drafty new <slug>` | Create new article workspace |
| `drafty research` | Gather and analyze sources |
| `drafty outline` | Generate article structure |
| `drafty draft` | Create content sections |
| `drafty edit` | Refine and optimize |
| `drafty link` | Manage internal/external links |
| `drafty export` | Generate final outputs |
| `drafty chat` | Interactive refinement mode |
| `drafty nlp` | Text analysis tools |
| `drafty config` | View/edit configuration |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
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
- **Type Safe**: Pydantic models throughout

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT

## Roadmap

- [ ] Ollama integration for local models
- [ ] Advanced fact-checking
- [ ] Multi-language support
- [ ] Web UI for review
- [ ] GitHub Actions integration
- [ ] Plugin marketplace