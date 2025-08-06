# Drafty - AI Writing Assistant CLI

A powerful, modular CLI tool for AI-assisted article drafting with support for multiple LLM providers, intelligent web scraping, and structured content generation.

## Features

- 🤖 **Multiple LLM Providers**: OpenAI (with structured outputs & JSON mode), Google Gemini, Anthropic Claude (coming soon)
- 🔍 **Smart Web Scraping**: Using trafilatura and selectolax for fast, accurate content extraction
- 📈 **Real SERP Data**: Data4SEO integration for Google search results, People Also Ask, and related searches
- 🚀 **One-Command Generation**: Automated workflow from topic to finished article
- 📝 **Complete Workflow**: Research → Outline → Draft → Edit → Export
- 🔗 **Smart Linking System**: AI-powered outbound link suggestions with semantic relevance
- 📚 **Citation Management**: Automatic citation generation (APA, MLA, Chicago, Harvard)
- 🧠 **Semantic Search**: Embeddings-based content matching for contextual linking
- 🕷️ **Deep Content Crawling**: Extract structured information from sources
- ✏️ **Advanced Editing**: Multi-mode content refinement (readability, SEO, tone, length, clarity)
- 📊 **Content Analysis**: Readability metrics, SEO analysis, and improvement suggestions
- 🎨 **Template System**: Jinja2-based templates for customizable outputs
- 📁 **Version Control**: Automatic draft versioning and workspace management
- 🔧 **JSON Config Support**: Define complete article specs in reusable JSON files
- ⚡ **Batch Processing**: Generate multiple articles from config files

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

### 2. Generate an article with one command

```bash
# Simplest usage - generate a complete article
drafty generate "The Future of AI in Healthcare" --audience "Medical professionals"

# With more options
drafty generate "Python Best Practices" \
  --audience "Python developers" \
  --keywords "Python,coding standards,best practices" \
  --sections 5 \
  --word-count 2000

# Specify where to save final exports (workspace in temp)
drafty generate "AI Ethics" \
  --output-dir ./articles/ai-ethics/ \
  --export-formats markdown,html

# With smart linking enabled
drafty generate "AI Writing Tools Guide" \
  --enhance-links \
  --max-links 15 \
  --link-density 3.0 \
  --include-bibliography

# Using a JSON config file
drafty generate --config article-config.json

# Dry run to preview what will happen
drafty generate "Test Topic" --dry-run

# Use a permanent workspace directory
drafty generate "Machine Learning Guide" \
  --workspace-dir ./workspaces/ml-guide/ \
  --output-dir ./published/

# Keep the temporary workspace for debugging
drafty generate "Debug Article" \
  --output-dir ./output/ \
  --keep-workspace

# Save workspace to specific location after generation
drafty generate "Research Paper" \
  --output-dir ./final/ \
  --save-workspace ./saved-workspaces/
```

### 3. Or use the step-by-step workflow

```bash
# Create workspace
drafty new my-article --topic "AI Writing Tools" --audience "Content creators"
cd my-article/

# Run individual steps
drafty research --max-sources 10
drafty outline --sections 5
drafty draft
drafty edit all
drafty export -f markdown -f html
```

## Workspace Management

### Workspace vs Output Directory

Drafty separates the **workspace** (where it stores research, drafts, and intermediate files) from the **output directory** (where final exports are saved):

- **Workspace**: Contains the full article structure with research/, drafts/, exports/, and metadata
- **Output Directory**: Receives only the final exported files (.md, .html, etc.)

### Workspace Options

```bash
# Default: Temporary workspace, exports to current directory
drafty generate "My Article"

# Separate output directory (workspace still in temp)
drafty generate "My Article" --output-dir ./published/

# Permanent workspace directory
drafty generate "My Article" --workspace-dir ./my-workspaces/article/

# Keep temporary workspace after completion
drafty generate "My Article" --keep-workspace

# Copy workspace to permanent location after generation
drafty generate "My Article" --save-workspace ./archived-workspaces/
```

## Project Structure

```
my-article/              # Workspace directory
├── article.json       # Configuration file
├── research/         
│   ├── sources.json   # Collected sources
│   └── notes.md      # Manual notes
├── drafts/           
│   ├── current.md    # Working draft
│   └── v*.md         # Version history
├── exports/          # Final outputs (if no --output-dir specified)
└── .drafty/          # Internal metadata

output-directory/      # Separate output location (with --output-dir)
├── article.md        # Final markdown export
└── article.html      # Final HTML export
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

## Automated Workflow (New!)

The `generate` command runs the entire article creation pipeline automatically:

### Basic Usage

```bash
# Generate a complete article with one command
drafty generate "Your Article Topic" --audience "Target Audience"
```

### JSON Configuration

Create reusable configuration files for consistent article generation:

```json
{
  "topic": "Complete Guide to AI Writing Tools",
  "audience": "Content creators and marketers",
  "keywords": ["AI writing", "content automation", "GPT"],
  "sections": 7,
  "word_count": 2500,
  "provider": "gemini",
  "style": "guide",
  "tone": "professional",
  "edit_types": ["all"],
  "export_formats": ["markdown", "html"],
  "output_dir": "./exports",
  "workspace_dir": "./workspaces/ai-tools",
  "keep_workspace": false
}
```

Use the config file:

```bash
drafty generate --config article-config.json

# Override config values from CLI
drafty generate --config base.json --topic "Different Topic" --sections 5
```

### Advanced Options

```bash
drafty generate "Article Topic" \
  --keywords "keyword1,keyword2,keyword3" \
  --sections 5 \
  --word-count 2000 \
  --provider gemini \
  --style guide \
  --tone professional \
  --edit-types all \
  --export-formats markdown,html,json \
  --output-dir ./my-articles \      # Final exports only
  --workspace-dir ./workspaces \     # Full workspace location
  --save-config my-config.json \     # Save config for reuse
  --verbose                           # Show detailed progress
```

### Workflow Control

```bash
# Skip specific steps
drafty generate "Quick Article" --skip-research --skip-edit

# Dry run to preview configuration
drafty generate "Test Article" --dry-run

# Save configuration for later use
drafty generate "My Article" --save-config saved-config.json
```

### Generate Command Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--output-dir` | Directory for final exports only (not workspace) | `--output-dir ./published/` |
| `--workspace-dir` | Directory for full workspace (default: temp) | `--workspace-dir ./workspaces/` |
| `--keep-workspace` | Preserve temporary workspace after completion | `--keep-workspace` |
| `--save-workspace` | Copy temp workspace to location after completion | `--save-workspace ./archived/` |
| `--config` | Load settings from JSON file | `--config article.json` |
| `--audience` | Target audience for the content | `--audience "Developers"` |
| `--keywords` | SEO keywords (comma-separated) | `--keywords "AI,ML,Python"` |
| `--sections` | Number of article sections | `--sections 5` |
| `--word-count` | Target word count | `--word-count 2000` |
| `--provider` | LLM provider to use | `--provider gemini` |
| `--style` | Article style (guide/howto/listicle) | `--style guide` |
| `--tone` | Writing tone | `--tone professional` |
| `--edit-types` | Edit modes to apply | `--edit-types clarity,seo` |
| `--export-formats` | Output formats | `--export-formats md,html` |
| `--skip-research` | Skip research phase | `--skip-research` |
| `--skip-edit` | Skip editing phase | `--skip-edit` |
| `--enhance-links` | Add smart outbound links | `--enhance-links` |
| `--dry-run` | Preview without executing | `--dry-run` |
| `--verbose` | Show detailed progress | `--verbose` |

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
| `drafty generate` | Generate complete article with one command (supports workspace management) |
| `drafty new <slug>` | Create new article workspace |
| `drafty config status` | Check API keys and configuration |
| `drafty research` | Gather and analyze sources |
| `drafty outline` | Generate article structure |
| `drafty draft` | Create content sections |
| `drafty edit <type>` | Edit and refine content (see editing modes above) |
| `drafty export` | Generate final outputs (markdown, HTML, JSON) |
| `drafty chat` | Interactive refinement mode (coming soon) |
| `drafty nlp` | Text analysis tools (coming soon) |

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
- **Automated Workflow**: One-command article generation with `drafty generate`
- **Workspace Management**: Separate workspace and output directories with flexible options
- **Smart Linking System**: RAG + DBSCAN clustering for contextual outbound links
- **Citation Management**: Automatic bibliography generation in multiple styles
- **Semantic Embeddings**: Sentence-transformers integration for content similarity
- **Deep Content Crawling**: Structured extraction from research sources
- **JSON Configuration**: Full workflow configuration via JSON files
- Multiple LLM provider support (OpenAI, Gemini)
- Data4SEO integration for real search data
- Complete article generation pipeline
- Advanced editing and refinement tools
- Content analysis and SEO optimization
- Multi-format export (Markdown, HTML, JSON, Text)
- Automatic draft versioning
- Temporary and permanent workspace options
- Workspace preservation and archiving
- Configuration management with .env support
- Dry-run mode for testing workflows

### 🚧 In Progress
- Anthropic Claude integration
- Interactive chat mode
- Batch article processing
- Advanced knowledge graph building
- Real-time link validation

### 📋 Roadmap
- [ ] Ollama integration for local models
- [ ] Advanced fact-checking
- [ ] Multi-language support
- [ ] Web UI for review
- [ ] GitHub Actions integration
- [ ] Plugin marketplace
- [ ] Real-time collaboration features
- [ ] Content scheduling and publishing