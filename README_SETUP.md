# Setting Up Your API Keys

## Quick Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   # Open in your editor
   nano .env
   # or
   vim .env
   # or
   code .env
   ```

3. **Check your configuration:**
   ```bash
   python -m drafty.cli.main config --status
   ```

## Required Keys

### For Basic Operation
At minimum, you need ONE of these LLM providers:
- **OpenAI**: `OPENAI_API_KEY=sk-...`
- **Anthropic**: `ANTHROPIC_API_KEY=sk-ant-...`
- **Gemini**: `GEMINI_API_KEY=AIza...` or `GOOGLE_API_KEY=AIza...`

### For Enhanced Features
- **Data4SEO** (for real SERP data):
  - `DATA4SEO_USERNAME=your_username`
  - `DATA4SEO_PASSWORD=your_password`
  - Get credentials at: https://dataforseo.com/

- **Browserless** (for JavaScript rendering):
  - Default: `http://localhost:3000`
  - Run with Docker: `docker-compose up browserless`

## Where to Get API Keys

1. **OpenAI**
   - Sign up at: https://platform.openai.com/
   - Create API key: https://platform.openai.com/api-keys
   - Pricing: https://openai.com/pricing

2. **Anthropic (Claude)**
   - Sign up at: https://console.anthropic.com/
   - Create API key in Console settings
   - Pricing: https://www.anthropic.com/pricing

3. **Google Gemini**
   - Get API key: https://makersuite.google.com/app/apikey
   - Free tier available
   - Documentation: https://ai.google.dev/

4. **Data4SEO**
   - Sign up: https://app.dataforseo.com/register
   - Dashboard: https://app.dataforseo.com/
   - API docs: https://docs.dataforseo.com/

## Environment File Locations

Drafty looks for `.env` files in this order:
1. Current directory (`./.env`)
2. User home directory (`~/.drafty/.env`)
3. Project root directory

## Security Notes

- **NEVER** commit your `.env` file to version control
- The `.gitignore` file already excludes `.env`
- Use different API keys for development and production
- Rotate keys regularly
- Set spending limits on your API accounts

## Testing Your Setup

After adding your keys, verify everything works:

```bash
# Check configuration
python -m drafty.cli.main config --status

# Test with a simple research task
cd test-article
python -m drafty.cli.main research --max-sources 3

# Test specific provider
python -m drafty.cli.main research --provider gemini
```

## Troubleshooting

If keys aren't being recognized:
1. Check the `.env` file is in the right location
2. Ensure no extra spaces around the `=` sign
3. Check for quotes (not needed unless value has spaces)
4. Try exporting directly: `export OPENAI_API_KEY=sk-...`

## Default Settings

You can also set defaults in `.env`:
```bash
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-1.5-flash
DEFAULT_TEMPERATURE=0.7
```