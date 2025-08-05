#!/usr/bin/env python3
"""Test script for Drafty functionality."""

import asyncio
import json
from pathlib import Path

from drafty.core.config import ArticleConfig, MetaConfig, ContentConfig
from drafty.core.workspace import Workspace
from drafty.utils.scraper import scrape_url
from drafty.providers.base import LLMProviderFactory, LLMMessage


def test_config():
    """Test configuration creation."""
    print("Testing configuration...")
    
    config = ArticleConfig(
        meta=MetaConfig(slug="test-article"),
        content=ContentConfig(
            topic="Testing Drafty CLI",
            audience="Developers",
        )
    )
    
    print(f"✓ Created config for: {config.meta.slug}")
    print(f"  Topic: {config.content.topic}")
    print(f"  Audience: {config.content.audience}")
    return config


def test_workspace():
    """Test workspace creation."""
    print("\nTesting workspace...")
    
    # Clean up if exists
    test_path = Path.cwd() / "test-workspace"
    if test_path.exists():
        import shutil
        shutil.rmtree(test_path)
    
    workspace = Workspace.create(
        slug="test-workspace",
        topic="Test Article",
        audience="Test Audience",
    )
    
    print(f"✓ Created workspace at: {workspace.base_path}")
    
    # Check structure
    assert workspace.config_file.exists()
    assert workspace.research_dir.exists()
    assert workspace.drafts_dir.exists()
    
    # Test saving draft
    workspace.save_draft("# Test Draft\n\nThis is a test.")
    current = workspace.get_current_draft()
    assert "Test Draft" in current
    
    print("✓ Workspace structure verified")
    return workspace


async def test_scraper():
    """Test web scraping functionality."""
    print("\nTesting web scraper...")
    
    # Test with a simple website
    url = "https://example.com"
    
    try:
        result = await scrape_url(
            url,
            extract_links=True,
            extract_images=False,
            output_format="markdown"
        )
        
        print(f"✓ Scraped {url}")
        print(f"  Title: {result['metadata'].get('title', 'N/A')}")
        print(f"  Word count: {result['word_count']}")
        print(f"  Links found: {len(result.get('links', []))}")
        
        return result
    except Exception as e:
        print(f"⚠ Scraping failed: {e}")
        return None


async def test_llm_provider():
    """Test LLM provider (requires API key)."""
    print("\nTesting LLM providers...")
    
    # Check if we have any API keys
    import os
    
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found")
        try:
            provider = LLMProviderFactory.create("openai", {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
            })
            
            messages = [
                LLMMessage(role="system", content="You are a helpful assistant."),
                LLMMessage(role="user", content="Say 'Hello, Drafty!' in exactly 3 words."),
            ]
            
            response = await provider.generate(messages)
            print(f"  Response: {response.content}")
            
            return response
        except Exception as e:
            print(f"  ⚠ OpenAI test failed: {e}")
    else:
        print("⚠ No OpenAI API key found")
    
    if os.getenv("GEMINI_API_KEY"):
        print("✓ Gemini API key found")
        # Add Gemini test here if needed
    else:
        print("⚠ No Gemini API key found")
    
    return None


def test_pydantic_extras():
    """Test the extra fields functionality."""
    print("\nTesting extra fields...")
    
    config = ArticleConfig(
        meta=MetaConfig(
            slug="test-extras",
            extra={"custom_field": "custom_value", "priority": "high"}
        ),
        content=ContentConfig(
            topic="Testing Extras",
            audience="Developers",
            extra={"research_depth": "comprehensive", "citations_required": True}
        )
    )
    
    # Test getting extras
    assert config.meta.extra["custom_field"] == "custom_value"
    assert config.content.extra["citations_required"] == True
    
    # Test helper methods
    config.set_extra("meta", "new_field", "new_value")
    assert config.get_extra("meta", "new_field") == "new_value"
    
    print("✓ Extra fields working correctly")
    print(f"  Meta extras: {config.meta.extra}")
    print(f"  Content extras: {config.content.extra}")
    
    return config


async def main():
    """Run all tests."""
    print("=" * 50)
    print("DRAFTY TEST SUITE")
    print("=" * 50)
    
    # Test configuration
    config = test_config()
    
    # Test workspace
    workspace = test_workspace()
    
    # Test extra fields
    test_pydantic_extras()
    
    # Test scraper
    await test_scraper()
    
    # Test LLM provider
    await test_llm_provider()
    
    print("\n" + "=" * 50)
    print("✓ All basic tests completed!")
    print("=" * 50)
    
    # Cleanup
    test_path = Path.cwd() / "test-workspace"
    if test_path.exists():
        import shutil
        shutil.rmtree(test_path)
        print("✓ Cleaned up test workspace")


if __name__ == "__main__":
    asyncio.run(main())