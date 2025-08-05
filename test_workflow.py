#!/usr/bin/env python3
"""Test the complete Drafty workflow."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

# Add drafty to path
sys.path.insert(0, str(Path(__file__).parent))

from drafty.core.config import ArticleConfig, MetaConfig, ContentConfig
from drafty.core.workspace import Workspace
from drafty.services.research import ResearchService
from drafty.services.outline import OutlineService
from drafty.services.draft import DraftService
from drafty.services.export import ExportService


async def test_workflow():
    """Test the complete article generation workflow."""
    print("=" * 60)
    print("DRAFTY WORKFLOW TEST")
    print("=" * 60)
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    print("\n📋 Configuration Status:")
    print(f"  OpenAI: {'✓' if has_openai else '✗'}")
    print(f"  Gemini: {'✓' if has_gemini else '✗'}")
    print(f"  Anthropic: {'✓' if has_anthropic else '✗'}")
    
    if not (has_openai or has_gemini):
        print("\n❌ No LLM providers configured. Please set API keys in .env")
        return False
    
    # Determine which provider to use
    provider = "openai" if has_openai else "gemini"
    print(f"\n🤖 Using provider: {provider}")
    
    # Clean up test workspace if it exists
    test_workspace_path = Path("test-workflow-article")
    if test_workspace_path.exists():
        shutil.rmtree(test_workspace_path)
    
    try:
        # Step 1: Create workspace
        print("\n1️⃣ Creating workspace...")
        workspace = Workspace.create(
            slug="test-workflow-article",
            topic="The Future of AI Writing Tools",
            audience="Content creators and marketers"
        )
        print(f"   ✓ Created workspace at: {workspace.base_path}")
        
        # Load config
        config = workspace.get_config()
        
        # Configure LLM provider
        if provider == "openai":
            config.llm.default = "openai"
            config.llm.providers["openai"] = {
                "model": "gpt-4o-mini",
                "json_mode": True
            }
        else:
            config.llm.default = "gemini"
            config.llm.providers["gemini"] = {
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            }
        
        workspace.save_config(config)
        
        # Step 2: Research (simplified - no actual web scraping)
        print("\n2️⃣ Conducting research...")
        research_service = ResearchService(config)
        
        # Just do topic analysis for testing
        topic_analysis = await research_service.analyze_topic(provider)
        
        # Save research data
        research_file = workspace.research_dir / "topic_analysis.json"
        research_file.write_text(json.dumps(topic_analysis, indent=2))
        print(f"   ✓ Research complete: {len(topic_analysis.get('key_concepts', []))} key concepts")
        
        # Step 3: Generate outline
        print("\n3️⃣ Generating outline...")
        outline_service = OutlineService(config)
        
        outline = await outline_service.generate_outline(
            research_data={"topic_analysis": topic_analysis},
            provider_name=provider,
            sections=3  # Keep it short for testing
        )
        
        # Save outline
        outline_file = workspace.drafts_dir / "outline.json"
        outline_file.write_text(json.dumps(outline.to_dict(), indent=2))
        print(f"   ✓ Outline generated: {len(outline.sections)} sections")
        
        # Step 4: Generate draft
        print("\n4️⃣ Generating draft...")
        draft_service = DraftService(config)
        
        # Generate only introduction for faster testing
        draft = await draft_service.generate_draft(
            outline=outline,
            research_data={"topic_analysis": topic_analysis},
            provider_name=provider,
            sections_to_generate=[outline.introduction.heading]
        )
        
        # Save draft
        draft_content = draft.to_markdown()
        workspace.save_draft(draft_content)
        print(f"   ✓ Draft generated: {draft.total_word_count} words")
        
        # Step 5: Export
        print("\n5️⃣ Exporting article...")
        export_service = ExportService(config)
        
        # Export as markdown
        md_content = export_service.export_markdown(draft_content, clean=True)
        md_file = workspace.exports_dir / "article.md"
        md_file.write_text(md_content)
        
        # Export as HTML
        html_content = export_service.export_html(draft_content)
        html_file = workspace.exports_dir / "article.html"
        html_file.write_text(html_content)
        
        print(f"   ✓ Exported to: {workspace.exports_dir}")
        
        # Success!
        print("\n✅ Workflow test completed successfully!")
        print(f"\nGenerated files:")
        print(f"  📁 {workspace.base_path}")
        print(f"  📄 {workspace.exports_dir / 'article.md'}")
        print(f"  🌐 {workspace.exports_dir / 'article.html'}")
        
        # Show sample of generated content
        print("\n📝 Sample of generated content:")
        print("-" * 40)
        print(draft_content[:500] + "...")
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the workflow test."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the test
    success = asyncio.run(test_workflow())
    
    # Clean up test workspace
    test_workspace_path = Path("test-workflow-article")
    if test_workspace_path.exists() and success:
        print("\n🧹 Cleaning up test workspace...")
        shutil.rmtree(test_workspace_path)
        print("   ✓ Cleanup complete")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()