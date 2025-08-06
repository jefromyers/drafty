"""Research service for gathering and analyzing sources."""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from base64 import b64encode

from drafty.core.config import ArticleConfig
from drafty.providers.base import LLMProviderFactory, LLMMessage
from drafty.services.templates import PromptBuilder
from drafty.utils.http import HTTPClient
from drafty.utils.scraper import scrape_multiple


class Data4SEOClient:
    """Client for Data4SEO API."""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize Data4SEO client.
        
        Args:
            username: Data4SEO username (or from env DATA4SEO_USERNAME)
            password: Data4SEO password (or from env DATA4SEO_PASSWORD)
        """
        self.username = username or os.getenv("DATA4SEO_USERNAME")
        self.password = password or os.getenv("DATA4SEO_PASSWORD")
        self.base_url = "https://api.dataforseo.com/v3"
        
        if not self.username or not self.password:
            raise ValueError("Data4SEO credentials not provided")
        
        # Create auth header
        credentials = f"{self.username}:{self.password}"
        self.auth_header = f"Basic {b64encode(credentials.encode()).decode()}"
    
    async def search_google(
        self,
        keyword: str,
        location: str = "United States",
        language: str = "en",
        num_results: int = 10,
        include_serp_features: bool = True
    ) -> Dict[str, Any]:
        """Search Google using Data4SEO Live API.
        
        Args:
            keyword: Search query
            location: Target location
            language: Language code
            num_results: Number of results to return
            include_serp_features: Include SERP features like PAA, featured snippets
        """
        async with HTTPClient() as client:
            # Use the LIVE endpoint for immediate results
            task_data = [{
                "keyword": keyword,
                "location_name": location,
                "language_code": language,
                "depth": num_results
            }]
            
            # Use the live/regular endpoint for immediate results
            response = await client.post(
                f"{self.base_url}/serp/google/organic/live/regular",
                json=task_data,
                headers={"Authorization": self.auth_header}
            )
            
            result = response.json()
            
            # Debug logging
            if result.get("status_code") == 20000:
                if result.get("tasks") and len(result["tasks"]) > 0:
                    task = result["tasks"][0]
                    if task.get("result") and len(task["result"]) > 0:
                        items = task["result"][0].get("items", [])
                        print(f"    Data4SEO Live API returned {len(items)} items for '{keyword}'")
                    else:
                        print(f"    Data4SEO Live API returned no results for '{keyword}'")
                else:
                    print(f"    Data4SEO Live API task structure issue for '{keyword}'")
            else:
                print(f"Data4SEO Live API failed: {result.get('status_message', 'Unknown error')}")
                print(f"Status code: {result.get('status_code')}")
            
            return result
    
    async def get_serp_features(self, keyword: str) -> Dict[str, Any]:
        """Get SERP features for a keyword (People Also Ask, Related Searches, etc.)."""
        result = await self.search_google(keyword, include_serp_features=True, num_results=5)
        
        features = {
            "people_also_ask": [],
            "related_searches": [],
            "featured_snippet": None,
            "knowledge_graph": None
        }
        
        if result.get("tasks") and result["tasks"][0].get("result"):
            serp_data = result["tasks"][0]["result"][0]
            
            # Extract People Also Ask
            if "people_also_ask" in serp_data:
                features["people_also_ask"] = serp_data["people_also_ask"]
            
            # Extract Related Searches
            if "related_searches" in serp_data:
                features["related_searches"] = serp_data["related_searches"]
            
            # Extract Featured Snippet
            if "featured_snippet" in serp_data:
                features["featured_snippet"] = serp_data["featured_snippet"]
            
            # Extract Knowledge Graph
            if "knowledge_graph" in serp_data:
                features["knowledge_graph"] = serp_data["knowledge_graph"]
        
        return features


class ResearchService:
    """Service for conducting research on article topics."""

    def __init__(self, config: ArticleConfig):
        """Initialize research service."""
        self.config = config
        self.prompt_builder = PromptBuilder()
        
        # Initialize Data4SEO if credentials available
        try:
            self.data4seo = Data4SEOClient()
            print("Data4SEO client initialized successfully")
        except ValueError as e:
            self.data4seo = None
            print(f"Data4SEO not available: {e}")
        except Exception as e:
            self.data4seo = None
            print(f"Error initializing Data4SEO: {e}")
        
    async def analyze_topic(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the topic and generate research queries.
        
        Args:
            provider_name: LLM provider to use (defaults to config default)
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
        
        # Build research prompt
        prompt = self.prompt_builder.build_research_prompt(
            topic=self.config.content.topic,
            audience=self.config.content.audience,
            queries=self.config.research.seed_queries,
        )
        
        # Generate research analysis
        messages = [
            LLMMessage(role="system", content="You are a research assistant helping to plan article research."),
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
                return response.as_json()
            else:
                return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: extract what we can from text
            return self._parse_text_response(response.content)
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse research analysis from text response."""
        # Simple extraction logic - in production, use better parsing
        lines = text.split('\n')
        result = {
            "key_concepts": [],
            "important_questions": [],
            "research_gaps": [],
            "recommended_source_types": [],
            "search_queries": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if 'key concept' in line.lower():
                current_section = "key_concepts"
            elif 'question' in line.lower():
                current_section = "important_questions"
            elif 'gap' in line.lower():
                current_section = "research_gaps"
            elif 'source' in line.lower():
                current_section = "recommended_source_types"
            elif 'quer' in line.lower() or 'search' in line.lower():
                current_section = "search_queries"
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                if current_section:
                    result[current_section].append(line.lstrip('-•*').strip())
            elif line.startswith('"') and current_section == "search_queries":
                # Handle quoted search queries
                result[current_section].append(line.strip('"'))
        
        # If no search queries found, generate some based on the topic
        if not result["search_queries"] and hasattr(self, 'config'):
            topic = self.config.content.topic
            result["search_queries"] = [
                topic,
                f"how to {topic.lower()}",
                f"best practices {topic.lower()}",
                f"{topic.lower()} guide",
                f"{topic.lower()} tips"
            ]
        
        return result
    
    async def search_web(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search the web for relevant sources using Data4SEO or fallback.
        
        Args:
            queries: Search queries to use
        """
        results = []
        
        if self.data4seo:
            # Use Data4SEO for real SERP data
            for query in queries[:5]:  # Limit to avoid too many API calls
                # Clean up query - remove trailing commas, quotes, etc.
                clean_query = query.strip().rstrip('",').strip('"\'')
                print(f"Searching for: '{clean_query}'")
                
                try:
                    serp_result = await self.data4seo.search_google(clean_query, num_results=20)  # Get more results per query
                    
                    # Debug: Log the response structure
                    if serp_result.get("status_code") != 20000:
                        print(f"  API Error: {serp_result.get('status_message', 'Unknown')}")
                    
                    if serp_result.get("tasks") and len(serp_result["tasks"]) > 0:
                        task = serp_result["tasks"][0]
                        if task.get("result") and len(task["result"]) > 0:
                            items = task["result"][0].get("items", [])
                            print(f"  Found {len(items)} results for '{clean_query}'")
                            
                            for item in items:
                                results.append({
                                    "query": clean_query,
                                    "url": item.get("url"),
                                    "title": item.get("title"),
                                    "snippet": item.get("description"),
                                    "position": item.get("rank_group"),
                                    "domain": item.get("domain"),
                                    "breadcrumb": item.get("breadcrumb"),
                                    "is_featured": item.get("is_featured_snippet", False),
                                    "relevance": 0.9 - (item.get("rank_group", 1) * 0.05)  # Higher rank = more relevant
                                })
                
                except Exception as e:
                    print(f"Data4SEO search failed for '{query}': {e}")
                    # Add fallback result
                    results.append({
                        "query": query,
                        "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                        "title": f"Search results for {query}",
                        "snippet": f"Fallback result for {query}",
                        "relevance": 0.5
                    })
        else:
            # Fallback: Generate conceptual sources without actual scraping
            print("Data4SEO not configured - generating conceptual sources")
            
            # For demo purposes, we'll generate conceptual sources based on the queries
            # In production, you'd want to use Data4SEO or another search API
            for i, query in enumerate(queries[:10], 1):  # Default to 10 sources
                results.append({
                    "query": query,
                    "url": f"https://example-source-{i}.com/article",
                    "title": f"Article: {query.title()}",
                    "snippet": f"This would be a comprehensive article about {query}. The content would cover key aspects, best practices, and expert insights.",
                    "relevance": 0.9 - (i * 0.05),
                    "domain": f"example-source-{i}.com",
                    "note": "Demo source - actual URL scraping requires Data4SEO API or manual URL input"
                })
        
        return results
    
    async def get_serp_analysis(self, keyword: str) -> Dict[str, Any]:
        """Get comprehensive SERP analysis including PAA, related searches, etc.
        
        Args:
            keyword: Main keyword to analyze
        """
        if not self.data4seo:
            return {
                "error": "Data4SEO not configured",
                "people_also_ask": [],
                "related_searches": []
            }
        
        try:
            features = await self.data4seo.get_serp_features(keyword)
            return features
        except Exception as e:
            print(f"SERP analysis failed: {e}")
            return {
                "error": str(e),
                "people_also_ask": [],
                "related_searches": []
            }
    
    async def scrape_sources(
        self, 
        urls: List[str],
        use_javascript: bool = False
    ) -> List[Dict[str, Any]]:
        """Scrape content from URLs.
        
        Args:
            urls: URLs to scrape
            use_javascript: Whether to use Browserless for JS rendering
        """
        return await scrape_multiple(
            urls,
            max_concurrent=self.config.scraping.max_concurrent,
            use_javascript=use_javascript,
            extract_links=True,
            clean_content=True,
            output_format="markdown"
        )
    
    async def analyze_sources(
        self,
        sources: List[Dict[str, Any]],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze scraped sources for relevance and key information.
        
        Args:
            sources: Scraped source data
            provider_name: LLM provider to use
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
        
        # Analyze each source
        analyzed = []
        for source in sources:
            if 'error' in source:
                continue
                
            # Build analysis prompt
            prompt = f"""Analyze this source for an article about "{self.config.content.topic}":

URL: {source['url']}
Title: {source.get('metadata', {}).get('title', 'Unknown')}

Content Preview:
{source.get('content', '')[:1000]}

Provide:
1. Relevance score (0-10)
2. Key information extracted
3. Useful quotes
4. How it helps the article

Respond in JSON format.
"""
            
            messages = [
                LLMMessage(role="system", content="You are analyzing sources for article research."),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await provider.generate(messages, json_mode=True)
            
            try:
                analysis = response.as_json() or json.loads(response.content)
            except:
                analysis = {"relevance": 5, "notes": "Could not analyze"}
            
            analyzed.append({
                "source": source,
                "analysis": analysis
            })
        
        return {
            "total_sources": len(sources),
            "analyzed": len(analyzed),
            "sources": analyzed
        }
    
    async def conduct_research(
        self,
        max_sources: int = 10,
        use_javascript: bool = False,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct full research process.
        
        Args:
            max_sources: Maximum number of sources to gather
            use_javascript: Whether to use JS rendering
            provider_name: LLM provider to use
        """
        results = {
            "topic_analysis": None,
            "serp_analysis": None,
            "search_results": [],
            "scraped_sources": [],
            "analysis": None
        }
        
        # Step 1: Analyze topic
        print("Analyzing topic...")
        results["topic_analysis"] = await self.analyze_topic(provider_name)
        
        # Step 2: Get SERP analysis for main topic
        print("Analyzing SERP features...")
        results["serp_analysis"] = await self.get_serp_analysis(self.config.content.topic)
        
        # Step 3: Search for sources
        queries = results["topic_analysis"].get("search_queries", [])[:5]
        if queries:
            print(f"Searching for sources using {len(queries)} queries...")
            print(f"Queries: {queries}")  # Debug: show what queries we're using
            results["search_results"] = await self.search_web(queries)
            print(f"Found {len(results.get('search_results', []))} search results")  # Debug
        else:
            print("No search queries generated from topic analysis")
            # Fallback: use the topic itself as a query
            results["search_results"] = await self.search_web([self.config.content.topic])
        
        # Step 4: Scrape top sources (skip if using demo sources)
        urls = [r["url"] for r in results["search_results"][:max_sources] if r.get("url")]
        # Skip actual scraping for demo/example sources
        if urls and not any("example-source" in url for url in urls):
            print(f"Scraping {len(urls)} sources...")
            results["scraped_sources"] = await self.scrape_sources(urls, use_javascript)
        elif urls:
            print("Skipping web scraping for demo sources")
        
        # Step 5: Analyze sources
        if results["scraped_sources"]:
            print("Analyzing sources...")
            results["analysis"] = await self.analyze_sources(
                results["scraped_sources"],
                provider_name
            )
        
        return results