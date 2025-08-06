"""Enhanced content crawler for deep source analysis."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from drafty.utils.scraper import ContentExtractor, scrape_url
from drafty.utils.http import HTTPClient
from drafty.services.embeddings import EmbeddingsService


class ContentCrawler:
    """Deep content crawler for extracting structured information from sources."""
    
    def __init__(
        self,
        embeddings_service: Optional[EmbeddingsService] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize content crawler.
        
        Args:
            embeddings_service: Optional embeddings service for semantic analysis
            cache_dir: Directory for caching crawled content
        """
        self.embeddings = embeddings_service or EmbeddingsService()
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".drafty" / "crawler_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base for crawled content
        self.knowledge_base: List[Dict[str, Any]] = []
    
    async def deep_crawl(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 10,
        follow_internal: bool = True,
        use_javascript: bool = False
    ) -> Dict[str, Any]:
        """Perform deep crawl of a source.
        
        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            follow_internal: Whether to follow internal links
            use_javascript: Use JS rendering
        
        Returns:
            Crawled content with structure
        """
        crawled = {}
        to_crawl = [(url, 0)]  # (url, depth)
        crawled_urls = set()
        base_domain = urlparse(url).netloc
        
        while to_crawl and len(crawled) < max_pages:
            current_url, depth = to_crawl.pop(0)
            
            if current_url in crawled_urls or depth > max_depth:
                continue
            
            crawled_urls.add(current_url)
            
            # Scrape page
            try:
                page_data = await scrape_url(
                    current_url,
                    use_javascript=use_javascript,
                    extract_links=True,
                    extract_images=True,
                    clean_content=True,
                    output_format="markdown"
                )
                
                # Extract structured sections
                sections = self._extract_sections(page_data)
                
                # Store crawled data
                crawled[current_url] = {
                    "url": current_url,
                    "depth": depth,
                    "metadata": page_data.get("metadata", {}),
                    "content": page_data.get("content", ""),
                    "sections": sections,
                    "headings": page_data.get("headings", []),
                    "word_count": page_data.get("word_count", 0),
                    "faqs": page_data.get("faqs", []),
                    "tables": page_data.get("tables", []),
                    "structured_data": page_data.get("structured_data", []),
                    "crawled_at": datetime.now().isoformat()
                }
                
                # Add to knowledge base
                self._add_to_knowledge_base(crawled[current_url])
                
                # Find internal links to crawl
                if follow_internal and depth < max_depth:
                    for link in page_data.get("links", []):
                        link_url = link.get("url", "")
                        link_domain = urlparse(link_url).netloc
                        
                        if link_domain == base_domain and link_url not in crawled_urls:
                            to_crawl.append((link_url, depth + 1))
                
            except Exception as e:
                print(f"Failed to crawl {current_url}: {e}")
                crawled[current_url] = {
                    "url": current_url,
                    "error": str(e),
                    "depth": depth
                }
        
        return {
            "base_url": url,
            "pages_crawled": len(crawled),
            "max_depth_reached": max(p.get("depth", 0) for p in crawled.values()),
            "pages": crawled
        }
    
    def _extract_sections(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured sections from page content.
        
        Args:
            page_data: Scraped page data
        
        Returns:
            List of extracted sections
        """
        sections = []
        content = page_data.get("content", "")
        headings = page_data.get("headings", [])
        
        if not content:
            return sections
        
        # Split content by headings
        lines = content.split("\n")
        current_section = {
            "title": "Introduction",
            "level": 0,
            "content": [],
            "type": "introduction"
        }
        
        for line in lines:
            # Check if line is a heading
            if line.startswith("#"):
                # Save previous section
                if current_section["content"]:
                    current_section["content"] = "\n".join(current_section["content"]).strip()
                    if current_section["content"]:
                        current_section["type"] = self._classify_section(
                            current_section["title"],
                            current_section["content"]
                        )
                        sections.append(current_section)
                
                # Start new section
                level = len(line.split()[0])  # Count #'s
                title = line.lstrip("#").strip()
                current_section = {
                    "title": title,
                    "level": level,
                    "content": [],
                    "type": "content"
                }
            else:
                current_section["content"].append(line)
        
        # Add last section
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"]).strip()
            if current_section["content"]:
                current_section["type"] = self._classify_section(
                    current_section["title"],
                    current_section["content"]
                )
                sections.append(current_section)
        
        return sections
    
    def _classify_section(self, title: str, content: str) -> str:
        """Classify section type based on title and content.
        
        Args:
            title: Section title
            content: Section content
        
        Returns:
            Section type
        """
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Classification rules
        if any(term in title_lower for term in ["introduction", "overview", "summary", "abstract"]):
            return "introduction"
        elif any(term in title_lower for term in ["method", "approach", "how to", "steps"]):
            return "methodology"
        elif any(term in title_lower for term in ["result", "finding", "outcome", "performance"]):
            return "findings"
        elif any(term in title_lower for term in ["example", "case study", "demo", "illustration"]):
            return "examples"
        elif any(term in title_lower for term in ["statistic", "data", "number", "metric"]):
            return "statistics"
        elif any(term in title_lower for term in ["conclusion", "summary", "takeaway", "final"]):
            return "conclusion"
        elif any(term in title_lower for term in ["reference", "citation", "source", "bibliography"]):
            return "references"
        elif any(term in title_lower for term in ["faq", "question", "q&a"]):
            return "faq"
        elif "%" in content or any(term in content_lower for term in ["percent", "increase", "decrease"]):
            return "statistics"
        else:
            return "content"
    
    def _add_to_knowledge_base(self, page_data: Dict[str, Any]) -> None:
        """Add crawled page to knowledge base.
        
        Args:
            page_data: Crawled page data
        """
        # Extract key information
        url = page_data.get("url", "")
        title = page_data.get("metadata", {}).get("title", "")
        sections = page_data.get("sections", [])
        
        # Create knowledge entries for each section
        for section in sections:
            if section.get("content"):
                entry = {
                    "source_url": url,
                    "source_title": title,
                    "section_title": section.get("title", ""),
                    "section_type": section.get("type", "content"),
                    "content": section.get("content", ""),
                    "level": section.get("level", 0),
                    "metadata": {
                        "crawled_at": page_data.get("crawled_at", ""),
                        "word_count": len(section.get("content", "").split()),
                        "depth": page_data.get("depth", 0)
                    }
                }
                
                # Add embedding if available
                if self.embeddings:
                    entry["embedding"] = self.embeddings.embed_text(entry["content"])
                
                self.knowledge_base.append(entry)
    
    async def extract_quotes(
        self,
        url: str,
        min_length: int = 50,
        max_length: int = 300
    ) -> List[Dict[str, str]]:
        """Extract quotable content from a source.
        
        Args:
            url: URL to extract quotes from
            min_length: Minimum quote length
            max_length: Maximum quote length
        
        Returns:
            List of extracted quotes
        """
        # Scrape content
        page_data = await scrape_url(url, output_format="text")
        content = page_data.get("content", "")
        
        quotes = []
        sentences = content.split(".")
        
        for sentence in sentences:
            sentence = sentence.strip()
            length = len(sentence)
            
            if min_length <= length <= max_length:
                # Check if it's a good quote (has substance)
                if any(indicator in sentence.lower() for indicator in [
                    "according to", "research shows", "studies",
                    "found that", "discovered", "revealed",
                    "important", "significant", "key",
                    "%" 
                ]):
                    quotes.append({
                        "text": sentence + ".",
                        "source_url": url,
                        "source_title": page_data.get("metadata", {}).get("title", ""),
                        "length": length
                    })
        
        return quotes
    
    async def extract_statistics(self, url: str) -> List[Dict[str, Any]]:
        """Extract statistics and data points from a source.
        
        Args:
            url: URL to extract statistics from
        
        Returns:
            List of extracted statistics
        """
        import re
        
        # Scrape content
        page_data = await scrape_url(url)
        content = page_data.get("content", "")
        
        statistics = []
        
        # Patterns for finding statistics
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)',  # Dollar amounts
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s+(million|billion|thousand)',  # Large numbers
            r'(\d+(?:\.\d+)?)[xX]\s+(?:increase|growth|faster|slower)',  # Multipliers
            r'(\d+)\s+out of\s+(\d+)',  # Ratios
        ]
        
        lines = content.split("\n")
        for line in lines:
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Get context (surrounding text)
                    start = max(0, match.start() - 50)
                    end = min(len(line), match.end() + 50)
                    context = line[start:end].strip()
                    
                    statistics.append({
                        "value": match.group(0),
                        "context": context,
                        "source_url": url,
                        "source_title": page_data.get("metadata", {}).get("title", ""),
                        "type": self._classify_statistic(match.group(0))
                    })
        
        # Also extract from tables
        for table in page_data.get("tables", []):
            for row in table.get("rows", []):
                for cell in row:
                    for pattern in patterns[:3]:  # Check first 3 patterns
                        if re.search(pattern, str(cell)):
                            statistics.append({
                                "value": cell,
                                "context": f"Table: {', '.join(table.get('headers', []))}",
                                "source_url": url,
                                "source_title": page_data.get("metadata", {}).get("title", ""),
                                "type": "table_data"
                            })
        
        return statistics
    
    def _classify_statistic(self, value: str) -> str:
        """Classify type of statistic."""
        if "%" in value:
            return "percentage"
        elif "$" in value:
            return "monetary"
        elif any(term in value.lower() for term in ["million", "billion", "thousand"]):
            return "large_number"
        elif "x" in value.lower():
            return "multiplier"
        else:
            return "numeric"
    
    def search_knowledge_base(
        self,
        query: str,
        top_k: int = 10,
        section_types: Optional[List[str]] = None,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant content.
        
        Args:
            query: Search query
            top_k: Number of results
            section_types: Filter by section types
            min_similarity: Minimum similarity threshold
        
        Returns:
            Relevant knowledge entries
        """
        if not self.knowledge_base:
            return []
        
        # Filter by section type if specified
        search_base = self.knowledge_base
        if section_types:
            search_base = [
                entry for entry in self.knowledge_base
                if entry.get("section_type") in section_types
            ]
        
        if not search_base:
            return []
        
        # Get embeddings
        embeddings = []
        for entry in search_base:
            if "embedding" in entry:
                embeddings.append(entry["embedding"])
            else:
                # Generate embedding if missing
                embedding = self.embeddings.embed_text(entry["content"])
                entry["embedding"] = embedding
                embeddings.append(embedding)
        
        import numpy as np
        embeddings = np.array(embeddings)
        
        # Search
        results = self.embeddings.semantic_search(
            query,
            embeddings,
            top_k=top_k,
            threshold=min_similarity
        )
        
        # Return entries with scores
        matched_entries = []
        for idx, score in results:
            entry = search_base[idx].copy()
            entry["relevance_score"] = score
            matched_entries.append(entry)
        
        return matched_entries
    
    def find_supporting_evidence(
        self,
        claim: str,
        evidence_types: List[str] = ["statistics", "findings", "examples"]
    ) -> List[Dict[str, Any]]:
        """Find evidence to support a claim.
        
        Args:
            claim: The claim to support
            evidence_types: Types of evidence to find
        
        Returns:
            Supporting evidence from knowledge base
        """
        return self.search_knowledge_base(
            claim,
            top_k=5,
            section_types=evidence_types,
            min_similarity=0.6
        )
    
    def build_topic_graph(self) -> Dict[str, List[str]]:
        """Build a graph of interconnected topics from knowledge base.
        
        Returns:
            Topic graph as adjacency list
        """
        if not self.knowledge_base:
            return {}
        
        # Extract all section titles
        titles = [entry["section_title"] for entry in self.knowledge_base if entry.get("section_title")]
        
        if not titles:
            return {}
        
        # Group similar topics
        topic_groups = self.embeddings.group_by_topic(titles, similarity_threshold=0.5)
        
        # Build graph
        graph = {}
        for cluster_id, indices in topic_groups.items():
            cluster_titles = [titles[i] for i in indices]
            
            # Use most common or longest title as representative
            representative = max(cluster_titles, key=len)
            
            # Find related topics (other clusters)
            related = []
            for other_cluster, other_indices in topic_groups.items():
                if other_cluster != cluster_id:
                    other_titles = [titles[i] for i in other_indices]
                    other_rep = max(other_titles, key=len)
                    
                    # Check similarity between cluster representatives
                    sim = self.embeddings.calculate_similarity(
                        self.embeddings.embed_text(representative),
                        self.embeddings.embed_text(other_rep)
                    )
                    
                    if sim > 0.3:  # Related topics
                        related.append(other_rep)
            
            graph[representative] = related
        
        return graph
    
    def save_knowledge_base(self, path: Path) -> None:
        """Save knowledge base to disk.
        
        Args:
            path: Path to save knowledge base
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove embeddings for JSON serialization
        save_data = []
        embeddings = []
        
        for entry in self.knowledge_base:
            entry_copy = entry.copy()
            if "embedding" in entry_copy:
                embeddings.append(entry_copy.pop("embedding"))
            else:
                embeddings.append(None)
            save_data.append(entry_copy)
        
        # Save JSON data
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        # Save embeddings separately
        if any(e is not None for e in embeddings):
            import numpy as np
            import pickle
            
            embeddings_path = path.with_suffix(".embeddings.pkl")
            with open(embeddings_path, "wb") as f:
                pickle.dump(np.array([e for e in embeddings if e is not None]), f)
    
    def load_knowledge_base(self, path: Path) -> None:
        """Load knowledge base from disk.
        
        Args:
            path: Path to knowledge base file
        """
        path = Path(path)
        
        # Load JSON data
        with open(path) as f:
            self.knowledge_base = json.load(f)
        
        # Try to load embeddings
        embeddings_path = path.with_suffix(".embeddings.pkl")
        if embeddings_path.exists():
            import pickle
            with open(embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
                
                # Re-attach embeddings
                emb_idx = 0
                for entry in self.knowledge_base:
                    if emb_idx < len(embeddings):
                        entry["embedding"] = embeddings[emb_idx]
                        emb_idx += 1