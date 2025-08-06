"""Smart link suggestion engine for contextual outbound linking."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from drafty.services.embeddings import EmbeddingsService
from drafty.services.crawler import ContentCrawler


class LinkSuggestionEngine:
    """Engine for suggesting relevant outbound links."""
    
    # Authority domains (higher weight)
    AUTHORITY_DOMAINS = [
        "wikipedia.org", "arxiv.org", "nature.com", "sciencedirect.com",
        "pubmed.ncbi.nlm.nih.gov", "scholar.google.com", "jstor.org",
        "ieee.org", "acm.org", "springer.com", "wiley.com",
        "harvard.edu", "mit.edu", "stanford.edu", "oxford.edu",
        "nytimes.com", "wsj.com", "reuters.com", "bbc.com",
        "github.com", "stackoverflow.com", "medium.com"
    ]
    
    def __init__(
        self,
        embeddings_service: Optional[EmbeddingsService] = None,
        crawler: Optional[ContentCrawler] = None
    ):
        """Initialize link suggestion engine.
        
        Args:
            embeddings_service: Embeddings service for semantic matching
            crawler: Content crawler with knowledge base
        """
        self.embeddings = embeddings_service or EmbeddingsService()
        self.crawler = crawler or ContentCrawler(self.embeddings)
    
    def suggest_links(
        self,
        article_content: str,
        sources: List[Dict[str, Any]],
        max_links: int = 10,
        min_relevance: float = 0.6,
        prefer_authority: bool = True,
        diversity_threshold: float = 0.3,
        link_density: float = 2.5  # Links per 1000 words
    ) -> List[Dict[str, Any]]:
        """Suggest relevant outbound links for article.
        
        Args:
            article_content: The article text
            sources: Available sources to link to
            max_links: Maximum number of links to suggest
            min_relevance: Minimum relevance score
            prefer_authority: Prefer authoritative sources
            diversity_threshold: Threshold for topic diversity
            link_density: Target links per 1000 words
        
        Returns:
            List of link suggestions with metadata
        """
        # Calculate target link count based on density
        word_count = len(article_content.split())
        target_links = int((word_count / 1000) * link_density)
        max_links = min(max_links, target_links)
        
        # Split article into sections
        sections = self._split_into_sections(article_content)
        
        # Prepare source embeddings
        source_texts = []
        source_metadata = []
        
        for source in sources:
            # Create text representation of source
            title = source.get('title', '')
            snippet = source.get('snippet', source.get('description', ''))  # Also check 'description' field
            content = source.get('content', '')[:500] if source.get('content') else ''
            text = f"{title} {snippet} {content}".strip()
            
            # Skip empty sources
            if not text:
                continue
                
            source_texts.append(text)
            
            # Calculate authority score
            authority_score = self._calculate_authority_score(source.get("url", ""))
            
            source_metadata.append({
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "snippet": source.get("snippet", ""),
                "domain": urlparse(source.get("url", "")).netloc,
                "authority_score": authority_score,
                "published_date": source.get("published_date"),
                "source_type": source.get("type", "web")
            })
        
        if not source_texts:
            print(f"Warning: No valid source texts found from {len(sources)} sources")
            return []
        
        print(f"Embedding {len(source_texts)} source texts for link suggestions")
        
        # Embed sources
        source_embeddings = self.embeddings.embed_text(source_texts)
        
        # Find diverse, relevant links
        suggestions = []
        used_domains = set()
        
        for i, section in enumerate(sections):
            if len(suggestions) >= max_links:
                break
            
            # Find relevant sources for this section
            results = self.embeddings.find_diverse_items(
                section["content"],
                source_embeddings,
                source_metadata,
                top_k=3,  # Max 3 links per section
                diversity_threshold=diversity_threshold,
                relevance_weight=0.7 if prefer_authority else 0.8
            )
            
            for idx, score, cluster_id in results:
                if len(suggestions) >= max_links:
                    break
                
                if score < min_relevance:
                    continue
                
                metadata = source_metadata[idx]
                
                # Avoid too many links from same domain
                domain = metadata["domain"]
                if domain in used_domains and len(used_domains) > 3:
                    continue
                
                # Calculate final score
                final_score = self._calculate_final_score(
                    relevance_score=score,
                    authority_score=metadata["authority_score"],
                    prefer_authority=prefer_authority
                )
                
                # Find best anchor text
                anchor_text = self._find_anchor_text(section["content"], metadata["title"])
                
                # Find optimal insertion point
                insertion_point = self._find_insertion_point(
                    section["content"],
                    anchor_text,
                    metadata["snippet"]
                )
                
                suggestions.append({
                    "url": metadata["url"],
                    "title": metadata["title"],
                    "domain": domain,
                    "section_index": i,
                    "section_title": section["title"],
                    "anchor_text": anchor_text,
                    "insertion_point": insertion_point,
                    "relevance_score": score,
                    "authority_score": metadata["authority_score"],
                    "final_score": final_score,
                    "cluster_id": cluster_id,
                    "link_type": self._classify_link_type(metadata)
                })
                
                used_domains.add(domain)
        
        # Sort by final score
        suggestions.sort(key=lambda x: x["final_score"], reverse=True)
        
        return suggestions[:max_links]
    
    def enhance_with_citations(
        self,
        article_content: str,
        link_suggestions: List[Dict[str, Any]],
        citation_style: str = "inline"
    ) -> str:
        """Enhance article with suggested links and citations.
        
        Args:
            article_content: Original article content
            link_suggestions: Suggested links to insert
            citation_style: Citation style (inline, footnote, endnote)
        
        Returns:
            Enhanced article content
        """
        # Sort suggestions by position (reverse to maintain positions)
        suggestions_by_position = sorted(
            link_suggestions,
            key=lambda x: (x["section_index"], x.get("insertion_point", {}).get("position", 0)),
            reverse=True
        )
        
        # Split into sections
        sections = self._split_into_sections(article_content)
        
        # Insert links into sections
        for suggestion in suggestions_by_position:
            section_idx = suggestion["section_index"]
            if section_idx >= len(sections):
                continue
            
            section = sections[section_idx]
            content = section["content"]
            
            if citation_style == "inline":
                # Insert as markdown link
                anchor = suggestion["anchor_text"]
                url = suggestion["url"]
                link_text = f"[{anchor}]({url})"
                
                # Find position to insert
                insertion = suggestion.get("insertion_point", {})
                if insertion.get("position") is not None:
                    pos = insertion["position"]
                    # Insert after the sentence containing position
                    sentences = content.split(". ")
                    for i, sentence in enumerate(sentences):
                        if pos < len(sentence):
                            sentences[i] = sentence[:pos] + link_text + sentence[pos:]
                            break
                        pos -= len(sentence) + 2  # Account for ". "
                    
                    section["content"] = ". ".join(sentences)
                else:
                    # Append to end of section
                    section["content"] += f" {link_text}"
            
            elif citation_style == "footnote":
                # Add footnote reference
                footnote_num = len([s for s in suggestions_by_position if s["section_index"] <= section_idx])
                section["content"] += f"[^{footnote_num}]"
        
        # Reconstruct article
        enhanced_content = []
        for section in sections:
            if section["title"] and section["title"] != "Introduction":
                enhanced_content.append(f"## {section['title']}\n")
            enhanced_content.append(section["content"])
        
        # Add citations section if using footnotes
        if citation_style == "footnote":
            enhanced_content.append("\n\n## References\n")
            for i, suggestion in enumerate(link_suggestions, 1):
                enhanced_content.append(
                    f"[^{i}]: [{suggestion['title']}]({suggestion['url']})\n"
                )
        
        return "\n\n".join(enhanced_content)
    
    def _split_into_sections(self, content: str) -> List[Dict[str, str]]:
        """Split article content into sections."""
        sections = []
        lines = content.split("\n")
        
        current_section = {
            "title": "Introduction",
            "content": []
        }
        
        for line in lines:
            if line.startswith("#"):
                # Save previous section
                if current_section["content"]:
                    current_section["content"] = "\n".join(current_section["content"])
                    sections.append(current_section)
                
                # Start new section
                title = line.lstrip("#").strip()
                current_section = {
                    "title": title,
                    "content": []
                }
            else:
                current_section["content"].append(line)
        
        # Add last section
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)
        
        return sections
    
    def _calculate_authority_score(self, url: str) -> float:
        """Calculate authority score for a URL."""
        if not url:
            return 0.5
        
        domain = urlparse(url).netloc.lower()
        
        # Check if it's an authority domain
        for auth_domain in self.AUTHORITY_DOMAINS:
            if auth_domain in domain:
                return 0.9
        
        # Check for educational domains
        if any(edu in domain for edu in [".edu", ".ac.uk", ".edu.au"]):
            return 0.85
        
        # Check for government domains
        if ".gov" in domain:
            return 0.85
        
        # Check for organization domains
        if ".org" in domain:
            return 0.7
        
        # Default score
        return 0.5
    
    def _calculate_final_score(
        self,
        relevance_score: float,
        authority_score: float,
        prefer_authority: bool = True
    ) -> float:
        """Calculate final link score."""
        if prefer_authority:
            # 60% relevance, 40% authority
            return (relevance_score * 0.6) + (authority_score * 0.4)
        else:
            # 80% relevance, 20% authority
            return (relevance_score * 0.8) + (authority_score * 0.2)
    
    def _find_anchor_text(self, section_content: str, link_title: str) -> str:
        """Find appropriate anchor text for a link."""
        # Clean link title
        anchor = link_title.lower()
        
        # Look for relevant phrases in section
        words = section_content.lower().split()
        title_words = anchor.split()
        
        # Find matching phrases
        for i in range(len(words) - len(title_words) + 1):
            phrase = " ".join(words[i:i+len(title_words)])
            similarity = len(set(phrase.split()) & set(title_words)) / len(title_words)
            
            if similarity > 0.5:
                # Found matching phrase in content
                # Get original case
                start_pos = section_content.lower().find(phrase)
                if start_pos != -1:
                    return section_content[start_pos:start_pos+len(phrase)]
        
        # Fallback: use shortened title
        if len(title_words) > 5:
            return " ".join(title_words[:4]) + "..."
        
        return link_title
    
    def _find_insertion_point(
        self,
        section_content: str,
        anchor_text: str,
        link_snippet: str
    ) -> Dict[str, Any]:
        """Find optimal point to insert a link."""
        # Try to find anchor text in content
        pos = section_content.lower().find(anchor_text.lower())
        
        if pos != -1:
            return {
                "position": pos,
                "type": "exact_match",
                "confidence": 1.0
            }
        
        # Find most relevant sentence
        sentences = section_content.split(". ")
        best_sentence_idx = 0
        best_score = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Calculate relevance to link
            score = self.embeddings.calculate_similarity(
                self.embeddings.embed_text(sentence),
                self.embeddings.embed_text(link_snippet)
            )
            
            if score > best_score:
                best_score = score
                best_sentence_idx = i
        
        # Calculate position at end of best sentence
        position = sum(len(s) + 2 for s in sentences[:best_sentence_idx + 1])
        
        return {
            "position": position,
            "type": "semantic_match",
            "confidence": best_score
        }
    
    def _classify_link_type(self, metadata: Dict[str, Any]) -> str:
        """Classify the type of link."""
        url = metadata.get("url", "").lower()
        title = metadata.get("title", "").lower()
        source_type = metadata.get("source_type", "")
        
        if source_type:
            return source_type
        
        # Classification based on URL and title
        if any(term in url for term in ["research", "study", "paper", "journal"]):
            return "research"
        elif any(term in url for term in ["news", "article", "post", "blog"]):
            return "article"
        elif any(term in url for term in ["tool", "app", "software", "github"]):
            return "tool"
        elif any(term in url for term in ["guide", "tutorial", "how-to"]):
            return "guide"
        elif any(term in url for term in ["wiki", "definition", "glossary"]):
            return "reference"
        elif any(term in title for term in ["case study", "example", "story"]):
            return "case_study"
        else:
            return "general"
    
    def analyze_link_distribution(
        self,
        link_suggestions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the distribution of suggested links.
        
        Args:
            link_suggestions: List of link suggestions
        
        Returns:
            Analysis of link distribution
        """
        if not link_suggestions:
            return {
                "total_links": 0,
                "sections_covered": 0,
                "link_types": {},
                "authority_distribution": {},
                "domain_diversity": 0
            }
        
        # Count by section
        sections_covered = len(set(s["section_index"] for s in link_suggestions))
        
        # Count by type
        link_types = {}
        for suggestion in link_suggestions:
            link_type = suggestion.get("link_type", "general")
            link_types[link_type] = link_types.get(link_type, 0) + 1
        
        # Authority distribution
        high_authority = sum(1 for s in link_suggestions if s["authority_score"] > 0.7)
        medium_authority = sum(1 for s in link_suggestions if 0.5 <= s["authority_score"] <= 0.7)
        low_authority = sum(1 for s in link_suggestions if s["authority_score"] < 0.5)
        
        # Domain diversity
        unique_domains = len(set(s["domain"] for s in link_suggestions))
        
        return {
            "total_links": len(link_suggestions),
            "sections_covered": sections_covered,
            "link_types": link_types,
            "authority_distribution": {
                "high": high_authority,
                "medium": medium_authority,
                "low": low_authority
            },
            "domain_diversity": unique_domains,
            "average_relevance": sum(s["relevance_score"] for s in link_suggestions) / len(link_suggestions),
            "average_authority": sum(s["authority_score"] for s in link_suggestions) / len(link_suggestions)
        }
    
    def validate_links(self, links: List[str]) -> List[Dict[str, Any]]:
        """Validate that links are working.
        
        Args:
            links: List of URLs to validate
        
        Returns:
            Validation results
        """
        import asyncio
        from drafty.utils.http import HTTPClient
        
        async def check_link(url: str) -> Dict[str, Any]:
            try:
                async with HTTPClient() as client:
                    response = await client.head(url, follow_redirects=True)
                    return {
                        "url": url,
                        "status": response.status_code,
                        "valid": 200 <= response.status_code < 400,
                        "redirect": str(response.url) if str(response.url) != url else None
                    }
            except Exception as e:
                return {
                    "url": url,
                    "status": 0,
                    "valid": False,
                    "error": str(e)
                }
        
        # Run validation
        loop = asyncio.get_event_loop()
        tasks = [check_link(url) for url in links]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        return results