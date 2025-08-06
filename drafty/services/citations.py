"""Citation management system for automatic citation generation."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


class CitationManager:
    """Manage citations and bibliography for articles."""
    
    def __init__(self):
        """Initialize citation manager."""
        self.citations = []
        self.citation_counter = 0
    
    def add_citation(
        self,
        url: str,
        title: str,
        author: Optional[str] = None,
        published_date: Optional[str] = None,
        accessed_date: Optional[str] = None,
        publisher: Optional[str] = None,
        citation_type: str = "web"
    ) -> Dict[str, Any]:
        """Add a citation to the bibliography.
        
        Args:
            url: Source URL
            title: Source title
            author: Author name(s)
            published_date: Publication date
            accessed_date: Access date
            publisher: Publisher/website name
            citation_type: Type of citation (web, article, book, etc.)
        
        Returns:
            Citation entry with ID
        """
        self.citation_counter += 1
        
        # Auto-generate accessed date if not provided
        if not accessed_date:
            accessed_date = datetime.now().strftime("%Y-%m-%d")
        
        # Extract publisher from URL if not provided
        if not publisher and url:
            domain = urlparse(url).netloc
            publisher = domain.replace("www.", "").split(".")[0].title()
        
        citation = {
            "id": self.citation_counter,
            "url": url,
            "title": title,
            "author": author or "Unknown",
            "published_date": published_date,
            "accessed_date": accessed_date,
            "publisher": publisher,
            "type": citation_type,
            "used": False  # Track if citation was used in text
        }
        
        self.citations.append(citation)
        return citation
    
    def format_citation(
        self,
        citation: Dict[str, Any],
        style: str = "apa"
    ) -> str:
        """Format a citation according to style guide.
        
        Args:
            citation: Citation data
            style: Citation style (apa, mla, chicago, harvard)
        
        Returns:
            Formatted citation string
        """
        if style.lower() == "apa":
            return self._format_apa(citation)
        elif style.lower() == "mla":
            return self._format_mla(citation)
        elif style.lower() == "chicago":
            return self._format_chicago(citation)
        elif style.lower() == "harvard":
            return self._format_harvard(citation)
        else:
            # Default simple format
            return self._format_simple(citation)
    
    def _format_apa(self, citation: Dict[str, Any]) -> str:
        """Format citation in APA style.
        
        Example: Author, A. (Year). Title. Publisher. URL
        """
        parts = []
        
        # Author
        author = citation.get("author", "Unknown")
        if author != "Unknown":
            # Format: Last, F.
            if "," in author:
                parts.append(author)
            else:
                # Try to split first and last name
                names = author.split()
                if len(names) >= 2:
                    parts.append(f"{names[-1]}, {names[0][0]}.")
                else:
                    parts.append(author)
        else:
            parts.append(citation.get("publisher", "Unknown"))
        
        # Year
        year = self._extract_year(citation.get("published_date"))
        parts.append(f"({year}).")
        
        # Title (italicized in real APA, but we'll use markdown)
        parts.append(f"*{citation.get('title', 'Untitled')}*.")
        
        # Publisher
        if citation.get("publisher") and citation.get("author") != "Unknown":
            parts.append(f"{citation['publisher']}.")
        
        # URL
        if citation.get("url"):
            parts.append(citation["url"])
        
        return " ".join(parts)
    
    def _format_mla(self, citation: Dict[str, Any]) -> str:
        """Format citation in MLA style.
        
        Example: Author. "Title." Publisher, Date. Web. Access Date.
        """
        parts = []
        
        # Author
        author = citation.get("author", "Unknown")
        if author != "Unknown":
            parts.append(f"{author}.")
        
        # Title in quotes
        parts.append(f'"{citation.get("title", "Untitled")}."')
        
        # Publisher (italicized)
        if citation.get("publisher"):
            parts.append(f"*{citation['publisher']}*,")
        
        # Date
        if citation.get("published_date"):
            parts.append(f"{citation['published_date']}.")
        
        # Medium
        parts.append("Web.")
        
        # Access date
        if citation.get("accessed_date"):
            parts.append(f"{citation['accessed_date']}.")
        
        return " ".join(parts)
    
    def _format_chicago(self, citation: Dict[str, Any]) -> str:
        """Format citation in Chicago style.
        
        Example: Author. "Title." Publisher. Date. URL.
        """
        parts = []
        
        # Author
        author = citation.get("author", "Unknown")
        if author != "Unknown":
            parts.append(f"{author}.")
        
        # Title in quotes
        parts.append(f'"{citation.get("title", "Untitled")}."')
        
        # Publisher
        if citation.get("publisher"):
            parts.append(f"{citation['publisher']}.")
        
        # Date
        if citation.get("published_date"):
            parts.append(f"{citation['published_date']}.")
        elif citation.get("accessed_date"):
            parts.append(f"Accessed {citation['accessed_date']}.")
        
        # URL
        if citation.get("url"):
            parts.append(citation["url"] + ".")
        
        return " ".join(parts)
    
    def _format_harvard(self, citation: Dict[str, Any]) -> str:
        """Format citation in Harvard style.
        
        Example: Author Year, Title, Publisher, viewed Date, <URL>.
        """
        parts = []
        
        # Author and year
        author = citation.get("author", citation.get("publisher", "Unknown"))
        year = self._extract_year(citation.get("published_date"))
        parts.append(f"{author} {year},")
        
        # Title (italicized)
        parts.append(f"*{citation.get('title', 'Untitled')}*,")
        
        # Publisher
        if citation.get("publisher") and citation.get("author") != "Unknown":
            parts.append(f"{citation['publisher']},")
        
        # Viewed date
        if citation.get("accessed_date"):
            parts.append(f"viewed {citation['accessed_date']},")
        
        # URL
        if citation.get("url"):
            parts.append(f"<{citation['url']}>")
        
        return " ".join(parts)
    
    def _format_simple(self, citation: Dict[str, Any]) -> str:
        """Simple citation format."""
        parts = []
        
        if citation.get("author") and citation["author"] != "Unknown":
            parts.append(citation["author"])
        
        parts.append(f'"{citation.get("title", "Untitled")}"')
        
        if citation.get("publisher"):
            parts.append(citation["publisher"])
        
        if citation.get("published_date"):
            parts.append(citation["published_date"])
        
        if citation.get("url"):
            parts.append(f"[{citation['url']}]")
        
        return ". ".join(parts) + "."
    
    def _extract_year(self, date_str: Optional[str]) -> str:
        """Extract year from date string."""
        if not date_str:
            return "n.d."
        
        # Try to find 4-digit year
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if year_match:
            return year_match.group(0)
        
        return "n.d."
    
    def generate_inline_citation(
        self,
        citation: Dict[str, Any],
        style: str = "numbered"
    ) -> str:
        """Generate inline citation marker.
        
        Args:
            citation: Citation data
            style: Inline style (numbered, author-year, footnote)
        
        Returns:
            Inline citation marker
        """
        citation["used"] = True
        
        if style == "numbered":
            return f"[{citation['id']}]"
        elif style == "author-year":
            author = citation.get("author", citation.get("publisher", "Unknown"))
            year = self._extract_year(citation.get("published_date"))
            # Shorten author if too long
            if len(author) > 20:
                author = author.split()[0] if " " in author else author[:15] + "..."
            return f"({author}, {year})"
        elif style == "footnote":
            return f"[^{citation['id']}]"
        else:
            return f"[{citation['id']}]"
    
    def insert_citations(
        self,
        text: str,
        citations_to_insert: List[Tuple[str, Dict[str, Any]]],
        style: str = "numbered"
    ) -> str:
        """Insert citations into text at specified locations.
        
        Args:
            text: Original text
            citations_to_insert: List of (anchor_text, citation) tuples
            style: Citation style
        
        Returns:
            Text with citations inserted
        """
        # Sort by position in text (reverse to maintain positions)
        positions = []
        for anchor, citation in citations_to_insert:
            pos = text.find(anchor)
            if pos != -1:
                positions.append((pos + len(anchor), citation))
        
        positions.sort(reverse=True)
        
        # Insert citations
        for pos, citation in positions:
            marker = self.generate_inline_citation(citation, style)
            text = text[:pos] + marker + text[pos:]
        
        return text
    
    def generate_bibliography(
        self,
        style: str = "apa",
        include_unused: bool = False
    ) -> str:
        """Generate formatted bibliography.
        
        Args:
            style: Citation style
            include_unused: Include citations not used in text
        
        Returns:
            Formatted bibliography
        """
        bibliography = []
        
        # Filter citations
        citations_to_include = self.citations
        if not include_unused:
            citations_to_include = [c for c in self.citations if c.get("used", False)]
        
        # Sort citations (alphabetically by author/title for most styles)
        citations_to_include.sort(
            key=lambda c: c.get("author", c.get("title", "")).lower()
        )
        
        # Format each citation
        for citation in citations_to_include:
            formatted = self.format_citation(citation, style)
            bibliography.append(formatted)
        
        # Create bibliography section
        if style.lower() in ["apa", "harvard"]:
            title = "References"
        elif style.lower() == "mla":
            title = "Works Cited"
        elif style.lower() == "chicago":
            title = "Bibliography"
        else:
            title = "Sources"
        
        result = f"## {title}\n\n"
        
        for i, entry in enumerate(bibliography, 1):
            if style == "numbered":
                result += f"{i}. {entry}\n\n"
            else:
                result += f"{entry}\n\n"
        
        return result
    
    def generate_footnotes(self) -> str:
        """Generate footnotes section.
        
        Returns:
            Formatted footnotes
        """
        footnotes = []
        
        for citation in self.citations:
            if citation.get("used", False):
                formatted = self.format_citation(citation, "simple")
                footnotes.append(f"[^{citation['id']}]: {formatted}")
        
        if footnotes:
            return "\n".join(footnotes)
        
        return ""
    
    def extract_citations_from_text(self, text: str) -> List[str]:
        """Extract existing citations from text.
        
        Args:
            text: Text containing citations
        
        Returns:
            List of citation markers found
        """
        citations = []
        
        # Look for numbered citations [1], [2], etc.
        numbered = re.findall(r"\[(\d+)\]", text)
        citations.extend([f"[{n}]" for n in numbered])
        
        # Look for author-year citations (Author, 2024)
        author_year = re.findall(r"\([^)]+,\s*\d{4}\)", text)
        citations.extend(author_year)
        
        # Look for footnotes [^1], [^2], etc.
        footnotes = re.findall(r"\[\^(\d+)\]", text)
        citations.extend([f"[^{n}]" for n in footnotes])
        
        return citations
    
    def calculate_credibility_score(
        self,
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate credibility score based on citations.
        
        Args:
            citations: List of citations
        
        Returns:
            Credibility analysis
        """
        if not citations:
            return {
                "score": 0,
                "total_citations": 0,
                "source_diversity": 0,
                "authority_sources": 0,
                "recent_sources": 0
            }
        
        # Count unique domains
        domains = set()
        authority_count = 0
        recent_count = 0
        
        for citation in citations:
            # Domain diversity
            if citation.get("url"):
                domain = urlparse(citation["url"]).netloc
                domains.add(domain)
                
                # Check if authority source
                authority_domains = [
                    ".edu", ".gov", ".org", "wikipedia.org",
                    "nature.com", "sciencedirect.com", "pubmed"
                ]
                if any(auth in domain for auth in authority_domains):
                    authority_count += 1
            
            # Check recency (within last 3 years)
            year = self._extract_year(citation.get("published_date"))
            if year != "n.d.":
                try:
                    if int(year) >= datetime.now().year - 3:
                        recent_count += 1
                except ValueError:
                    pass
        
        # Calculate score (0-100)
        diversity_score = min(len(domains) / max(len(citations) * 0.5, 1), 1) * 30
        authority_score = (authority_count / len(citations)) * 40
        recency_score = (recent_count / len(citations)) * 30
        
        total_score = diversity_score + authority_score + recency_score
        
        return {
            "score": round(total_score, 1),
            "total_citations": len(citations),
            "source_diversity": len(domains),
            "authority_sources": authority_count,
            "recent_sources": recent_count,
            "average_sources_per_domain": round(len(citations) / max(len(domains), 1), 1)
        }
    
    def export_citations(self, format: str = "json") -> str:
        """Export citations in various formats.
        
        Args:
            format: Export format (json, bibtex, ris)
        
        Returns:
            Exported citations
        """
        if format == "json":
            import json
            return json.dumps(self.citations, indent=2)
        
        elif format == "bibtex":
            bibtex = []
            for citation in self.citations:
                entry_type = "@misc"  # Default type
                if citation.get("type") == "article":
                    entry_type = "@article"
                elif citation.get("type") == "book":
                    entry_type = "@book"
                
                key = f"{citation.get('author', 'unknown').split()[0].lower()}{self._extract_year(citation.get('published_date'))}"
                
                bibtex.append(f"{entry_type}{{{key},")
                bibtex.append(f"  title = {{{citation.get('title', 'Untitled')}}},")
                if citation.get("author"):
                    bibtex.append(f"  author = {{{citation['author']}}},")
                if citation.get("published_date"):
                    bibtex.append(f"  year = {{{self._extract_year(citation['published_date'])}}},")
                if citation.get("publisher"):
                    bibtex.append(f"  publisher = {{{citation['publisher']}}},")
                if citation.get("url"):
                    bibtex.append(f"  url = {{{citation['url']}}},")
                bibtex.append("}\n")
            
            return "\n".join(bibtex)
        
        elif format == "ris":
            ris = []
            for citation in self.citations:
                ris.append("TY  - ELEC")  # Electronic source
                ris.append(f"TI  - {citation.get('title', 'Untitled')}")
                if citation.get("author"):
                    ris.append(f"AU  - {citation['author']}")
                if citation.get("published_date"):
                    ris.append(f"PY  - {self._extract_year(citation['published_date'])}")
                if citation.get("publisher"):
                    ris.append(f"PB  - {citation['publisher']}")
                if citation.get("url"):
                    ris.append(f"UR  - {citation['url']}")
                ris.append("ER  -\n")
            
            return "\n".join(ris)
        
        else:
            return str(self.citations)