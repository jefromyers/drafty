"""Web scraping utilities using selectolax and trafilatura."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser, Node
import trafilatura


class ContentExtractor:
    """Extract and clean content from HTML using trafilatura and selectolax."""

    # Common selectors for content extraction (fallback)
    ARTICLE_SELECTORS = [
        "article",
        "main",
        "[role='main']",
        ".content",
        "#content",
        ".post",
        ".entry-content",
        ".article-content",
        ".post-content",
    ]

    # Elements to remove from content
    REMOVE_SELECTORS = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        ".sidebar",
        ".advertisement",
        ".ads",
        ".social-share",
        ".comments",
        "#comments",
        ".related-posts",
    ]

    def __init__(self, html: str, base_url: Optional[str] = None):
        """Initialize with HTML content."""
        self.html = html
        self.parser = HTMLParser(html)
        self.base_url = base_url

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract page metadata using trafilatura and selectolax."""
        metadata = {}
        
        # Use trafilatura for metadata extraction
        traf_metadata = trafilatura.extract_metadata(self.html)
        if traf_metadata:
            metadata.update({
                "title": traf_metadata.title,
                "author": traf_metadata.author,
                "description": traf_metadata.description,
                "published_date": traf_metadata.date,
                "site_name": traf_metadata.sitename,
                "categories": traf_metadata.categories,
                "tags": traf_metadata.tags,
            })
        
        # Supplement with selectolax for additional metadata
        # Open Graph metadata
        og_tags = self.parser.css('meta[property^="og:"]')
        for tag in og_tags:
            prop = tag.attributes.get("property", "").replace("og:", "")
            content = tag.attributes.get("content", "")
            if prop and content:
                metadata[f"og_{prop}"] = content

        # Keywords (if not captured by trafilatura)
        if "keywords" not in metadata:
            keywords_elem = self.parser.css_first('meta[name="keywords"]')
            if keywords_elem:
                keywords = keywords_elem.attributes.get("content", "")
                metadata["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]

        return {k: v for k, v in metadata.items() if v}  # Remove None values

    def extract_main_content(self, output_format: str = "markdown", clean: bool = True) -> str:
        """Extract the main content using trafilatura with fallback to selectolax.
        
        Args:
            output_format: Output format - 'markdown', 'text', or 'xml'
            clean: Whether to clean the content
        """
        # Try trafilatura first - it's optimized for content extraction
        content = trafilatura.extract(
            self.html,
            url=self.base_url,
            output_format=output_format,
            include_links=True,
            include_formatting=True,
            include_tables=True,
            deduplicate=clean,
            target_language="en",
        )
        
        if content:
            return content
        
        # Fallback to selectolax-based extraction
        return self._extract_with_selectolax(clean)

    def _extract_with_selectolax(self, clean: bool = True) -> str:
        """Fallback content extraction using selectolax."""
        # Remove unwanted elements first
        if clean:
            for selector in self.REMOVE_SELECTORS:
                for elem in self.parser.css(selector):
                    elem.decompose()

        # Try to find main content container
        content = None
        for selector in self.ARTICLE_SELECTORS:
            elem = self.parser.css_first(selector)
            if elem:
                content = elem
                break

        # Fallback to body if no specific container found
        if not content:
            content = self.parser.css_first("body")

        if not content:
            return ""

        # Extract text with basic formatting preserved
        return self._extract_text_with_formatting(content)

    def _extract_text_with_formatting(self, node: Node) -> str:
        """Extract text while preserving basic formatting."""
        text_parts = []

        def process_node(n: Node, depth: int = 0):
            # Handle text nodes
            if n.tag == "-text":
                text = n.text(strip=False)
                if text.strip():
                    text_parts.append(text)
                return

            # Handle block-level elements
            if n.tag in ["p", "div", "section", "article", "header", "footer"]:
                if text_parts and text_parts[-1] != "\n\n":
                    text_parts.append("\n\n")

            # Handle headings
            elif n.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if text_parts and text_parts[-1] != "\n\n":
                    text_parts.append("\n\n")
                level = int(n.tag[1])
                text_parts.append("#" * level + " ")

            # Handle lists
            elif n.tag == "ul" or n.tag == "ol":
                if text_parts and text_parts[-1] != "\n\n":
                    text_parts.append("\n\n")

            elif n.tag == "li":
                text_parts.append("\n- ")

            # Handle line breaks
            elif n.tag == "br":
                text_parts.append("\n")

            # Process children
            for child in n.iter():
                if child != n:  # Skip self
                    process_node(child, depth + 1)

            # Add spacing after block elements
            if n.tag in ["p", "div", "section", "article", "ul", "ol"]:
                if text_parts and text_parts[-1] not in ["\n", "\n\n"]:
                    text_parts.append("\n\n")

        process_node(content)

        # Clean up the text
        text = "".join(text_parts)
        
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        
        return text.strip()

    def extract_links(self, internal_only: bool = False) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        for link in self.parser.css("a[href]"):
            href = link.attributes.get("href", "")
            if not href or href.startswith("#"):
                continue

            # Resolve relative URLs
            if self.base_url:
                href = urljoin(self.base_url, href)

            # Filter internal links if requested
            if internal_only and self.base_url:
                base_domain = urlparse(self.base_url).netloc
                link_domain = urlparse(href).netloc
                if base_domain != link_domain:
                    continue

            links.append({
                "url": href,
                "text": link.text(strip=True),
                "title": link.attributes.get("title", ""),
            })

        return links

    def extract_images(self) -> List[Dict[str, str]]:
        """Extract all images from the page."""
        images = []
        for img in self.parser.css("img"):
            src = img.attributes.get("src", "")
            if not src:
                continue

            # Resolve relative URLs
            if self.base_url:
                src = urljoin(self.base_url, src)

            images.append({
                "src": src,
                "alt": img.attributes.get("alt", ""),
                "title": img.attributes.get("title", ""),
                "width": img.attributes.get("width", ""),
                "height": img.attributes.get("height", ""),
            })

        return images

    def extract_headings(self) -> List[Dict[str, Any]]:
        """Extract all headings with hierarchy."""
        headings = []
        for level in range(1, 7):
            for heading in self.parser.css(f"h{level}"):
                headings.append({
                    "level": level,
                    "text": heading.text(strip=True),
                    "id": heading.attributes.get("id", ""),
                })
        return headings

    def extract_structured_data(self) -> List[Dict[str, Any]]:
        """Extract JSON-LD structured data."""
        structured_data = []
        
        for script in self.parser.css('script[type="application/ld+json"]'):
            try:
                import json
                data = json.loads(script.text())
                structured_data.append(data)
            except json.JSONDecodeError:
                continue
        
        return structured_data

    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract tables as structured data."""
        tables = []
        
        for table in self.parser.css("table"):
            table_data = {
                "headers": [],
                "rows": [],
            }
            
            # Extract headers
            for th in table.css("th"):
                table_data["headers"].append(th.text(strip=True))
            
            # Extract rows
            for tr in table.css("tr"):
                row = []
                for td in tr.css("td"):
                    row.append(td.text(strip=True))
                if row:
                    table_data["rows"].append(row)
            
            if table_data["headers"] or table_data["rows"]:
                tables.append(table_data)
        
        return tables

    def extract_faq(self) -> List[Dict[str, str]]:
        """Extract FAQ-style Q&A pairs."""
        faqs = []
        
        # Look for common FAQ patterns
        faq_selectors = [
            (".faq-question", ".faq-answer"),
            (".question", ".answer"),
            ("dt", "dd"),
            ("[itemprop='name']", "[itemprop='acceptedAnswer']"),
        ]
        
        for q_selector, a_selector in faq_selectors:
            questions = self.parser.css(q_selector)
            answers = self.parser.css(a_selector)
            
            if questions and len(questions) == len(answers):
                for q, a in zip(questions, answers):
                    faqs.append({
                        "question": q.text(strip=True),
                        "answer": a.text(strip=True),
                    })
                break
        
        # Also check for FAQ structured data
        for data in self.extract_structured_data():
            if data.get("@type") == "FAQPage" and "mainEntity" in data:
                for item in data["mainEntity"]:
                    if item.get("@type") == "Question":
                        faqs.append({
                            "question": item.get("name", ""),
                            "answer": item.get("acceptedAnswer", {}).get("text", ""),
                        })
        
        return faqs

    def extract_comments(self) -> Optional[str]:
        """Extract comments using trafilatura."""
        return trafilatura.extract(
            self.html,
            url=self.base_url,
            include_comments=True,
            only_with_metadata=False,
            target_language="en",
        )

    def get_text_content(self) -> str:
        """Get all text content from the page using trafilatura."""
        # Use trafilatura for clean text extraction
        text = trafilatura.extract(
            self.html,
            url=self.base_url,
            output_format="txt",
            include_links=False,
            include_formatting=False,
            target_language="en",
        )
        
        if text:
            return text
        
        # Fallback to selectolax
        for elem in self.parser.css("script, style"):
            elem.decompose()
        return self.parser.text(strip=True)

    def get_word_count(self) -> int:
        """Get approximate word count of main content."""
        content = self.get_text_content()
        return len(content.split()) if content else 0


async def scrape_url(
    url: str,
    use_javascript: bool = False,
    extract_links: bool = True,
    extract_images: bool = False,
    clean_content: bool = True,
    output_format: str = "markdown",
) -> Dict[str, Any]:
    """Scrape a URL and extract structured content.
    
    Args:
        url: The URL to scrape
        use_javascript: Whether to use Browserless for JS rendering
        extract_links: Whether to extract links
        extract_images: Whether to extract images
        clean_content: Whether to clean the content
        output_format: Format for content extraction ('markdown', 'text', 'xml')
    """
    from drafty.utils.http import HTTPClient
    
    async with HTTPClient() as client:
        if use_javascript:
            html = await client.fetch_with_javascript(url)
        else:
            response = await client.get(url)
            html = response.text
    
    extractor = ContentExtractor(html, base_url=url)
    
    result = {
        "url": url,
        "metadata": extractor.extract_metadata(),
        "content": extractor.extract_main_content(output_format=output_format, clean=clean_content),
        "headings": extractor.extract_headings(),
        "word_count": extractor.get_word_count(),
    }
    
    if extract_links:
        result["links"] = extractor.extract_links()
    
    if extract_images:
        result["images"] = extractor.extract_images()
    
    # Try to extract FAQ content
    faqs = extractor.extract_faq()
    if faqs:
        result["faqs"] = faqs
    
    # Extract tables if present
    tables = extractor.extract_tables()
    if tables:
        result["tables"] = tables
    
    # Extract structured data
    structured = extractor.extract_structured_data()
    if structured:
        result["structured_data"] = structured
    
    return result


async def scrape_multiple(
    urls: List[str],
    max_concurrent: int = 5,
    use_javascript: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Scrape multiple URLs concurrently."""
    import asyncio
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_limit(url: str) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await scrape_url(url, use_javascript=use_javascript, **kwargs)
            except Exception as e:
                return {
                    "url": url,
                    "error": str(e),
                    "status": "failed",
                }
    
    tasks = [scrape_with_limit(url) for url in urls]
    return await asyncio.gather(*tasks)