"""HTTP client utilities using httpx."""

import asyncio
from typing import Any, Dict, List, Optional

import httpx
from httpx import AsyncClient, Response


class HTTPClient:
    """Async HTTP client wrapper with retry logic and connection pooling."""

    # Realistic Mac Chrome user agent
    DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        browserless_url: str = "http://localhost:3000",
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.browserless_url = browserless_url
        
        self.headers = headers or {}
        self.headers.setdefault("User-Agent", self.user_agent)
        
        # Configure client with connection pooling
        self.client = AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers=self.headers,
            follow_redirects=True,
        )

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Response:
        """Make an HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise
                last_exception = e
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
            
            # Wait before retry with exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
        
        # Raise the last exception if all retries failed
        if last_exception:
            raise last_exception
        raise Exception(f"Failed to complete request after {self.max_retries} attempts")

    async def get(self, url: str, **kwargs) -> Response:
        """GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Response:
        """POST request."""
        return await self.request("POST", url, **kwargs)

    async def get_json(self, url: str, **kwargs) -> Any:
        """GET request returning JSON."""
        response = await self.get(url, **kwargs)
        return response.json()

    async def post_json(self, url: str, data: Any, **kwargs) -> Any:
        """POST request with JSON data."""
        response = await self.post(url, json=data, **kwargs)
        return response.json()

    async def fetch_with_javascript(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        wait_time: int = 3000,
        block_ads: bool = True,
    ) -> str:
        """Fetch URL content using Browserless for JavaScript rendering.
        
        Args:
            url: The URL to fetch
            wait_for_selector: CSS selector to wait for before returning
            wait_time: Time to wait in milliseconds
            block_ads: Whether to block ads and trackers
        """
        browserless_endpoint = f"{self.browserless_url}/content"
        
        payload = {
            "url": url,
            "waitUntil": "networkidle2",
            "userAgent": self.user_agent,
        }
        
        if wait_for_selector:
            payload["waitForSelector"] = wait_for_selector
            payload["waitForSelectorTimeout"] = wait_time
        else:
            payload["waitForTimeout"] = wait_time
        
        if block_ads:
            payload["blockAds"] = True
        
        try:
            response = await self.post_json(browserless_endpoint, payload)
            return response.get("data", "")
        except Exception as e:
            # Fallback to regular fetch if Browserless is not available
            print(f"Browserless failed, falling back to regular fetch: {e}")
            response = await self.get(url)
            return response.text

    async def screenshot(
        self,
        url: str,
        full_page: bool = True,
        width: int = 1920,
        height: int = 1080,
    ) -> bytes:
        """Take a screenshot of a webpage using Browserless.
        
        Args:
            url: The URL to screenshot
            full_page: Whether to capture the full page
            width: Viewport width
            height: Viewport height
        """
        browserless_endpoint = f"{self.browserless_url}/screenshot"
        
        payload = {
            "url": url,
            "fullPage": full_page,
            "viewport": {
                "width": width,
                "height": height,
            },
            "userAgent": self.user_agent,
        }
        
        response = await self.post(browserless_endpoint, json=payload)
        return response.content

    async def pdf(
        self,
        url: str,
        format: str = "A4",
        print_background: bool = True,
    ) -> bytes:
        """Convert a webpage to PDF using Browserless.
        
        Args:
            url: The URL to convert
            format: Paper format (A4, Letter, etc.)
            print_background: Whether to print background graphics
        """
        browserless_endpoint = f"{self.browserless_url}/pdf"
        
        payload = {
            "url": url,
            "format": format,
            "printBackground": print_background,
            "userAgent": self.user_agent,
        }
        
        response = await self.post(browserless_endpoint, json=payload)
        return response.content

    async def fetch_with_cache(
        self,
        url: str,
        cache_dir: Optional[str] = None,
        cache_duration: int = 3600,
        use_javascript: bool = False,
        **kwargs,
    ) -> str:
        """Fetch URL content with optional caching.
        
        Args:
            url: The URL to fetch
            cache_dir: Directory to store cache files
            cache_duration: How long cache is valid in seconds
            use_javascript: Whether to use Browserless for JS rendering
        """
        import hashlib
        import time
        from pathlib import Path
        
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Create cache key from URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_file = cache_path / f"{url_hash}.cache"
            
            # Check if cache exists and is fresh
            if cache_file.exists():
                age = time.time() - cache_file.stat().st_mtime
                if age < cache_duration:
                    return cache_file.read_text()
        
        # Fetch the content
        if use_javascript:
            content = await self.fetch_with_javascript(url, **kwargs)
        else:
            response = await self.get(url, **kwargs)
            content = response.text
        
        # Save to cache if enabled
        if cache_dir and cache_file:
            cache_file.write_text(content)
        
        return content


class ConcurrentFetcher:
    """Fetch multiple URLs concurrently with rate limiting."""

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        browserless_url: str = "http://localhost:3000",
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or HTTPClient.DEFAULT_USER_AGENT
        self.browserless_url = browserless_url
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(
        self,
        client: HTTPClient,
        url: str,
        use_javascript: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fetch a single URL with rate limiting."""
        async with self.semaphore:
            try:
                if use_javascript:
                    content = await client.fetch_with_javascript(url, **kwargs)
                    return {
                        "url": url,
                        "status": "success",
                        "content": content,
                        "rendered": True,
                    }
                else:
                    response = await client.get(url, **kwargs)
                    return {
                        "url": url,
                        "status": "success",
                        "content": response.text,
                        "headers": dict(response.headers),
                        "status_code": response.status_code,
                        "rendered": False,
                    }
            except Exception as e:
                return {
                    "url": url,
                    "status": "error",
                    "error": str(e),
                }

    async def fetch_all(
        self,
        urls: List[str],
        use_javascript: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently."""
        async with HTTPClient(
            timeout=self.timeout,
            max_retries=self.max_retries,
            user_agent=self.user_agent,
            browserless_url=self.browserless_url,
        ) as client:
            tasks = [
                self.fetch_one(client, url, use_javascript, **kwargs) 
                for url in urls
            ]
            results = await asyncio.gather(*tasks)
            return results


async def check_url(url: str, timeout: int = 10) -> bool:
    """Check if a URL is accessible."""
    try:
        async with HTTPClient(timeout=timeout, max_retries=1) as client:
            response = await client.get(url)
            return 200 <= response.status_code < 400
    except Exception:
        return False