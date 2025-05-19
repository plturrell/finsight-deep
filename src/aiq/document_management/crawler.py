"""Web crawler for deep research data collection."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re

from .models import Document, DocumentType, DublinCoreMetadata
from ..settings.global_settings import GlobalSettings


logger = logging.getLogger(__name__)


class ResearchWebCrawler:
    """Advanced web crawler for research data collection."""
    
    def __init__(self, max_depth: int = 3, max_pages: int = 100):
        """Initialize web crawler with depth and page limits."""
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Content patterns for research-relevant data
        self.research_patterns = {
            "academic": [r"abstract", r"methodology", r"results", r"conclusion"],
            "technical": [r"documentation", r"api", r"specification", r"architecture"],
            "data": [r"dataset", r"statistics", r"metrics", r"analysis"],
            "code": [r"github\.com", r"gitlab\.com", r"source code", r"repository"]
        }
        
        # File type handlers
        self.file_handlers = {
            ".pdf": self._extract_pdf_metadata,
            ".json": self._extract_json_data,
            ".csv": self._extract_csv_metadata,
            ".xml": self._extract_xml_data
        }
    
    async def crawl_site(self, start_url: str, focus_keywords: Optional[List[str]] = None) -> List[Document]:
        """Crawl a website starting from the given URL."""
        self.visited_urls.clear()
        collected_documents = []
        
        async with aiohttp.ClientSession() as self.session:
            await self._crawl_recursive(
                start_url, 
                0, 
                collected_documents,
                focus_keywords or []
            )
        
        return collected_documents
    
    async def _crawl_recursive(self, url: str, depth: int, documents: List[Document], 
                             keywords: List[str]):
        """Recursively crawl pages up to max depth."""
        if (depth > self.max_depth or 
            len(self.visited_urls) >= self.max_pages or
            url in self.visited_urls):
            return
        
        self.visited_urls.add(url)
        
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    return
                
                content_type = response.headers.get('Content-Type', '')
                
                if 'text/html' in content_type:
                    # Process HTML page
                    html = await response.text()
                    document = await self._process_html_page(url, html, keywords)
                    if document:
                        documents.append(document)
                    
                    # Extract and crawl links
                    links = self._extract_links(html, url)
                    filtered_links = self._filter_relevant_links(links, keywords)
                    
                    # Crawl child pages
                    tasks = []
                    for link in filtered_links[:10]:  # Limit concurrent crawls
                        task = self._crawl_recursive(link, depth + 1, documents, keywords)
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                else:
                    # Handle other file types
                    document = await self._process_file(url, response, content_type)
                    if document:
                        documents.append(document)
        
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
    
    async def _process_html_page(self, url: str, html: str, keywords: List[str]) -> Optional[Document]:
        """Process HTML page and extract research-relevant content."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.text.strip() if title else urlparse(url).netloc
        
        # Extract main content
        content = self._extract_main_content(soup)
        
        # Check relevance
        relevance_score = self._calculate_relevance_score(content, keywords)
        if relevance_score < 0.1:  # Threshold for relevance
            return None
        
        # Extract structured data
        structured_data = self._extract_structured_data(soup)
        
        # Create document
        metadata = DublinCoreMetadata(
            title=title_text,
            source=url,
            type="web_page",
            date=datetime.utcnow(),
            description=self._generate_description(content),
            subject=self._extract_keywords(content)
        )
        
        document = Document(
            type=DocumentType.WEB_PAGE,
            title=title_text,
            content=content,
            url=url,
            metadata=metadata,
            extracted_data=structured_data
        )
        
        return document
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data from HTML."""
        structured_data = {}
        
        # JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data['json_ld'] = data
            except:
                pass
        
        # Meta tags
        meta_data = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property', '')
            content = meta.get('content', '')
            if name and content:
                meta_data[name] = content
        
        if meta_data:
            structured_data['meta'] = meta_data
        
        # Tables
        tables = []
        for table in soup.find_all('table'):
            table_data = self._parse_table(table)
            if table_data:
                tables.append(table_data)
        
        if tables:
            structured_data['tables'] = tables
        
        return structured_data
    
    def _parse_table(self, table) -> Optional[Dict[str, Any]]:
        """Parse HTML table into structured data."""
        headers = []
        rows = []
        
        # Extract headers
        header_row = table.find('thead', 'tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        
        if headers or rows:
            return {
                'headers': headers,
                'rows': rows
            }
        return None
    
    def _calculate_relevance_score(self, content: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keywords and patterns."""
        content_lower = content.lower()
        score = 0.0
        
        # Check keywords
        for keyword in keywords:
            count = content_lower.count(keyword.lower())
            score += count * 0.1
        
        # Check research patterns
        for category, patterns in self.research_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    score += 0.2
        
        # Normalize score
        return min(score, 1.0)
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML page."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for tag in soup.find_all(['a', 'link']):
            href = tag.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
        
        return links
    
    def _filter_relevant_links(self, links: List[str], keywords: List[str]) -> List[str]:
        """Filter links based on relevance criteria."""
        filtered = []
        
        for link in links:
            # Skip already visited
            if link in self.visited_urls:
                continue
            
            # Check domain
            parsed = urlparse(link)
            
            # Check file extensions
            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in ['.pdf', '.json', '.csv', '.xml']):
                filtered.append(link)
                continue
            
            # Check keywords in URL
            url_lower = link.lower()
            if any(keyword.lower() in url_lower for keyword in keywords):
                filtered.append(link)
                continue
            
            # Check research-relevant paths
            research_paths = ['research', 'paper', 'article', 'publication', 'data', 'dataset']
            if any(path in url_lower for path in research_paths):
                filtered.append(link)
        
        return filtered
    
    async def _process_file(self, url: str, response: aiohttp.ClientResponse, 
                          content_type: str) -> Optional[Document]:
        """Process non-HTML files."""
        try:
            # Determine file type
            file_ext = self._get_file_extension(url, content_type)
            
            if file_ext in self.file_handlers:
                content = await response.read()
                metadata = await self.file_handlers[file_ext](url, content, content_type)
                
                document = Document(
                    type=self._map_file_type(file_ext),
                    title=metadata.title,
                    url=url,
                    metadata=metadata,
                    extracted_data={"file_type": file_ext, "size": len(content)}
                )
                
                return document
        
        except Exception as e:
            logger.error(f"Error processing file {url}: {e}")
        
        return None
    
    def _get_file_extension(self, url: str, content_type: str) -> str:
        """Determine file extension from URL or content type."""
        # Try URL first
        path = urlparse(url).path
        if '.' in path:
            return path.split('.')[-1].lower()
        
        # Fall back to content type
        type_mapping = {
            'application/pdf': 'pdf',
            'application/json': 'json',
            'text/csv': 'csv',
            'application/xml': 'xml',
            'text/xml': 'xml'
        }
        
        for mime, ext in type_mapping.items():
            if mime in content_type:
                return ext
        
        return 'unknown'
    
    def _map_file_type(self, extension: str) -> DocumentType:
        """Map file extension to document type."""
        mapping = {
            'pdf': DocumentType.PDF,
            'json': DocumentType.JSON,
            'csv': DocumentType.EXCEL,
            'xml': DocumentType.TEXT
        }
        return mapping.get(extension, DocumentType.TEXT)
    
    async def _extract_pdf_metadata(self, url: str, content: bytes, content_type: str) -> DublinCoreMetadata:
        """Extract metadata from PDF file."""
        # In production, use PyPDF2 or similar to extract real metadata
        return DublinCoreMetadata(
            title=f"PDF Document from {urlparse(url).netloc}",
            source=url,
            type="pdf",
            format=content_type,
            date=datetime.utcnow()
        )
    
    async def _extract_json_data(self, url: str, content: bytes, content_type: str) -> DublinCoreMetadata:
        """Extract metadata from JSON file."""
        try:
            data = json.loads(content)
            
            # Look for common metadata fields
            title = data.get('title') or data.get('name') or f"JSON Data from {urlparse(url).netloc}"
            description = data.get('description') or data.get('summary', '')
            
            return DublinCoreMetadata(
                title=title,
                description=description,
                source=url,
                type="dataset",
                format=content_type,
                date=datetime.utcnow()
            )
        except:
            return DublinCoreMetadata(
                title=f"JSON Data from {urlparse(url).netloc}",
                source=url,
                type="dataset",
                format=content_type,
                date=datetime.utcnow()
            )
    
    async def _extract_csv_metadata(self, url: str, content: bytes, content_type: str) -> DublinCoreMetadata:
        """Extract metadata from CSV file."""
        return DublinCoreMetadata(
            title=f"CSV Dataset from {urlparse(url).netloc}",
            source=url,
            type="dataset",
            format=content_type,
            date=datetime.utcnow()
        )
    
    async def _extract_xml_data(self, url: str, content: bytes, content_type: str) -> DublinCoreMetadata:
        """Extract metadata from XML file."""
        return DublinCoreMetadata(
            title=f"XML Document from {urlparse(url).netloc}",
            source=url,
            type="structured_data",
            format=content_type,
            date=datetime.utcnow()
        )
    
    def _generate_description(self, content: str) -> str:
        """Generate description from content."""
        # Take first few sentences
        sentences = content.split('.')[:3]
        return '. '.join(sentences) + '.' if sentences else ''
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction - in production use NLP
        words = content.lower().split()
        
        # Common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]