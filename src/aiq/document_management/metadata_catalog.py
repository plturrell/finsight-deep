"""Automated metadata cataloguing with Dublin Core."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from langdetect import detect
import mimetypes

from .models import Document, DublinCoreMetadata, DocumentType
from ..llm.openai_llm import OpenAILLM
from ..data_models.config import WorkflowConfig


logger = logging.getLogger(__name__)


class MetadataCataloguer:
    """Automated metadata cataloguer using Dublin Core standard."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize metadata cataloguer."""
        self.config = config
        self.llm = None
        
        # Dublin Core elements mapping
        self.dc_elements = {
            'title': 'A name given to the resource',
            'creator': 'An entity primarily responsible for making the resource',
            'subject': 'The topic of the resource',
            'description': 'An account of the resource',
            'publisher': 'An entity responsible for making the resource available',
            'contributor': 'An entity responsible for making contributions',
            'date': 'A point or period of time associated with an event',
            'type': 'The nature or genre of the resource',
            'format': 'The file format, physical medium, or dimensions',
            'identifier': 'An unambiguous reference to the resource',
            'source': 'A related resource from which the described resource is derived',
            'language': 'A language of the resource',
            'relation': 'A related resource',
            'coverage': 'The spatial or temporal topic of the resource',
            'rights': 'Information about rights held in and over the resource'
        }
        
        # Content type patterns
        self.content_patterns = {
            'research_paper': [
                r'abstract', r'introduction', r'methodology', r'results', 
                r'conclusion', r'references', r'doi:\s*\S+'
            ],
            'technical_documentation': [
                r'api', r'endpoint', r'parameter', r'response', r'example',
                r'installation', r'configuration', r'usage'
            ],
            'dataset': [
                r'data', r'dataset', r'csv', r'json', r'xml', r'schema',
                r'column', r'row', r'field', r'record'
            ],
            'presentation': [
                r'slide', r'presentation', r'agenda', r'overview',
                r'objectives', r'summary', r'questions'
            ],
            'legal_document': [
                r'whereas', r'hereby', r'agreement', r'contract',
                r'terms', r'conditions', r'liability', r'jurisdiction'
            ]
        }
    
    def generate_metadata(self, document: Document, 
                         additional_metadata: Optional[Dict[str, Any]] = None) -> DublinCoreMetadata:
        """Generate comprehensive Dublin Core metadata for a document."""
        try:
            # Start with basic metadata
            metadata = DublinCoreMetadata(
                title=self._extract_title(document, additional_metadata),
                creator=self._extract_creator(document, additional_metadata),
                date=self._extract_date(document, additional_metadata),
                type=self._determine_document_type(document),
                format=self._determine_format(document),
                language=self._detect_language(document),
                source=document.url or document.file_path,
                identifier=f"aiq:{document.id}"
            )
            
            # Extract content-based metadata
            if document.content:
                metadata.subject = self._extract_subjects(document)
                metadata.description = self._generate_description(document)
                metadata.coverage = self._extract_coverage(document)
            
            # Add any additional metadata provided
            if additional_metadata:
                for key, value in additional_metadata.items():
                    if hasattr(metadata, key) and value is not None:
                        setattr(metadata, key, value)
            
            # Use LLM for enhanced metadata generation if available
            if self.config and document.content:
                enhanced_metadata = self._enhance_with_llm(document, metadata)
                return enhanced_metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return DublinCoreMetadata(title=document.title or "Unknown")
    
    def generate_metadata_from_dict(self, data: Dict[str, Any]) -> DublinCoreMetadata:
        """Generate metadata from dictionary data."""
        metadata = DublinCoreMetadata(
            title=data.get('title', 'Untitled'),
            creator=data.get('creator'),
            subject=data.get('subject'),
            description=data.get('description'),
            publisher=data.get('publisher'),
            contributor=data.get('contributor'),
            date=data.get('date'),
            type=data.get('type'),
            format=data.get('format'),
            identifier=data.get('identifier'),
            source=data.get('source'),
            language=data.get('language', 'en'),
            relation=data.get('relation'),
            coverage=data.get('coverage'),
            rights=data.get('rights')
        )
        return metadata
    
    def _extract_title(self, document: Document, additional_metadata: Optional[Dict[str, Any]]) -> str:
        """Extract or generate title."""
        if additional_metadata and 'title' in additional_metadata:
            return additional_metadata['title']
        
        if document.title:
            return document.title
        
        if document.content:
            # Try to extract title from content
            lines = document.content.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and len(line) < 200:  # Reasonable title length
                    return line
        
        return f"Document {document.id}"
    
    def _extract_creator(self, document: Document, additional_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract creator information."""
        if additional_metadata and 'creator' in additional_metadata:
            return additional_metadata['creator']
        
        if document.content:
            # Look for author patterns
            author_patterns = [
                r'Author:\s*(.+)',
                r'By\s+(.+)',
                r'Created by:\s*(.+)',
                r'Written by:\s*(.+)'
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, document.content, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def _extract_date(self, document: Document, additional_metadata: Optional[Dict[str, Any]]) -> datetime:
        """Extract or determine date."""
        if additional_metadata and 'date' in additional_metadata:
            if isinstance(additional_metadata['date'], datetime):
                return additional_metadata['date']
            # Try to parse string date
            try:
                from dateutil import parser
                return parser.parse(additional_metadata['date'])
            except:
                pass
        
        if document.created_at:
            return document.created_at
        
        # Try to extract date from content
        if document.content:
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, document.content)
                if match:
                    try:
                        from dateutil import parser
                        return parser.parse(match.group(0))
                    except:
                        pass
        
        return datetime.utcnow()
    
    def _determine_document_type(self, document: Document) -> str:
        """Determine the type of document based on content analysis."""
        if document.type == DocumentType.PDF:
            return "document/pdf"
        elif document.type == DocumentType.EXCEL:
            return "dataset/spreadsheet"
        elif document.type == DocumentType.WEB_PAGE:
            return "webpage"
        
        # Analyze content to determine type
        if document.content:
            content_lower = document.content.lower()
            
            type_scores = {}
            for doc_type, patterns in self.content_patterns.items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        score += 1
                type_scores[doc_type] = score
            
            # Get highest scoring type
            if type_scores:
                best_type = max(type_scores.items(), key=lambda x: x[1])
                if best_type[1] > 0:
                    return best_type[0]
        
        return "text"
    
    def _determine_format(self, document: Document) -> str:
        """Determine the format of the document."""
        if document.file_path:
            mime_type, _ = mimetypes.guess_type(document.file_path)
            if mime_type:
                return mime_type
        
        type_to_format = {
            DocumentType.PDF: "application/pdf",
            DocumentType.EXCEL: "application/vnd.ms-excel",
            DocumentType.WEB_PAGE: "text/html",
            DocumentType.TEXT: "text/plain",
            DocumentType.JSON: "application/json"
        }
        
        return type_to_format.get(document.type, "text/plain")
    
    def _detect_language(self, document: Document) -> str:
        """Detect the language of the document."""
        if document.content:
            try:
                # Use langdetect for language detection
                lang = detect(document.content[:1000])  # Use first 1000 chars
                return lang
            except:
                pass
        
        return "en"  # Default to English
    
    def _extract_subjects(self, document: Document) -> List[str]:
        """Extract subject keywords from document."""
        subjects = []
        
        if document.content:
            # Simple keyword extraction
            content_lower = document.content.lower()
            
            # Technical terms
            tech_terms = [
                'machine learning', 'artificial intelligence', 'neural network',
                'deep learning', 'data science', 'algorithm', 'database',
                'cloud computing', 'cybersecurity', 'blockchain'
            ]
            
            for term in tech_terms:
                if term in content_lower:
                    subjects.append(term)
            
            # Extract capitalized phrases (potential topics)
            capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            matches = re.findall(capitalized_pattern, document.content)
            
            # Filter and add unique subjects
            for match in matches:
                if len(match) > 3 and match not in subjects:
                    subjects.append(match)
            
            # Limit to top 10 subjects
            return subjects[:10]
        
        return []
    
    def _generate_description(self, document: Document) -> str:
        """Generate a description of the document."""
        if document.content:
            # Extract first paragraph or first few sentences
            paragraphs = document.content.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50:  # Meaningful paragraph
                    # Limit to ~200 characters
                    if len(para) > 200:
                        return para[:197] + "..."
                    return para
            
            # Fallback to first few sentences
            sentences = document.content.split('.')[:3]
            return '. '.join(sentences) + '.'
        
        return ""
    
    def _extract_coverage(self, document: Document) -> Optional[str]:
        """Extract spatial or temporal coverage."""
        if document.content:
            # Look for date ranges
            date_range_pattern = r'(\d{4})\s*-\s*(\d{4})'
            match = re.search(date_range_pattern, document.content)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
            
            # Look for geographic locations
            location_patterns = [
                r'(United States|US|USA|America)',
                r'(Europe|European Union|EU)',
                r'(Asia|Asian)',
                r'(Global|Worldwide|International)'
            ]
            
            for pattern in location_patterns:
                if re.search(pattern, document.content, re.IGNORECASE):
                    return re.search(pattern, document.content, re.IGNORECASE).group(1)
        
        return None
    
    def _enhance_with_llm(self, document: Document, metadata: DublinCoreMetadata) -> DublinCoreMetadata:
        """Use LLM to enhance metadata generation."""
        try:
            if not self.llm:
                self.llm = OpenAILLM(self.config)
            
            # Create prompt for metadata enhancement
            prompt = f"""
            Analyze the following document content and enhance the metadata according to Dublin Core standards.
            
            Current metadata:
            - Title: {metadata.title}
            - Type: {metadata.type}
            - Language: {metadata.language}
            
            Document content (first 1000 characters):
            {document.content[:1000]}
            
            Please provide:
            1. A more descriptive title if appropriate
            2. A comprehensive description (100-200 words)
            3. Key subjects/topics (5-10 keywords)
            4. Any identifiable creator/author information
            5. Relevant coverage (temporal or spatial)
            
            Format the response as JSON.
            """
            
            # Get LLM response
            llm_instance = self.llm.get_llm()
            response = llm_instance.complete(prompt)
            
            # Parse and apply enhancements
            try:
                import json
                enhancements = json.loads(response)
                
                if 'title' in enhancements and enhancements['title']:
                    metadata.title = enhancements['title']
                if 'description' in enhancements:
                    metadata.description = enhancements['description']
                if 'subjects' in enhancements:
                    metadata.subject = enhancements['subjects']
                if 'creator' in enhancements:
                    metadata.creator = enhancements['creator']
                if 'coverage' in enhancements:
                    metadata.coverage = enhancements['coverage']
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
            
        except Exception as e:
            logger.error(f"Error enhancing metadata with LLM: {e}")
        
        return metadata
    
    def validate_metadata(self, metadata: DublinCoreMetadata) -> List[str]:
        """Validate metadata against Dublin Core standards."""
        issues = []
        
        # Required elements
        if not metadata.title:
            issues.append("Missing required element: title")
        
        # Validate date format
        if metadata.date and not isinstance(metadata.date, datetime):
            issues.append("Date should be a datetime object")
        
        # Validate language code
        if metadata.language and len(metadata.language) != 2:
            issues.append("Language should be a 2-letter ISO code")
        
        # Validate identifier format
        if metadata.identifier and not metadata.identifier.startswith('aiq:'):
            issues.append("Identifier should start with 'aiq:' prefix")
        
        return issues
    
    def export_to_xml(self, metadata: DublinCoreMetadata) -> str:
        """Export metadata to Dublin Core XML format."""
        xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{metadata.title}</dc:title>
    {f'<dc:creator>{metadata.creator}</dc:creator>' if metadata.creator else ''}
    {f'<dc:description>{metadata.description}</dc:description>' if metadata.description else ''}
    {f'<dc:date>{metadata.date.isoformat()}</dc:date>' if metadata.date else ''}
    {f'<dc:type>{metadata.type}</dc:type>' if metadata.type else ''}
    {f'<dc:format>{metadata.format}</dc:format>' if metadata.format else ''}
    {f'<dc:identifier>{metadata.identifier}</dc:identifier>' if metadata.identifier else ''}
    {f'<dc:source>{metadata.source}</dc:source>' if metadata.source else ''}
    {f'<dc:language>{metadata.language}</dc:language>' if metadata.language else ''}
    {f'<dc:coverage>{metadata.coverage}</dc:coverage>' if metadata.coverage else ''}
    {f'<dc:rights>{metadata.rights}</dc:rights>' if metadata.rights else ''}
    {''.join(f'<dc:subject>{subject}</dc:subject>' for subject in metadata.subject or [])}
</metadata>"""
        
        return xml_template