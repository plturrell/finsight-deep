"""Input validation for document management system."""

import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, validator, Field
from fastapi import HTTPException, status
from pathlib import Path
import mimetypes


# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {
    'pdf': ['application/pdf'],
    'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    'xls': ['application/vnd.ms-excel'],
    'csv': ['text/csv'],
    'txt': ['text/plain'],
    'md': ['text/markdown'],
    'json': ['application/json'],
    'xml': ['application/xml', 'text/xml']
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_TITLE_LENGTH = 255
MAX_DESCRIPTION_LENGTH = 5000
MAX_SUBJECT_LENGTH = 50
MAX_SUBJECTS = 20


class DocumentUploadRequest(BaseModel):
    """Validated document upload request."""
    title: Optional[str] = Field(None, max_length=MAX_TITLE_LENGTH)
    creator: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=MAX_DESCRIPTION_LENGTH)
    subject: Optional[List[str]] = Field(None, max_items=MAX_SUBJECTS)
    
    @validator('title')
    def validate_title(cls, v):
        if v and not v.strip():
            raise ValueError("Title cannot be empty")
        if v and len(v.strip()) < 3:
            raise ValueError("Title must be at least 3 characters long")
        return v.strip() if v else v
    
    @validator('creator')
    def validate_creator(cls, v):
        if v and not re.match(r'^[\w\s\-\.]+$', v):
            raise ValueError("Creator name contains invalid characters")
        return v
    
    @validator('subject')
    def validate_subjects(cls, v):
        if v:
            # Validate each subject
            validated = []
            for subj in v:
                if not subj.strip():
                    continue
                if len(subj) > MAX_SUBJECT_LENGTH:
                    raise ValueError(f"Subject '{subj}' exceeds maximum length")
                validated.append(subj.strip().lower())
            return validated
        return v


class DocumentSearchRequest(BaseModel):
    """Validated document search request."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(10, ge=1, le=100)
    search_type: str = Field("hybrid", regex='^(hybrid|semantic|keyword)$')
    filters: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def validate_query(cls, v):
        # Remove potentially dangerous characters
        cleaned = re.sub(r'[<>\"\'`;]', '', v)
        if not cleaned.strip():
            raise ValueError("Query cannot be empty")
        return cleaned.strip()
    
    @validator('filters')
    def validate_filters(cls, v):
        if v:
            allowed_filters = {
                'type', 'status', 'creator', 'date_range',
                'subject', 'tenant_id', 'language'
            }
            
            for key in v.keys():
                if key not in allowed_filters:
                    raise ValueError(f"Invalid filter: {key}")
            
            # Validate date range format
            if 'date_range' in v:
                date_range = v['date_range']
                if not re.match(r'^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$', date_range):
                    raise ValueError("Invalid date range format. Use YYYY-MM-DD/YYYY-MM-DD")
        
        return v


class CrawlRequest(BaseModel):
    """Validated web crawl request."""
    url: str = Field(..., regex=r'^https?://[\w\-\.]+(:\d+)?(/.*)?$')
    max_depth: int = Field(3, ge=1, le=10)
    max_pages: int = Field(100, ge=1, le=1000)
    keywords: Optional[List[str]] = Field(None, max_items=50)
    
    @validator('url')
    def validate_url(cls, v):
        # Additional URL validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        
        # Block internal/private IPs
        private_patterns = [
            r'127\.0\.0\.\d+',
            r'localhost',
            r'10\.\d+\.\d+\.\d+',
            r'172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+',
            r'192\.168\.\d+\.\d+'
        ]
        
        for pattern in private_patterns:
            if re.search(pattern, v):
                raise ValueError("Cannot crawl internal/private URLs")
        
        return v
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if v:
            validated = []
            for keyword in v:
                if keyword and len(keyword) > 2:
                    validated.append(keyword.strip().lower())
            return validated
        return v


def validate_document_upload(file: Any) -> bool:
    """Validate uploaded document file."""
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
    
    # Check file extension
    filename = getattr(file, 'filename', '')
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    ext = Path(filename).suffix.lower()[1:]  # Remove the dot
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{ext}' is not allowed. Allowed types: {list(ALLOWED_EXTENSIONS.keys())}"
        )
    
    # Check MIME type if available
    content_type = getattr(file, 'content_type', None)
    if content_type:
        allowed_mimes = ALLOWED_EXTENSIONS.get(ext, [])
        if content_type not in allowed_mimes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid MIME type '{content_type}' for extension '{ext}'"
            )
    
    return True


def validate_search_query(query: str) -> str:
    """Validate and sanitize search query."""
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty"
        )
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>\"\'`;]', '', query)
    
    # Limit query length
    if len(sanitized) > 500:
        sanitized = sanitized[:500]
    
    return sanitized.strip()


def validate_sparql_query(query: str) -> str:
    """Validate SPARQL query for safety."""
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SPARQL query cannot be empty"
        )
    
    # Block dangerous operations
    dangerous_keywords = [
        'INSERT', 'DELETE', 'DROP', 'CREATE', 'CLEAR',
        'LOAD', 'COPY', 'MOVE', 'ADD'
    ]
    
    query_upper = query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation '{keyword}' is not allowed in queries"
            )
    
    # Ensure it's a SELECT query
    if not query_upper.strip().startswith('SELECT'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only SELECT queries are allowed"
        )
    
    return query


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate document metadata."""
    if not metadata:
        return {}
    
    # Define allowed metadata fields and their types
    allowed_fields = {
        'title': str,
        'creator': str,
        'subject': list,
        'description': str,
        'publisher': str,
        'contributor': list,
        'date': str,
        'type': str,
        'format': str,
        'identifier': str,
        'source': str,
        'language': str,
        'relation': list,
        'coverage': str,
        'rights': str
    }
    
    validated = {}
    
    for key, value in metadata.items():
        if key not in allowed_fields:
            continue  # Skip unknown fields
        
        expected_type = allowed_fields[key]
        
        # Type validation
        if expected_type == str and isinstance(value, str):
            validated[key] = value.strip()[:1000]  # Limit string length
        elif expected_type == list and isinstance(value, list):
            # Validate list items
            validated[key] = [str(item).strip()[:100] for item in value if item][:50]
        elif expected_type == str and value is not None:
            validated[key] = str(value).strip()[:1000]
    
    return validated


def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID format."""
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(uuid_string))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove path components
    filename = Path(filename).name
    
    # Replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem[:200], Path(filename).suffix
        filename = f"{name}{ext}"
    
    return filename