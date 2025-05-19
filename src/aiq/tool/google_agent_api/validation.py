# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, ValidationError
import html
import urllib.parse
from dataclasses import dataclass

log = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation for security"""
    
    # Patterns for common attacks
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
        r"(-{2}|\/\*|\*\/|;)",  # SQL comments and statement terminators
        r"(\b(and|or)\b\s*[\d']+=[\d']+)",  # Boolean conditions
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"<object[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|]\s*\w+",  # Command chaining
        r"`[^`]*`",  # Command substitution
        r"\$\([^)]*\)",  # Command substitution
        r">\s*\/\w+",  # Output redirection
    ]
    
    # Safe patterns
    SAFE_AGENT_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
    SAFE_PROJECT_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
    SAFE_LOCATION = re.compile(r"^[a-z0-9-]{1,50}$")
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Truncate to max length
        value = value[:max_length]
        
        # HTML escape
        value = html.escape(value)
        
        # URL encode special characters
        value = urllib.parse.quote(value, safe='')
        
        return value
    
    @classmethod
    def validate_agent_id(cls, agent_id: str) -> str:
        """Validate agent ID format"""
        if not cls.SAFE_AGENT_ID.match(agent_id):
            raise ValueError(f"Invalid agent ID format: {agent_id}")
        return agent_id
    
    @classmethod
    def validate_project_id(cls, project_id: str) -> str:
        """Validate project ID format"""
        if not cls.SAFE_PROJECT_ID.match(project_id):
            raise ValueError(f"Invalid project ID format: {project_id}")
        return project_id
    
    @classmethod
    def validate_location(cls, location: str) -> str:
        """Validate location format"""
        if not cls.SAFE_LOCATION.match(location):
            raise ValueError(f"Invalid location format: {location}")
        return location
    
    @classmethod
    def check_sql_injection(cls, value: str) -> bool:
        """Check for SQL injection patterns"""
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def check_xss(cls, value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def check_command_injection(cls, value: str) -> bool:
        """Check for command injection patterns"""
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def validate_message(cls, message: str, max_length: int = 5000) -> str:
        """Validate user message"""
        if not message or not isinstance(message, str):
            raise ValueError("Message must be a non-empty string")
        
        if len(message) > max_length:
            raise ValueError(f"Message too long: {len(message)} > {max_length}")
        
        # Check for injection attacks
        if cls.check_sql_injection(message):
            raise ValueError("Potential SQL injection detected")
        
        if cls.check_xss(message):
            raise ValueError("Potential XSS detected")
        
        if cls.check_command_injection(message):
            raise ValueError("Potential command injection detected")
        
        return message


# Pydantic models for structured validation

class AgentConfig(BaseModel):
    """Validated agent configuration"""
    agent_id: str = Field(..., min_length=1, max_length=100)
    project_id: str = Field(..., min_length=1, max_length=100)
    location: str = Field(default="us-central1", regex=r"^[a-z0-9-]{1,50}$")
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        return InputValidator.validate_agent_id(v)
    
    @validator('project_id')
    def validate_project_id(cls, v):
        return InputValidator.validate_project_id(v)


class AgentMessage(BaseModel):
    """Validated agent message"""
    message: str = Field(..., min_length=1, max_length=5000)
    context: Optional[Dict[str, Any]] = Field(default=None)
    
    @validator('message')
    def validate_message(cls, v):
        return InputValidator.validate_message(v)
    
    @validator('context')
    def validate_context(cls, v):
        if v is not None:
            # Recursively validate context values
            for key, value in v.items():
                if isinstance(value, str):
                    InputValidator.validate_message(value, max_length=1000)
        return v


class RouteRequest(BaseModel):
    """Validated routing request"""
    message: str = Field(..., min_length=1, max_length=5000)
    target_capabilities: Optional[List[str]] = Field(default=None)
    broadcast: bool = Field(default=False)
    context: Optional[Dict[str, Any]] = Field(default=None)
    
    @validator('message')
    def validate_message(cls, v):
        return InputValidator.validate_message(v)
    
    @validator('target_capabilities')
    def validate_capabilities(cls, v):
        if v is not None:
            for cap in v:
                if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", cap):
                    raise ValueError(f"Invalid capability format: {cap}")
        return v


class ValidationMiddleware:
    """Validation middleware for API requests"""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
    
    async def validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming request"""
        
        try:
            # Extract and validate path parameters
            path = request.get("path", "")
            if not re.match(r"^[a-zA-Z0-9/_-]*$", path):
                raise ValueError("Invalid path format")
            
            # Validate headers
            headers = request.get("headers", {})
            for key, value in headers.items():
                if isinstance(value, str) and len(value) > 1000:
                    raise ValueError(f"Header value too long: {key}")
            
            # Validate body based on endpoint
            body = request.get("body", {})
            validated_body = self._validate_body(path, body)
            
            return {
                **request,
                "body": validated_body,
                "validated": True
            }
            
        except Exception as e:
            log.error(f"Validation error: {e}")
            if self.strict_mode:
                raise
            else:
                return request
    
    def _validate_body(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request body based on endpoint"""
        
        if "/agent/call" in path:
            # Validate agent call request
            validated = AgentMessage(**body)
            return validated.dict()
        
        elif "/agent/route" in path:
            # Validate routing request
            validated = RouteRequest(**body)
            return validated.dict()
        
        elif "/agent/config" in path:
            # Validate agent configuration
            validated = AgentConfig(**body)
            return validated.dict()
        
        else:
            # Generic validation for unknown endpoints
            for key, value in body.items():
                if isinstance(value, str):
                    InputValidator.validate_message(value)
            
            return body