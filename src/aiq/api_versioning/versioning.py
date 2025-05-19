"""
API Versioning module for AIQToolkit.

This module provides comprehensive API versioning capabilities including:
- Version management and routing
- Backward compatibility support
- API deprecation warnings
- Version negotiation
- API migration utilities
"""

from typing import Dict, List, Optional, Callable, Any, Union
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta
import warnings
import re

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aiq.utils.exception_handlers import ValidationError, handle_errors
from aiq.data_models.api_server import APIVersion


class APIVersionStatus(str, Enum):
    """API version lifecycle status"""
    ALPHA = "alpha"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class VersionInfo(BaseModel):
    """Information about an API version"""
    version: str
    status: APIVersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    description: str
    breaking_changes: List[str] = []
    new_features: List[str] = []


class APIVersionManager:
    """Manages API versions and routing"""
    
    def __init__(self, default_version: str = "v1"):
        self.versions: Dict[str, VersionInfo] = {}
        self.default_version = default_version
        self.version_validators: Dict[str, Callable] = {}
        self.migration_handlers: Dict[tuple, Callable] = {}
    
    def register_version(
        self,
        version: str,
        status: APIVersionStatus,
        release_date: datetime,
        description: str,
        deprecation_date: Optional[datetime] = None,
        sunset_date: Optional[datetime] = None,
        breaking_changes: List[str] = None,
        new_features: List[str] = None
    ):
        """Register a new API version"""
        if not re.match(r"^v\d+(\.\d+)?$", version):
            raise ValidationError(f"Invalid version format: {version}. Use 'v1' or 'v1.0' format")
        
        self.versions[version] = VersionInfo(
            version=version,
            status=status,
            release_date=release_date,
            deprecation_date=deprecation_date,
            sunset_date=sunset_date,
            description=description,
            breaking_changes=breaking_changes or [],
            new_features=new_features or []
        )
    
    def get_version_info(self, version: str) -> VersionInfo:
        """Get information about a specific version"""
        if version not in self.versions:
            raise ValidationError(f"Unknown API version: {version}")
        return self.versions[version]
    
    def is_version_supported(self, version: str) -> bool:
        """Check if a version is currently supported"""
        if version not in self.versions:
            return False
        
        info = self.versions[version]
        if info.status == APIVersionStatus.SUNSET:
            return False
        
        if info.sunset_date and datetime.now() >= info.sunset_date:
            return False
        
        return True
    
    def get_supported_versions(self) -> List[str]:
        """Get list of currently supported versions"""
        return [v for v in self.versions.keys() if self.is_version_supported(v)]
    
    def get_latest_stable_version(self) -> str:
        """Get the latest stable version"""
        stable_versions = [
            v for v, info in self.versions.items()
            if info.status == APIVersionStatus.STABLE
        ]
        if not stable_versions:
            return self.default_version
        return sorted(stable_versions, reverse=True)[0]
    
    def register_migration(
        self,
        from_version: str,
        to_version: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Register a migration handler between versions"""
        self.migration_handlers[(from_version, to_version)] = handler
    
    def migrate_data(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Migrate data from one version to another"""
        if from_version == to_version:
            return data
        
        # Find migration path
        migration_key = (from_version, to_version)
        if migration_key in self.migration_handlers:
            return self.migration_handlers[migration_key](data)
        
        # Try to find multi-step migration
        # This is a simplified version - production would need graph traversal
        raise ValidationError(
            f"No migration path found from {from_version} to {to_version}"
        )


# Decorators for versioned endpoints
def versioned_endpoint(
    supported_versions: List[str],
    deprecated_in: Optional[str] = None,
    removed_in: Optional[str] = None
):
    """Decorator for versioned API endpoints"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Extract version from request path or header
            version = extract_version_from_request(request)
            
            if version not in supported_versions:
                raise HTTPException(
                    status_code=406,
                    detail=f"Version {version} not supported for this endpoint"
                )
            
            # Add deprecation warning if applicable
            if deprecated_in and version >= deprecated_in:
                warnings.warn(
                    f"Endpoint is deprecated in version {deprecated_in}",
                    DeprecationWarning
                )
                # Add deprecation header to response
                response = await func(request, *args, **kwargs)
                if isinstance(response, JSONResponse):
                    response.headers["X-API-Deprecation-Warning"] = (
                        f"Deprecated in {deprecated_in}"
                    )
                return response
            
            return await func(request, *args, **kwargs)
        
        # Add metadata for documentation
        wrapper._api_versions = supported_versions
        wrapper._deprecated_in = deprecated_in
        wrapper._removed_in = removed_in
        
        return wrapper
    return decorator


def extract_version_from_request(request: Request) -> str:
    """Extract API version from request"""
    # Check path for version (e.g., /v1/endpoint)
    path_parts = request.url.path.split("/")
    for part in path_parts:
        if re.match(r"^v\d+(\.\d+)?$", part):
            return part
    
    # Check header for version
    if "X-API-Version" in request.headers:
        return request.headers["X-API-Version"]
    
    # Check query parameter
    if "api_version" in request.query_params:
        return request.query_params["api_version"]
    
    # Default to latest stable version
    return version_manager.get_latest_stable_version()


# Version negotiation
class VersionNegotiator:
    """Handle API version negotiation"""
    
    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager
    
    def negotiate_version(
        self,
        requested_version: Optional[str],
        accept_versions: List[str]
    ) -> str:
        """Negotiate the best version to use"""
        supported = self.version_manager.get_supported_versions()
        
        # If specific version requested, use it if supported
        if requested_version:
            if requested_version in supported:
                return requested_version
            raise ValidationError(f"Requested version {requested_version} not supported")
        
        # Find best match from accept versions
        for version in accept_versions:
            if version in supported:
                return version
        
        # Default to latest stable
        return self.version_manager.get_latest_stable_version()


# Response transformation
class ResponseTransformer:
    """Transform responses for different API versions"""
    
    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager
        self.transformers: Dict[str, Dict[str, Callable]] = {}
    
    def register_transformer(
        self,
        version: str,
        endpoint: str,
        transformer: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Register a response transformer for a specific version and endpoint"""
        if version not in self.transformers:
            self.transformers[version] = {}
        self.transformers[version][endpoint] = transformer
    
    def transform_response(
        self,
        response: Dict[str, Any],
        version: str,
        endpoint: str
    ) -> Dict[str, Any]:
        """Transform response for the specified version"""
        if version in self.transformers and endpoint in self.transformers[version]:
            return self.transformers[version][endpoint](response)
        return response


# Middleware for API versioning
class APIVersionMiddleware:
    """Middleware to handle API versioning"""
    
    def __init__(
        self,
        app: FastAPI,
        version_manager: APIVersionManager,
        negotiator: VersionNegotiator,
        transformer: ResponseTransformer
    ):
        self.app = app
        self.version_manager = version_manager
        self.negotiator = negotiator
        self.transformer = transformer
    
    async def __call__(self, request: Request, call_next):
        # Extract version from request
        version = extract_version_from_request(request)
        
        # Check if version is supported
        if not self.version_manager.is_version_supported(version):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UnsupportedVersion",
                    "message": f"API version {version} is not supported",
                    "supported_versions": self.version_manager.get_supported_versions()
                }
            )
        
        # Add version to request state
        request.state.api_version = version
        
        # Check for deprecation
        version_info = self.version_manager.get_version_info(version)
        response = await call_next(request)
        
        # Add version headers
        response.headers["X-API-Version"] = version
        response.headers["X-API-Supported-Versions"] = ",".join(
            self.version_manager.get_supported_versions()
        )
        
        # Add deprecation warning if applicable
        if version_info.status == APIVersionStatus.DEPRECATED:
            response.headers["X-API-Deprecation-Warning"] = (
                f"Version {version} is deprecated. "
                f"Please upgrade to {self.version_manager.get_latest_stable_version()}"
            )
            if version_info.sunset_date:
                response.headers["X-API-Sunset-Date"] = (
                    version_info.sunset_date.isoformat()
                )
        
        return response


# Global instance
version_manager = APIVersionManager()

# Register default versions
version_manager.register_version(
    version="v1",
    status=APIVersionStatus.STABLE,
    release_date=datetime(2024, 1, 1),
    description="Initial stable API version",
    new_features=[
        "Basic workflow management",
        "LLM integration",
        "Function execution"
    ]
)

version_manager.register_version(
    version="v2",
    status=APIVersionStatus.BETA,
    release_date=datetime(2024, 10, 1),
    description="Enhanced API with streaming support",
    new_features=[
        "WebSocket streaming",
        "Batch operations",
        "Enhanced error reporting"
    ],
    breaking_changes=[
        "Changed response format for workflow execution",
        "Renamed 'run' endpoint to 'execute'"
    ]
)


# Export public interface
__all__ = [
    "APIVersionManager",
    "APIVersionStatus",
    "VersionInfo",
    "versioned_endpoint",
    "extract_version_from_request",
    "VersionNegotiator",
    "ResponseTransformer",
    "APIVersionMiddleware",
    "version_manager"
]