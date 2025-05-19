"""
SSO/SAML integration for AIQToolkit.

This module provides:
- SAML2 authentication support
- OAuth2/OIDC integration
- Enterprise SSO providers (Okta, Azure AD, Google Workspace)
- JWT token mapping
- User provisioning and syncing
"""

import base64
import json
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field
import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate

from aiq.security.auth import (
    User,
    UserRole,
    auth_manager
)
from aiq.security.audit_logger import (
    log_security_event,
    SecurityEventType,
    SeverityLevel
)
from aiq.utils.exception_handlers import (
    AuthenticationError,
    handle_errors,
    async_handle_errors
)


class SAMLConfig(BaseModel):
    """SAML configuration"""
    entity_id: str  # Service Provider entity ID
    sso_url: str  # Identity Provider SSO URL
    slo_url: Optional[str] = None  # Single Logout URL
    certificate: str  # IdP certificate for signature verification
    metadata_url: Optional[str] = None
    attribute_mapping: Dict[str, str] = Field(default_factory=dict)


class OIDCConfig(BaseModel):
    """OpenID Connect configuration"""
    client_id: str
    client_secret: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    scopes: List[str] = ["openid", "profile", "email"]
    redirect_uri: str


class SSOProvider(BaseModel):
    """SSO provider configuration"""
    name: str
    type: str  # "saml" or "oidc"
    enabled: bool = True
    config: Dict[str, Any]
    role_mapping: Dict[str, UserRole] = Field(default_factory=dict)


class SAMLRequest(BaseModel):
    """SAML authentication request"""
    id: str = Field(default_factory=lambda: f"_{''.join(secrets.token_hex(16))}")
    issue_instant: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    assertion_consumer_service_url: str
    issuer: str
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"


class SAMLResponse(BaseModel):
    """SAML authentication response"""
    raw_response: str
    decoded_response: Optional[Dict[str, Any]] = None
    user_attributes: Dict[str, Any] = Field(default_factory=dict)
    name_id: Optional[str] = None
    session_index: Optional[str] = None


class SSOManager:
    """Manage SSO/SAML authentication"""
    
    def __init__(self):
        self.providers: Dict[str, SSOProvider] = {}
        self.saml_requests: Dict[str, SAMLRequest] = {}  # Track pending requests
        self.oidc_states: Dict[str, Dict[str, Any]] = {}  # Track OIDC states
    
    def register_provider(self, provider: SSOProvider):
        """Register an SSO provider"""
        self.providers[provider.name] = provider
        
        log_security_event(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            action="register_sso_provider",
            result="success",
            details={"provider": provider.name, "type": provider.type}
        )
    
    @async_handle_errors(reraise=True)
    async def create_saml_request(
        self,
        provider_name: str,
        relay_state: Optional[str] = None
    ) -> Dict[str, str]:
        """Create SAML authentication request"""
        provider = self._get_provider(provider_name)
        if provider.type != "saml":
            raise ValueError(f"Provider {provider_name} is not a SAML provider")
        
        config = SAMLConfig(**provider.config)
        
        # Create SAML request
        request = SAMLRequest(
            assertion_consumer_service_url=f"/sso/callback/{provider_name}",
            issuer=config.entity_id
        )
        
        # Build XML request
        xml_request = self._build_saml_request_xml(request, config)
        
        # Encode and compress
        encoded_request = base64.b64encode(xml_request.encode()).decode()
        
        # Store request for validation
        self.saml_requests[request.id] = request
        
        # Build redirect URL
        redirect_url = f"{config.sso_url}?SAMLRequest={encoded_request}"
        if relay_state:
            redirect_url += f"&RelayState={relay_state}"
        
        log_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            action="create_saml_request",
            result="success",
            details={
                "provider": provider_name,
                "request_id": request.id
            }
        )
        
        return {
            "redirect_url": redirect_url,
            "request_id": request.id
        }
    
    @async_handle_errors(reraise=True)
    async def process_saml_response(
        self,
        provider_name: str,
        saml_response: str,
        relay_state: Optional[str] = None
    ) -> User:
        """Process SAML response and create/update user"""
        provider = self._get_provider(provider_name)
        if provider.type != "saml":
            raise ValueError(f"Provider {provider_name} is not a SAML provider")
        
        config = SAMLConfig(**provider.config)
        
        # Decode response
        try:
            decoded_response = base64.b64decode(saml_response)
            response_xml = ET.fromstring(decoded_response)
        except Exception as e:
            raise AuthenticationError(f"Invalid SAML response: {e}")
        
        # Verify signature
        if not self._verify_saml_signature(response_xml, config):
            raise AuthenticationError("SAML signature verification failed")
        
        # Extract attributes
        response = SAMLResponse(
            raw_response=saml_response,
            decoded_response=self._xml_to_dict(response_xml)
        )
        
        # Extract user information
        attributes = self._extract_saml_attributes(response_xml, config)
        response.user_attributes = attributes
        response.name_id = self._extract_name_id(response_xml)
        
        # Create or update user
        user = await self._provision_user(
            provider=provider,
            attributes=attributes,
            name_id=response.name_id
        )
        
        log_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            action="process_saml_response",
            result="success",
            user_id=user.user_id,
            details={
                "provider": provider_name,
                "name_id": response.name_id
            }
        )
        
        return user
    
    @async_handle_errors(reraise=True)
    async def create_oidc_request(
        self,
        provider_name: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> Dict[str, str]:
        """Create OIDC authentication request"""
        provider = self._get_provider(provider_name)
        if provider.type != "oidc":
            raise ValueError(f"Provider {provider_name} is not an OIDC provider")
        
        config = OIDCConfig(**provider.config)
        
        # Generate state if not provided
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Store state for validation
        self.oidc_states[state] = {
            "provider": provider_name,
            "redirect_uri": redirect_uri,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Build authorization URL
        auth_url = (
            f"{config.authorization_endpoint}"
            f"?client_id={config.client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri}"
            f"&scope={' '.join(config.scopes)}"
            f"&state={state}"
        )
        
        return {
            "authorization_url": auth_url,
            "state": state
        }
    
    @async_handle_errors(reraise=True)
    async def process_oidc_callback(
        self,
        provider_name: str,
        code: str,
        state: str
    ) -> User:
        """Process OIDC callback and exchange code for tokens"""
        # Validate state
        if state not in self.oidc_states:
            raise AuthenticationError("Invalid state parameter")
        
        state_data = self.oidc_states.pop(state)
        if state_data["provider"] != provider_name:
            raise AuthenticationError("State provider mismatch")
        
        provider = self._get_provider(provider_name)
        config = OIDCConfig(**provider.config)
        
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                config.token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": state_data["redirect_uri"],
                    "client_id": config.client_id,
                    "client_secret": config.client_secret
                }
            )
            
            if token_response.status_code != 200:
                raise AuthenticationError("Failed to exchange code for tokens")
            
            tokens = token_response.json()
        
        # Get user info
        async with httpx.AsyncClient() as client:
            userinfo_response = await client.get(
                config.userinfo_endpoint,
                headers={"Authorization": f"Bearer {tokens['access_token']}"}
            )
            
            if userinfo_response.status_code != 200:
                raise AuthenticationError("Failed to get user info")
            
            userinfo = userinfo_response.json()
        
        # Create or update user
        user = await self._provision_user(
            provider=provider,
            attributes=userinfo,
            name_id=userinfo.get("email") or userinfo.get("sub")
        )
        
        log_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            action="process_oidc_callback",
            result="success",
            user_id=user.user_id,
            details={
                "provider": provider_name,
                "sub": userinfo.get("sub")
            }
        )
        
        return user
    
    def _get_provider(self, name: str) -> SSOProvider:
        """Get SSO provider by name"""
        if name not in self.providers:
            raise ValueError(f"Unknown SSO provider: {name}")
        
        provider = self.providers[name]
        if not provider.enabled:
            raise ValueError(f"SSO provider {name} is disabled")
        
        return provider
    
    def _build_saml_request_xml(
        self,
        request: SAMLRequest,
        config: SAMLConfig
    ) -> str:
        """Build SAML request XML"""
        xml = f"""
        <samlp:AuthnRequest
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request.id}"
            Version="2.0"
            IssueInstant="{request.issue_instant}"
            AssertionConsumerServiceURL="{request.assertion_consumer_service_url}">
            <saml:Issuer>{request.issuer}</saml:Issuer>
            <samlp:NameIDPolicy Format="{request.name_id_format}"/>
        </samlp:AuthnRequest>
        """
        return xml.strip()
    
    def _verify_saml_signature(
        self,
        response_xml: ET.Element,
        config: SAMLConfig
    ) -> bool:
        """Verify SAML response signature"""
        # This is a simplified version - production would need full XML signature verification
        # using libraries like signxml or xmlsec
        try:
            # Load certificate
            cert_pem = f"-----BEGIN CERTIFICATE-----\n{config.certificate}\n-----END CERTIFICATE-----"
            cert = load_pem_x509_certificate(cert_pem.encode())
            
            # In production, implement full XML signature verification
            # For now, we'll just check that a signature exists
            signature_elem = response_xml.find(".//{http://www.w3.org/2000/09/xmldsig#}Signature")
            return signature_elem is not None
        except Exception:
            return False
    
    def _extract_saml_attributes(
        self,
        response_xml: ET.Element,
        config: SAMLConfig
    ) -> Dict[str, Any]:
        """Extract attributes from SAML response"""
        attributes = {}
        
        # Find all attribute statements
        for attr_statement in response_xml.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement"):
            for attr in attr_statement.findall("{urn:oasis:names:tc:SAML:2.0:assertion}Attribute"):
                name = attr.get("Name")
                values = []
                
                for value_elem in attr.findall("{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue"):
                    values.append(value_elem.text)
                
                if values:
                    # Map attributes using config
                    mapped_name = config.attribute_mapping.get(name, name)
                    attributes[mapped_name] = values[0] if len(values) == 1 else values
        
        return attributes
    
    def _extract_name_id(self, response_xml: ET.Element) -> Optional[str]:
        """Extract NameID from SAML response"""
        name_id_elem = response_xml.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}NameID")
        if name_id_elem is not None:
            return name_id_elem.text
        return None
    
    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result["@attributes"] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result["#text"] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            
            if tag in result:
                # Convert to list if multiple elements with same tag
                if not isinstance(result[tag], list):
                    result[tag] = [result[tag]]
                result[tag].append(child_data)
            else:
                result[tag] = child_data
        
        return result
    
    async def _provision_user(
        self,
        provider: SSOProvider,
        attributes: Dict[str, Any],
        name_id: str
    ) -> User:
        """Create or update user from SSO attributes"""
        # Extract user information
        email = attributes.get("email") or name_id
        username = attributes.get("username") or email.split("@")[0]
        
        # Check if user exists
        existing_user = None
        for user in auth_manager.users.values():
            if user.email == email:
                existing_user = user
                break
        
        if existing_user:
            # Update existing user
            user = existing_user
            
            # Update roles if specified in provider config
            if provider.role_mapping:
                for attr_value, role in provider.role_mapping.items():
                    if attr_value in attributes.values():
                        if role not in user.roles:
                            user.roles.append(role)
            
            log_security_event(
                event_type=SecurityEventType.ACCESS_CONTROL,
                action="update_sso_user",
                result="success",
                user_id=user.user_id,
                details={"provider": provider.name}
            )
        else:
            # Create new user
            roles = [UserRole.USER]  # Default role
            
            # Apply role mapping
            if provider.role_mapping:
                for attr_value, role in provider.role_mapping.items():
                    if attr_value in attributes.values():
                        roles.append(role)
            
            user = auth_manager.create_user(
                username=username,
                email=email,
                password=secrets.token_urlsafe(32),  # Random password for SSO users
                roles=roles
            )
            
            log_security_event(
                event_type=SecurityEventType.ACCESS_CONTROL,
                action="create_sso_user",
                result="success",
                user_id=user.user_id,
                details={"provider": provider.name}
            )
        
        return user


# Preset provider configurations
PRESET_PROVIDERS = {
    "okta": {
        "type": "saml",
        "config": {
            "entity_id": "http://localhost:8080",
            "sso_url": "https://{domain}.okta.com/app/{app_id}/sso/saml",
            "slo_url": "https://{domain}.okta.com/app/{app_id}/slo/saml",
            "certificate": "{certificate}",
            "attribute_mapping": {
                "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": "email",
                "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name": "username"
            }
        }
    },
    "azure_ad": {
        "type": "oidc",
        "config": {
            "client_id": "{client_id}",
            "client_secret": "{client_secret}",
            "issuer": "https://login.microsoftonline.com/{tenant_id}/v2.0",
            "authorization_endpoint": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
            "token_endpoint": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            "userinfo_endpoint": "https://graph.microsoft.com/oidc/userinfo",
            "jwks_uri": "https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
            "scopes": ["openid", "profile", "email", "User.Read"]
        }
    },
    "google": {
        "type": "oidc",
        "config": {
            "client_id": "{client_id}",
            "client_secret": "{client_secret}",
            "issuer": "https://accounts.google.com",
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
            "scopes": ["openid", "profile", "email"]
        }
    }
}


# Global SSO manager instance
sso_manager = SSOManager()


# FastAPI endpoints
from fastapi import APIRouter

sso_router = APIRouter(prefix="/sso", tags=["SSO"])


@sso_router.get("/login/{provider}")
async def sso_login(provider: str, request: Request):
    """Initiate SSO login"""
    if provider not in sso_manager.providers:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    
    provider_config = sso_manager.providers[provider]
    
    if provider_config.type == "saml":
        result = await sso_manager.create_saml_request(provider)
        return {"redirect_url": result["redirect_url"]}
    elif provider_config.type == "oidc":
        redirect_uri = str(request.url).replace("/login/", "/callback/")
        result = await sso_manager.create_oidc_request(provider, redirect_uri)
        return {"redirect_url": result["authorization_url"]}
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider type")


@sso_router.post("/callback/{provider}")
async def sso_callback_saml(
    provider: str,
    request: Request,
    SAMLResponse: str = None,
    RelayState: str = None
):
    """Handle SAML callback"""
    if not SAMLResponse:
        raise HTTPException(status_code=400, detail="Missing SAML response")
    
    try:
        user = await sso_manager.process_saml_response(provider, SAMLResponse, RelayState)
        
        # Create session
        auth_result = await auth_manager.authenticate(
            username=user.username,
            password="",  # SSO bypass
            ip_address=request.client.host if request.client else None
        )
        
        return auth_result
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


@sso_router.get("/callback/{provider}")
async def sso_callback_oidc(
    provider: str,
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None
):
    """Handle OIDC callback"""
    if error:
        raise HTTPException(status_code=400, detail=f"Authentication error: {error}")
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")
    
    try:
        user = await sso_manager.process_oidc_callback(provider, code, state)
        
        # Create session
        auth_result = await auth_manager.authenticate(
            username=user.username,
            password="",  # SSO bypass
            ip_address=request.client.host if request.client else None
        )
        
        return auth_result
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


# Export public interface
__all__ = [
    "SSOManager",
    "SSOProvider",
    "SAMLConfig",
    "OIDCConfig",
    "sso_manager",
    "sso_router",
    "PRESET_PROVIDERS"
]