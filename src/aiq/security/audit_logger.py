"""
Audit logging module for security events.

This module provides comprehensive audit logging capabilities including:
- Security event tracking
- Access control logging
- API call auditing
- Compliance reporting
- Threat detection
"""

import json
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import asyncio
import threading
from collections import deque
import uuid

from pydantic import BaseModel, Field
import aiofiles

from aiq.utils.exception_handlers import (
    handle_errors,
    async_handle_errors,
    ErrorContext
)


class SecurityEventType(str, Enum):
    """Types of security events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_CONTROL = "access_control"
    DATA_ACCESS = "data_access"
    API_CALL = "api_call"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_ALERT = "security_alert"
    ERROR = "error"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class SeverityLevel(str, Enum):
    """Security event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Audit event model"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: SecurityEventType
    severity: SeverityLevel
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    resource: Optional[str] = None
    action: str
    result: str
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.dict(), default=str)


class AuditLogger:
    """Thread-safe audit logger with async support"""
    
    def __init__(
        self,
        log_file: str = "audit.log",
        max_queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        enable_encryption: bool = False,
        encryption_key: Optional[bytes] = None
    ):
        self.log_file = Path(log_file)
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enable_encryption = enable_encryption
        self.encryption_key = encryption_key
        
        # Thread-safe queue for events
        self.event_queue = deque(maxlen=max_queue_size)
        self.queue_lock = threading.Lock()
        
        # Background task for writing logs
        self.writer_task = None
        self.stop_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_dropped": 0,
            "write_errors": 0,
            "last_write": None
        }
    
    async def start(self):
        """Start the background writer task"""
        self.writer_task = asyncio.create_task(self._writer_loop())
    
    async def stop(self):
        """Stop the logger and flush remaining events"""
        self.stop_event.set()
        if self.writer_task:
            await self.writer_task
        await self.flush()
    
    @handle_errors(default_return=None)
    def log_event(
        self,
        event_type: SecurityEventType,
        action: str,
        result: str,
        severity: SeverityLevel = SeverityLevel.INFO,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None
    ):
        """Log a security event (sync version)"""
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            error_message=error_message,
            stack_trace=stack_trace
        )
        
        with self.queue_lock:
            if len(self.event_queue) >= self.max_queue_size:
                self.stats["events_dropped"] += 1
                # Remove oldest event to make room
                self.event_queue.popleft()
            self.event_queue.append(event)
            self.stats["events_logged"] += 1
    
    @async_handle_errors(default_return=None)
    async def async_log_event(self, **kwargs):
        """Log a security event (async version)"""
        # Delegate to sync version in thread-safe manner
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.log_event(**kwargs)
        )
    
    async def _writer_loop(self):
        """Background task to write events to disk"""
        while not self.stop_event.is_set():
            try:
                await self._write_batch()
                await asyncio.sleep(self.flush_interval)
            except Exception as e:
                self.stats["write_errors"] += 1
                # Log error but continue
                print(f"Error in audit writer: {e}")
    
    async def _write_batch(self):
        """Write a batch of events to disk"""
        events_to_write = []
        
        with self.queue_lock:
            for _ in range(min(self.batch_size, len(self.event_queue))):
                if self.event_queue:
                    events_to_write.append(self.event_queue.popleft())
        
        if not events_to_write:
            return
        
        # Prepare log entries
        log_entries = []
        for event in events_to_write:
            entry = event.to_json()
            if self.enable_encryption:
                entry = self._encrypt_entry(entry)
            log_entries.append(entry + "\n")
        
        # Write to file
        async with aiofiles.open(self.log_file, "a") as f:
            await f.writelines(log_entries)
        
        self.stats["last_write"] = datetime.now(timezone.utc).isoformat()
    
    async def flush(self):
        """Flush all pending events"""
        while self.event_queue:
            await self._write_batch()
    
    def _encrypt_entry(self, entry: str) -> str:
        """Encrypt log entry if encryption is enabled"""
        if not self.encryption_key:
            return entry
        
        # Simple encryption using hashlib (replace with proper encryption in production)
        iv = hashlib.sha256(str(time.time()).encode()).digest()[:16]
        # This is a placeholder - use proper encryption library
        encrypted = base64.b64encode(iv + entry.encode()).decode()
        return f"ENCRYPTED:{encrypted}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        with self.queue_lock:
            queue_size = len(self.event_queue)
        
        return {
            **self.stats,
            "queue_size": queue_size,
            "max_queue_size": self.max_queue_size
        }


class ComplianceReporter:
    """Generate compliance reports from audit logs"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.report_formats = {
            "soc2": self._generate_soc2_report,
            "hipaa": self._generate_hipaa_report,
            "gdpr": self._generate_gdpr_report,
            "pci_dss": self._generate_pci_dss_report
        }
    
    async def generate_report(
        self,
        report_type: str,
        start_date: datetime,
        end_date: datetime,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        if report_type not in self.report_formats:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # Read and filter audit logs
        events = await self._read_events(start_date, end_date)
        
        # Generate report
        report = await self.report_formats[report_type](events)
        
        # Save if output file specified
        if output_file:
            await self._save_report(report, output_file)
        
        return report
    
    async def _read_events(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AuditEvent]:
        """Read events from audit log within date range"""
        events = []
        
        async with aiofiles.open(self.audit_logger.log_file, "r") as f:
            async for line in f:
                try:
                    if line.startswith("ENCRYPTED:"):
                        # Decrypt if needed
                        line = self._decrypt_entry(line)
                    
                    event_data = json.loads(line.strip())
                    event = AuditEvent(**event_data)
                    
                    # Filter by date
                    event_time = datetime.fromisoformat(event.timestamp)
                    if start_date <= event_time <= end_date:
                        events.append(event)
                except Exception:
                    # Skip malformed entries
                    continue
        
        return events
    
    async def _generate_soc2_report(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate SOC2 compliance report"""
        return {
            "report_type": "SOC2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": min(e.timestamp for e in events) if events else None,
                "end": max(e.timestamp for e in events) if events else None
            },
            "security_events": {
                "total": len(events),
                "by_type": self._count_by_type(events),
                "by_severity": self._count_by_severity(events)
            },
            "access_control": self._analyze_access_control(events),
            "data_security": self._analyze_data_security(events),
            "availability": self._analyze_availability(events),
            "confidentiality": self._analyze_confidentiality(events)
        }
    
    async def _generate_hipaa_report(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate HIPAA compliance report"""
        return {
            "report_type": "HIPAA",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phi_access": self._analyze_phi_access(events),
            "security_incidents": self._analyze_security_incidents(events),
            "access_controls": self._analyze_hipaa_access_controls(events),
            "audit_controls": self._analyze_audit_controls(events)
        }
    
    async def _generate_gdpr_report(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        return {
            "report_type": "GDPR",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_processing": self._analyze_data_processing(events),
            "consent_management": self._analyze_consent(events),
            "data_breaches": self._analyze_breaches(events),
            "data_subject_requests": self._analyze_dsr(events)
        }
    
    async def _generate_pci_dss_report(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate PCI-DSS compliance report"""
        return {
            "report_type": "PCI-DSS",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cardholder_data_access": self._analyze_cardholder_access(events),
            "network_security": self._analyze_network_security(events),
            "vulnerability_management": self._analyze_vulnerabilities(events),
            "access_control_measures": self._analyze_pci_access_control(events)
        }
    
    def _count_by_type(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by type"""
        counts = {}
        for event in events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
    
    def _count_by_severity(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by severity"""
        counts = {}
        for event in events:
            counts[event.severity] = counts.get(event.severity, 0) + 1
        return counts
    
    # Additional analysis methods would be implemented here
    def _analyze_access_control(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze access control events"""
        access_events = [e for e in events if e.event_type == SecurityEventType.ACCESS_CONTROL]
        return {
            "total_access_attempts": len(access_events),
            "successful": len([e for e in access_events if e.result == "success"]),
            "failed": len([e for e in access_events if e.result == "failure"]),
            "unique_users": len(set(e.user_id for e in access_events if e.user_id))
        }
    
    def _analyze_data_security(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze data security events"""
        data_events = [e for e in events if e.event_type == SecurityEventType.DATA_ACCESS]
        return {
            "data_access_events": len(data_events),
            "sensitive_data_access": len([e for e in data_events if "sensitive" in e.details.get("tags", [])]),
            "encryption_events": len([e for e in events if "encryption" in e.action.lower()])
        }
    
    async def _save_report(self, report: Dict[str, Any], output_file: str):
        """Save report to file"""
        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(report, indent=2))


# Global audit logger instance
audit_logger = AuditLogger()

# Convenience functions
def log_security_event(**kwargs):
    """Log a security event"""
    audit_logger.log_event(**kwargs)


async def async_log_security_event(**kwargs):
    """Log a security event asynchronously"""
    await audit_logger.async_log_event(**kwargs)


# Export public interface
__all__ = [
    "AuditLogger",
    "ComplianceReporter",
    "SecurityEventType",
    "SeverityLevel",
    "AuditEvent",
    "audit_logger",
    "log_security_event",
    "async_log_security_event"
]