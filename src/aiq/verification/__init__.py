# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIQ Toolkit Verification System

This module provides real-time citation verification with W3C PROV-compliant
provenance tracking and multi-method confidence scoring.
"""

from aiq.verification.verification_system import (
    VerificationSystem,
    VerificationResult,
    ConfidenceMethod,
    SourceType,
    Source,
    ProvenanceRecord,
    create_verification_system
)

__all__ = [
    'VerificationSystem',
    'VerificationResult',
    'ConfidenceMethod',
    'SourceType',
    'Source',
    'ProvenanceRecord',
    'create_verification_system'
]