# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from sklearn.mixture import GaussianMixture
import torch
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfidenceMethod(Enum):
    """Methods for calculating confidence scores"""
    BAYESIAN = "bayesian"
    FUZZY = "fuzzy"
    DEMPSTER_SHAFER = "dempster_shafer"
    ENSEMBLE = "ensemble"

class SourceType(Enum):
    """Types of sources for verification"""
    PAPER = "paper"
    BOOK = "book"
    WEBSITE = "website"
    DATABASE = "database"
    API = "api"
    EXPERT = "expert"

@dataclass
class Source:
    """Source information for verification"""
    url: str
    type: SourceType
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None
    reliability_score: float = 0.5

@dataclass
class ProvenanceRecord:
    """W3C PROV-compliant provenance record"""
    entity_id: str
    activity_id: str
    agent_id: str
    timestamp: datetime
    derivation: Optional[str] = None
    attribution: Optional[str] = None

@dataclass
class VerificationResult:
    """Result of claim verification"""
    claim: str
    confidence: float
    sources_verified: List[Source]
    method_scores: Dict[ConfidenceMethod, float]
    provenance_chain: List[ProvenanceRecord]
    verification_time_ms: float
    gpu_utilization: float
    explanations: Dict[str, str]

class BayesianConfidenceScorer:
    """Implements Bayesian confidence scoring"""
    
    def __init__(self):
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
    
    def score(self, evidence: List[float]) -> float:
        """
        Calculate Bayesian confidence score
        
        Args:
            evidence: List of evidence values (0-1)
        
        Returns:
            Confidence score (0-1)
        """
        if not evidence:
            return 0.5
        
        # Update beta distribution parameters
        successes = sum(evidence)
        failures = len(evidence) - successes
        
        alpha = self.prior_alpha + successes
        beta_param = self.prior_beta + failures
        
        # Return mean of posterior distribution
        return beta.mean(alpha, beta_param)

class FuzzyLogicScorer:
    """Implements fuzzy logic confidence scoring"""
    
    def __init__(self):
        self.membership_functions = {
            'low': lambda x: max(0, 1 - x/0.5),
            'medium': lambda x: max(0, min(x/0.5, (1-x)/0.5)),
            'high': lambda x: max(0, (x-0.5)/0.5)
        }
    
    def score(self, evidence: List[float]) -> float:
        """
        Calculate fuzzy logic confidence score
        
        Args:
            evidence: List of evidence values (0-1)
        
        Returns:
            Confidence score (0-1)
        """
        if not evidence:
            return 0.5
        
        # Apply fuzzy rules
        memberships = {
            'low': [],
            'medium': [],
            'high': []
        }
        
        for e in evidence:
            for level, func in self.membership_functions.items():
                memberships[level].append(func(e))
        
        # Aggregate using centroid defuzzification
        low_weight = np.mean(memberships['low']) * 0.2
        medium_weight = np.mean(memberships['medium']) * 0.5
        high_weight = np.mean(memberships['high']) * 0.9
        
        total_weight = low_weight + medium_weight + high_weight
        if total_weight == 0:
            return 0.5
        
        return (low_weight * 0.2 + medium_weight * 0.5 + high_weight * 0.9) / total_weight

class DempsterShaferScorer:
    """Implements Dempster-Shafer evidence theory"""
    
    def __init__(self):
        self.frame_of_discernment = {'true', 'false', 'uncertain'}
    
    def combine_masses(self, m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """
        Combine two mass functions using Dempster's rule
        
        Args:
            m1: First mass function
            m2: Second mass function
        
        Returns:
            Combined mass function
        """
        combined = {}
        normalization = 1.0
        
        for s1, v1 in m1.items():
            for s2, v2 in m2.items():
                # Find intersection of focal elements
                intersection = self._intersect(s1, s2)
                
                if intersection:
                    if intersection not in combined:
                        combined[intersection] = 0
                    combined[intersection] += v1 * v2
                else:
                    # Conflict mass
                    normalization -= v1 * v2
        
        # Normalize to handle conflict
        if normalization > 0:
            for key in combined:
                combined[key] /= normalization
        
        return combined
    
    def _intersect(self, s1: str, s2: str) -> Optional[str]:
        """Find intersection of two focal elements"""
        if s1 == s2:
            return s1
        if s1 == 'uncertain' or s2 == 'uncertain':
            return s1 if s2 == 'uncertain' else s2
        return None
    
    def score(self, mass_functions: List[Dict[str, float]]) -> float:
        """
        Calculate Dempster-Shafer confidence score
        
        Args:
            mass_functions: List of mass functions
        
        Returns:
            Confidence score (0-1)
        """
        if not mass_functions:
            return 0.5
        
        # Combine all mass functions
        result = mass_functions[0]
        for mf in mass_functions[1:]:
            result = self.combine_masses(result, mf)
        
        # Return belief in 'true'
        return result.get('true', 0.0)

class ProvenanceTracker:
    """Tracks W3C PROV-compliant provenance"""
    
    def __init__(self):
        self.records: List[ProvenanceRecord] = []
    
    def add_record(
        self,
        entity_id: str,
        activity_id: str,
        agent_id: str,
        derivation: Optional[str] = None,
        attribution: Optional[str] = None
    ) -> ProvenanceRecord:
        """Add a provenance record"""
        record = ProvenanceRecord(
            entity_id=entity_id,
            activity_id=activity_id,
            agent_id=agent_id,
            timestamp=datetime.now(),
            derivation=derivation,
            attribution=attribution
        )
        self.records.append(record)
        return record
    
    def get_chain(self) -> List[ProvenanceRecord]:
        """Get the complete provenance chain"""
        return sorted(self.records, key=lambda r: r.timestamp)

class SourceValidator:
    """Validates and scores sources"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reliability_model = None  # Would be loaded in production
    
    async def validate_source(self, source: Source) -> float:
        """
        Validate a source and return reliability score
        
        Args:
            source: Source to validate
        
        Returns:
            Reliability score (0-1)
        """
        # Placeholder for actual source validation
        # In production, this would check:
        # - Domain authority
        # - Publication reputation
        # - Author credentials
        # - Citation count
        # - Peer review status
        
        base_scores = {
            SourceType.PAPER: 0.8,
            SourceType.BOOK: 0.75,
            SourceType.DATABASE: 0.9,
            SourceType.API: 0.85,
            SourceType.WEBSITE: 0.6,
            SourceType.EXPERT: 0.7
        }
        
        return base_scores.get(source.type, 0.5)

class VerificationSystem:
    """
    Implements W3C PROV-compliant verification with multiple confidence methods
    """
    def __init__(self, config: Dict[str, Any]):
        self.enable_source_validation = config.get('enable_source_validation', True)
        self.confidence_methods = [
            ConfidenceMethod(m) for m in config.get('confidence_methods', ['bayesian'])
        ]
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.bayesian_verifier = BayesianConfidenceScorer()
        self.fuzzy_verifier = FuzzyLogicScorer()
        self.dempster_shafer_verifier = DempsterShaferScorer()
        self.source_validator = SourceValidator(self.device)
        self.provenance_tracker = ProvenanceTracker()
        
        # GPU optimization
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
        
        logger.info(f"Initialized VerificationSystem with methods: {self.confidence_methods}")
    
    async def verify_claim(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> VerificationResult:
        """
        Verify a claim using multiple confidence methods
        
        Args:
            claim: The claim to verify
            sources: List of sources to check against
        
        Returns:
            VerificationResult with confidence scores and provenance
        """
        start_time = asyncio.get_event_loop().time()
        
        # Convert source dictionaries to Source objects
        source_objects = [
            Source(
                url=s.get('url', ''),
                type=SourceType(s.get('type', 'website')),
                title=s.get('title'),
                author=s.get('author')
            ) for s in sources
        ]
        
        # Validate sources if enabled
        if self.enable_source_validation:
            for source in source_objects:
                source.reliability_score = await self.source_validator.validate_source(source)
        
        # Calculate confidence using each method
        method_scores = {}
        evidence = [s.reliability_score for s in source_objects]
        
        if ConfidenceMethod.BAYESIAN in self.confidence_methods:
            method_scores[ConfidenceMethod.BAYESIAN] = self.bayesian_verifier.score(evidence)
        
        if ConfidenceMethod.FUZZY in self.confidence_methods:
            method_scores[ConfidenceMethod.FUZZY] = self.fuzzy_verifier.score(evidence)
        
        if ConfidenceMethod.DEMPSTER_SHAFER in self.confidence_methods:
            # Convert evidence to mass functions
            mass_functions = []
            for e in evidence:
                mass_functions.append({
                    'true': e,
                    'false': 1 - e,
                    'uncertain': 0.1
                })
            method_scores[ConfidenceMethod.DEMPSTER_SHAFER] = \
                self.dempster_shafer_verifier.score(mass_functions)
        
        # Calculate ensemble score if multiple methods
        if len(method_scores) > 1:
            method_scores[ConfidenceMethod.ENSEMBLE] = np.mean(list(method_scores.values()))
        
        # Final confidence is ensemble or single method
        final_confidence = method_scores.get(
            ConfidenceMethod.ENSEMBLE,
            next(iter(method_scores.values())) if method_scores else 0.5
        )
        
        # Track provenance
        self.provenance_tracker.add_record(
            entity_id=f"claim_{hash(claim)}",
            activity_id="verification",
            agent_id="verification_system",
            attribution="; ".join([s.url for s in source_objects])
        )
        
        # Generate explanations
        explanations = {
            "confidence_calculation": f"Used methods: {list(method_scores.keys())}",
            "source_validation": f"Validated {len(source_objects)} sources",
            "reliability_range": f"{min(evidence):.2f} - {max(evidence):.2f}" if evidence else "N/A"
        }
        
        # Calculate metrics
        verification_time = (asyncio.get_event_loop().time() - start_time) * 1000
        gpu_utilization = self._get_gpu_utilization() if self.device == 'cuda' else 0.0
        
        return VerificationResult(
            claim=claim,
            confidence=final_confidence,
            sources_verified=source_objects,
            method_scores=method_scores,
            provenance_chain=self.provenance_tracker.get_chain(),
            verification_time_ms=verification_time,
            gpu_utilization=gpu_utilization,
            explanations=explanations
        )
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except:
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

# Factory function
def create_verification_system(config: Dict[str, Any]) -> VerificationSystem:
    """Create a verification system with configuration"""
    return VerificationSystem(config)