"""
Comprehensive test suite for the Verification System
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
from typing import Dict, List, Any

from aiq.verification.verification_system import (
    VerificationSystem,
    VerificationResult,
    ConfidenceScore,
    ProvenanceTracker,
    SourceAttribution,
    VerificationConfig
)


class TestVerificationSystem:
    """Test suite for VerificationSystem"""
    
    @pytest.fixture
    def verification_system(self):
        """Create verification system instance"""
        config = VerificationConfig(
            confidence_threshold=0.7,
            required_sources=2,
            enable_provenance=True
        )
        return VerificationSystem(config)
    
    @pytest.fixture
    def mock_sources(self):
        """Mock sources for testing"""
        return [
            {
                "url": "http://source1.com",
                "content": "The Earth is round",
                "credibility": 0.9,
                "date": "2024-01-01"
            },
            {
                "url": "http://source2.com", 
                "content": "The Earth is a sphere",
                "credibility": 0.85,
                "date": "2024-01-02"
            },
            {
                "url": "http://source3.com",
                "content": "The Earth is flat",
                "credibility": 0.2,
                "date": "2024-01-03"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_verify_claim_with_consensus(self, verification_system, mock_sources):
        """Test claim verification with source consensus"""
        claim = "The Earth is round"
        result = await verification_system.verify_claim(claim, mock_sources[:2])
        
        assert result.is_verified
        assert result.confidence > 0.8
        assert len(result.supporting_sources) == 2
        assert result.prov_record is not None
    
    @pytest.mark.asyncio
    async def test_verify_claim_with_conflict(self, verification_system, mock_sources):
        """Test claim verification with conflicting sources"""
        claim = "The Earth is flat"
        result = await verification_system.verify_claim(claim, mock_sources)
        
        assert not result.is_verified
        assert result.confidence < 0.5
        assert len(result.conflicting_sources) > 0
    
    @pytest.mark.asyncio
    async def test_bayesian_confidence_scoring(self, verification_system):
        """Test Bayesian confidence scoring method"""
        claim = "AI is transforming healthcare"
        prior = 0.5
        evidence = [
            {"support": True, "strength": 0.8},
            {"support": True, "strength": 0.9},
            {"support": False, "strength": 0.3}
        ]
        
        score = await verification_system._bayesian_confidence(claim, prior, evidence)
        
        assert 0.7 < score < 0.9
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_fuzzy_logic_confidence_scoring(self, verification_system):
        """Test fuzzy logic confidence scoring"""
        claim = "The stock market will rise tomorrow"
        factors = {
            "economic_indicators": 0.7,
            "market_sentiment": 0.6,
            "technical_analysis": 0.8,
            "news_sentiment": 0.5
        }
        
        score = await verification_system._fuzzy_logic_confidence(claim, factors)
        
        assert 0.5 < score < 0.8
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_dempster_shafer_confidence(self, verification_system):
        """Test Dempster-Shafer confidence method"""
        claim = "Product X is safe"
        evidence_masses = [
            {"support": 0.7, "against": 0.2, "uncertain": 0.1},
            {"support": 0.8, "against": 0.1, "uncertain": 0.1},
            {"support": 0.6, "against": 0.3, "uncertain": 0.1}
        ]
        
        score = await verification_system._dempster_shafer_confidence(claim, evidence_masses)
        
        assert 0.6 < score < 0.9
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_provenance_tracking(self, verification_system, mock_sources):
        """Test W3C PROV compliant provenance tracking"""
        claim = "Climate change is real"
        result = await verification_system.verify_claim(claim, mock_sources[:2])
        
        prov_record = result.prov_record
        assert prov_record is not None
        assert "entity" in prov_record
        assert "activity" in prov_record
        assert "agent" in prov_record
        assert "wasGeneratedBy" in prov_record
    
    @pytest.mark.asyncio
    async def test_source_attribution(self, verification_system, mock_sources):
        """Test source attribution tracking"""
        claim = "Python is a programming language"
        result = await verification_system.verify_claim(claim, mock_sources[:1])
        
        assert len(result.source_attributions) > 0
        attribution = result.source_attributions[0]
        assert attribution.source_url == mock_sources[0]["url"]
        assert attribution.confidence > 0
        assert attribution.extraction_method in ["exact_match", "semantic_similarity"]
    
    @pytest.mark.asyncio
    async def test_multi_threaded_verification(self, verification_system, mock_sources):
        """Test multi-threaded verification processing"""
        claims = [
            "The sun is a star",
            "Water boils at 100°C",
            "Python is interpreted",
            "AI needs data"
        ]
        
        # Verify multiple claims concurrently
        tasks = [
            verification_system.verify_claim(claim, mock_sources[:2])
            for claim in claims
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(claims)
        assert all(isinstance(r, VerificationResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_verification_with_timeout(self, verification_system, mock_sources):
        """Test verification with timeout"""
        claim = "Complex scientific claim requiring extensive processing"
        
        with patch.object(verification_system, '_analyze_sources', 
                         new_callable=AsyncMock) as mock_analyze:
            # Simulate slow processing
            mock_analyze.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(asyncio.TimeoutError):
                await verification_system.verify_claim(
                    claim, 
                    mock_sources,
                    timeout=0.1
                )
    
    @pytest.mark.asyncio
    async def test_confidence_aggregation(self, verification_system):
        """Test aggregation of multiple confidence scores"""
        scores = {
            "bayesian": 0.85,
            "fuzzy_logic": 0.75,
            "dempster_shafer": 0.80
        }
        
        aggregated = await verification_system._aggregate_confidence_scores(scores)
        
        assert 0.75 < aggregated < 0.85
        assert isinstance(aggregated, float)
    
    @pytest.mark.asyncio
    async def test_source_credibility_assessment(self, verification_system, mock_sources):
        """Test source credibility assessment"""
        credibility = await verification_system._assess_source_credibility(
            mock_sources[0]
        )
        
        assert credibility == mock_sources[0]["credibility"]
        
        # Test with missing credibility
        source_no_cred = {"url": "http://test.com", "content": "Test"}
        default_cred = await verification_system._assess_source_credibility(
            source_no_cred
        )
        
        assert 0 < default_cred < 1
    
    @pytest.mark.asyncio
    async def test_claim_extraction(self, verification_system):
        """Test claim extraction from complex text"""
        text = """
        According to recent studies, artificial intelligence is revolutionizing healthcare.
        The studies show that AI can diagnose diseases with 95% accuracy.
        However, some experts argue that human oversight is still necessary.
        """
        
        claims = await verification_system._extract_claims(text)
        
        assert len(claims) >= 2
        assert any("artificial intelligence" in claim for claim in claims)
        assert any("95% accuracy" in claim for claim in claims)
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_matching(self, verification_system):
        """Test semantic similarity for source matching"""
        claim = "Global warming is causing ice caps to melt"
        source_content = "Climate change leads to polar ice reduction"
        
        similarity = await verification_system._compute_semantic_similarity(
            claim, source_content
        )
        
        assert similarity > 0.7
        assert isinstance(similarity, float)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, verification_system):
        """Test error handling in verification system"""
        # Test with empty sources
        result = await verification_system.verify_claim("Test claim", [])
        assert not result.is_verified
        assert result.error_message is not None
        
        # Test with invalid claim
        result = await verification_system.verify_claim("", [{"content": "test"}])
        assert not result.is_verified
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, verification_system, mock_sources):
        """Test caching of verification results"""
        claim = "The moon orbits Earth"
        
        # First call
        result1 = await verification_system.verify_claim(claim, mock_sources[:2])
        
        # Second call (should use cache)
        with patch.object(verification_system, '_analyze_sources') as mock_analyze:
            result2 = await verification_system.verify_claim(claim, mock_sources[:2])
            mock_analyze.assert_not_called()
        
        assert result1.confidence == result2.confidence
        assert result1.is_verified == result2.is_verified
    
    @pytest.mark.asyncio
    async def test_batch_verification(self, verification_system, mock_sources):
        """Test batch verification of multiple claims"""
        claims = [
            "Python is a programming language",
            "The Earth orbits the Sun",
            "Water is H2O"
        ]
        
        results = await verification_system.batch_verify(claims, mock_sources)
        
        assert len(results) == len(claims)
        assert all(isinstance(r, VerificationResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_verification_with_metadata(self, verification_system, mock_sources):
        """Test verification with metadata tracking"""
        claim = "AI improves efficiency"
        metadata = {
            "domain": "technology",
            "importance": "high",
            "user_id": "test_user"
        }
        
        result = await verification_system.verify_claim(
            claim, 
            mock_sources[:2],
            metadata=metadata
        )
        
        assert result.metadata == metadata
        assert result.prov_record["metadata"] == metadata


class TestProvenanceTracker:
    """Test suite for ProvenanceTracker"""
    
    @pytest.fixture
    def provenance_tracker(self):
        """Create provenance tracker instance"""
        return ProvenanceTracker()
    
    def test_create_entity(self, provenance_tracker):
        """Test PROV entity creation"""
        entity = provenance_tracker.create_entity(
            "claim_123",
            {"type": "claim", "content": "Test claim"}
        )
        
        assert entity["id"] == "claim_123"
        assert entity["type"] == "entity"
        assert entity["attributes"]["type"] == "claim"
    
    def test_create_activity(self, provenance_tracker):
        """Test PROV activity creation"""
        activity = provenance_tracker.create_activity(
            "verification_456",
            {"method": "bayesian", "duration": 1.5}
        )
        
        assert activity["id"] == "verification_456"
        assert activity["type"] == "activity"
        assert activity["attributes"]["method"] == "bayesian"
    
    def test_create_agent(self, provenance_tracker):
        """Test PROV agent creation"""
        agent = provenance_tracker.create_agent(
            "system_789",
            {"name": "VerificationSystem", "version": "1.0"}
        )
        
        assert agent["id"] == "system_789"
        assert agent["type"] == "agent"
        assert agent["attributes"]["version"] == "1.0"
    
    def test_create_generation_relationship(self, provenance_tracker):
        """Test PROV generation relationship"""
        entity = provenance_tracker.create_entity("result_1", {})
        activity = provenance_tracker.create_activity("process_1", {})
        
        generation = provenance_tracker.create_generation(
            entity["id"],
            activity["id"],
            datetime.now()
        )
        
        assert generation["entity"] == entity["id"]
        assert generation["activity"] == activity["id"]
        assert "time" in generation
    
    def test_serialize_prov_record(self, provenance_tracker):
        """Test PROV record serialization"""
        entity = provenance_tracker.create_entity("e1", {"key": "value"})
        activity = provenance_tracker.create_activity("a1", {"step": 1})
        
        provenance_tracker.add_entity(entity)
        provenance_tracker.add_activity(activity)
        
        serialized = provenance_tracker.serialize()
        
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert "entities" in parsed
        assert "activities" in parsed


class TestSourceAttribution:
    """Test suite for SourceAttribution"""
    
    def test_source_attribution_creation(self):
        """Test source attribution object creation"""
        attribution = SourceAttribution(
            source_url="http://example.com",
            source_title="Example Source",
            author="John Doe",
            publication_date=datetime.now(),
            confidence=0.9,
            extraction_method="exact_match",
            quote="This is a test quote"
        )
        
        assert attribution.source_url == "http://example.com"
        assert attribution.confidence == 0.9
        assert attribution.extraction_method == "exact_match"
    
    def test_source_attribution_validation(self):
        """Test source attribution validation"""
        # Valid attribution
        attribution = SourceAttribution(
            source_url="http://valid.com",
            confidence=0.8,
            extraction_method="semantic_similarity"
        )
        assert attribution.is_valid()
        
        # Invalid attribution (no URL)
        with pytest.raises(ValueError):
            SourceAttribution(
                source_url="",
                confidence=0.8,
                extraction_method="exact_match"
            )
        
        # Invalid confidence
        with pytest.raises(ValueError):
            SourceAttribution(
                source_url="http://test.com",
                confidence=1.5,  # > 1.0
                extraction_method="exact_match"
            )


class TestVerificationConfig:
    """Test suite for VerificationConfig"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = VerificationConfig()
        
        assert config.confidence_threshold == 0.7
        assert config.required_sources == 2
        assert config.enable_provenance == True
        assert config.timeout == 30.0
        assert config.cache_ttl == 3600
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = VerificationConfig(
            confidence_threshold=0.8,
            required_sources=3,
            enable_caching=True
        )
        assert config.is_valid()
        
        # Invalid threshold
        with pytest.raises(ValueError):
            VerificationConfig(confidence_threshold=1.5)
        
        # Invalid source count
        with pytest.raises(ValueError):
            VerificationConfig(required_sources=0)
    
    def test_config_serialization(self):
        """Test configuration serialization"""
        config = VerificationConfig(
            confidence_threshold=0.85,
            required_sources=5,
            verification_methods=["bayesian", "fuzzy_logic"]
        )
        
        serialized = config.to_dict()
        assert serialized["confidence_threshold"] == 0.85
        assert serialized["required_sources"] == 5
        
        # Test deserialization
        restored = VerificationConfig.from_dict(serialized)
        assert restored.confidence_threshold == config.confidence_threshold
        assert restored.required_sources == config.required_sources