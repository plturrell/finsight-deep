"""
Integration tests for Production Digital Human System
Tests all real services and integrations
"""

import os
import asyncio
import pytest
import numpy as np
import jwt
from datetime import datetime, timedelta

from aiq.digital_human.deployment.production_implementation import (
    ProductionDigitalHuman,
    ProductionTokkioOrchestrator
)


class TestProductionDigitalHuman:
    """Test suite for production digital human system"""
    
    @pytest.fixture
    async def production_system(self):
        """Create production system instance"""
        # Use test configuration
        config_path = os.path.join(
            os.path.dirname(__file__),
            'test_production_config.yaml'
        )
        
        system = ProductionDigitalHuman(config_path)
        await system.initialize_orchestrator()
        
        yield system
        
        await system.close()
    
    @pytest.fixture
    def auth_token(self):
        """Generate test authentication token"""
        secret = os.environ.get('JWT_SECRET_KEY')
        payload = {
            'user_id': 'test_user',
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, secret, algorithm='HS256')
    
    @pytest.mark.asyncio
    async def test_nvidia_ace_integration(self, production_system):
        """Test NVIDIA ACE platform integration"""
        # Test Audio2Face-2D
        audio_data = np.random.randn(16000).astype(np.float32)  # 1 second
        
        result = await production_system.orchestrator.ace_platform.render_avatar(
            audio_data=audio_data,
            emotion="confident",
            intensity=0.8
        )
        
        assert result is not None
        assert 'frames' in result
        assert result['frames'] is not None
        assert result['emotion'] == "confident"
    
    @pytest.mark.asyncio
    async def test_speech_recognition(self, production_system):
        """Test Parakeet-CTC-1.1B ASR"""
        # Generate test audio (you would use real audio in production)
        audio_data = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
        
        text = await production_system.orchestrator.ace_platform.speech_to_text(
            audio_data
        )
        
        assert isinstance(text, str)
        assert len(text) > 0  # Should transcribe something
    
    @pytest.mark.asyncio
    async def test_neural_supercomputer_connection(self, production_system):
        """Test connection to neural supercomputer"""
        connector = production_system.orchestrator.neural_connector
        
        # Test health check
        response = await connector.session.get(
            f"{connector.endpoint}/health"
        )
        
        assert response.status == 200
    
    @pytest.mark.asyncio
    async def test_financial_data_retrieval(self, production_system):
        """Test real financial data providers"""
        providers = production_system.financial_providers
        
        # Test Yahoo Finance
        if 'yahoo' in providers:
            ticker = providers['yahoo'].Ticker("AAPL")
            info = ticker.info
            
            assert info is not None
            assert 'currentPrice' in info
            assert info['currentPrice'] > 0
        
        # Test Alpha Vantage
        if 'alpha_vantage' in providers:
            data, meta_data = providers['alpha_vantage'].get_daily("AAPL")
            
            assert data is not None
            assert len(data) > 0
    
    @pytest.mark.asyncio
    async def test_model_context_server(self, production_system):
        """Test Model Context Server with RAG"""
        context_server = production_system.orchestrator.context_server
        
        results = await context_server.retrieve_context(
            query="What is Apple's current stock price?",
            sources=["financial", "web"],
            context_type="financial"
        )
        
        assert results is not None
        assert 'sources' in results
        assert len(results['sources']) > 0
        assert any('AAPL' in str(source) for source in results['sources'])
    
    @pytest.mark.asyncio
    async def test_google_search_integration(self, production_system):
        """Test Google Custom Search API"""
        google_service = production_system.google_service
        
        results = google_service.cse().list(
            q="NVIDIA stock price",
            cx=os.environ.get('GOOGLE_CSE_ID'),
            num=5
        ).execute()
        
        assert 'items' in results
        assert len(results['items']) > 0
    
    @pytest.mark.asyncio
    async def test_database_connections(self, production_system):
        """Test all database connections"""
        # Test PostgreSQL
        with production_system.pg_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        # Test Redis
        production_system.redis_client.set("test_key", "test_value")
        value = production_system.redis_client.get("test_key")
        assert value.decode() == "test_value"
        
        # Test Milvus
        from pymilvus import utility
        assert utility.has_collection("digital_human_production")
    
    @pytest.mark.asyncio
    async def test_full_interaction_pipeline(self, production_system, auth_token):
        """Test complete interaction pipeline"""
        # Start session
        session_id = await production_system.start_session(
            user_id="test_user",
            auth_token=auth_token
        )
        
        assert session_id is not None
        
        # Process interaction
        response = await production_system.process_interaction(
            session_id=session_id,
            user_input="What's your analysis of the current market conditions?",
            audio_data=None
        )
        
        assert response is not None
        assert 'text' in response
        assert 'reasoning' in response
        assert 'context' in response
        assert 'avatar' in response
        assert response['reasoning']['confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_tokkio_workflow(self, production_system):
        """Test NVIDIA Tokkio workflow orchestration"""
        tokkio_client = production_system.tokkio_client
        
        # Test workflow creation
        workflow = tokkio_client.create_workflow(
            workflow_id="test_financial_advisor"
        )
        
        assert workflow is not None
        
        # Test workflow execution
        result = await workflow.execute({
            "input": "Test financial query",
            "context": {"test": True}
        })
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_emotional_response_mapping(self, production_system):
        """Test emotional response generation"""
        orchestrator = production_system.orchestrator
        
        # Test with positive market news
        response_positive = await orchestrator.process_interaction(
            session_id="test_session",
            user_input="The market is up 5% today!",
            audio_data=None
        )
        
        assert response_positive['emotion'] in ['confident', 'happy', 'optimistic']
        
        # Test with negative market news
        response_negative = await orchestrator.process_interaction(
            session_id="test_session",
            user_input="The market crashed 10% today.",
            audio_data=None
        )
        
        assert response_negative['emotion'] in ['concerned', 'empathetic', 'thoughtful']
    
    @pytest.mark.asyncio
    async def test_security_features(self, production_system):
        """Test security implementations"""
        # Test JWT validation
        invalid_token = "invalid.jwt.token"
        
        with pytest.raises(ValueError, match="Authentication failed"):
            await production_system.start_session(
                user_id="test_user",
                auth_token=invalid_token
            )
        
        # Test API key encryption
        test_key = "test_api_key"
        encrypted = production_system.fernet.encrypt(test_key.encode())
        decrypted = production_system.fernet.decrypt(encrypted).decode()
        
        assert decrypted == test_key
    
    @pytest.mark.asyncio
    async def test_monitoring_metrics(self, production_system):
        """Test monitoring and metrics collection"""
        # Process some interactions to generate metrics
        session_id = await production_system.start_session(
            user_id="test_user",
            auth_token=self.auth_token()
        )
        
        await production_system.process_interaction(
            session_id=session_id,
            user_input="Test query",
            audio_data=None
        )
        
        # Check metrics
        from prometheus_client import REGISTRY
        
        request_count = REGISTRY.get_sample_value('digital_human_requests_total')
        assert request_count > 0
        
        active_sessions = REGISTRY.get_sample_value('digital_human_active_sessions')
        assert active_sessions >= 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, production_system):
        """Test error handling and recovery"""
        # Test with invalid session
        with pytest.raises(ValueError, match="Invalid or inactive session"):
            await production_system.process_interaction(
                session_id="invalid_session",
                user_input="Test",
                audio_data=None
            )
        
        # Test with network error simulation
        # This would test retry logic and circuit breakers
        
    @pytest.mark.asyncio
    async def test_performance_requirements(self, production_system):
        """Test performance meets requirements"""
        import time
        
        session_id = await production_system.start_session(
            user_id="test_user",
            auth_token=self.auth_token()
        )
        
        # Measure response time
        start_time = time.time()
        
        response = await production_system.process_interaction(
            session_id=session_id,
            user_input="Quick test query",
            audio_data=None
        )
        
        response_time = time.time() - start_time
        
        # Should respond within 3 seconds for text-only queries
        assert response_time < 3.0
        assert response['processing_time'] < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])