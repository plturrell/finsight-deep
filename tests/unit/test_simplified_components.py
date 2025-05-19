"""Simplified tests for AIQ components that match actual implementation."""

import pytest
from unittest.mock import Mock, patch
import asyncio

from aiq.builder.context import AIQContext, AIQContextState
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.runtime.user_metadata import RequestAttributes
from aiq.data_models.config import Config


class TestAIQContext:
    """Test the AIQContext class."""
    
    def test_context_singleton(self):
        """Test that AIQContextState is a singleton."""
        state1 = AIQContextState.get()
        state2 = AIQContextState.get()
        assert state1 is state2
    
    def test_context_creation(self):
        """Test AIQContext creation."""
        context = AIQContext.get()
        assert context is not None
        assert isinstance(context._context_state, AIQContextState)
    
    def test_context_properties(self):
        """Test AIQContext basic properties."""
        context = AIQContext.get()
        
        # Test properties exist
        assert hasattr(context, 'input_message')
        assert hasattr(context, 'metadata')
        assert hasattr(context, 'active_function')
        assert hasattr(context, 'user_interaction_manager')
        assert hasattr(context, 'intermediate_step_manager')
    
    def test_push_active_function(self):
        """Test push_active_function context manager."""
        context = AIQContext.get()
        
        # Save initial state
        initial_function = context.active_function
        
        with context.push_active_function("test_function", {"input": "data"}) as manager:
            assert manager is not None
            active = context.active_function
            assert active.function_name == "test_function"
        
        # After context, should revert
        final_function = context.active_function
        assert final_function.function_id == initial_function.function_id


class TestRequestAttributes:
    """Test the RequestAttributes class."""
    
    def test_creation(self):
        """Test RequestAttributes creation."""
        attrs = RequestAttributes()
        assert attrs is not None
        assert hasattr(attrs, '_request')
    
    def test_default_properties(self):
        """Test default properties."""
        attrs = RequestAttributes()
        
        # Default values should be None
        assert attrs.method is None
        assert attrs.headers == {}
        assert attrs.query_params == {}
        assert attrs.path_params == {}
        assert attrs.body is None


class TestWorkflowBuilder:
    """Test WorkflowBuilder class."""
    
    def test_workflow_builder_creation(self):
        """Test WorkflowBuilder creation."""
        builder = WorkflowBuilder()
        assert builder is not None
        assert hasattr(builder, '_function_mapping')
        assert hasattr(builder, '_embedder_mapping')
        assert hasattr(builder, '_llm_mapping')
    
    @pytest.mark.asyncio
    async def test_build_workflow_method(self):
        """Test WorkflowBuilder build method."""
        builder = WorkflowBuilder()
        
        # Create a minimal config
        config = Config(
            type="workflow",
            name="test_workflow",
            functions=[]
        )
        
        with patch.object(builder, '_build_workflow') as mock_build:
            mock_build.return_value = Mock()
            
            # The actual method might be async
            if asyncio.iscoroutinefunction(builder.build):
                result = await builder.build(config)
            else:
                result = builder.build(config)
            
            mock_build.assert_called_once()


class TestDataModels:
    """Test data models can be imported and created."""
    
    def test_config_creation(self):
        """Test Config model creation."""
        from aiq.data_models.config import Config
        
        config = Config(
            type="workflow",
            name="test_config"
        )
        
        assert config.type == "workflow"
        assert config.name == "test_config"
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig creation."""
        from aiq.data_models.workflow import WorkflowConfig
        
        workflow = WorkflowConfig(
            type="workflow",
            name="test_workflow",
            functions=[]
        )
        
        assert workflow.name == "test_workflow"
        assert workflow.functions == []
    
    def test_invocation_node_creation(self):
        """Test InvocationNode creation."""
        from aiq.data_models.invocation_node import InvocationNode
        
        node = InvocationNode(
            function_id="test_id",
            function_name="test_function"
        )
        
        assert node.function_id == "test_id"
        assert node.function_name == "test_function"


class TestIntegration:
    """Integration tests between components."""
    
    def test_context_and_metadata(self):
        """Test context with request metadata."""
        context = AIQContext.get()
        metadata = context.metadata
        
        assert isinstance(metadata, RequestAttributes)
        
        # Should be able to access request properties
        assert metadata.method is None  # Default value
    
    @pytest.mark.asyncio
    async def test_workflow_builder_with_config(self):
        """Test WorkflowBuilder with actual config."""
        from aiq.data_models.config import Config
        
        builder = WorkflowBuilder()
        config = Config(
            type="workflow",
            name="integration_test"
        )
        
        # Mock the internal build process
        with patch.object(builder, '_build_workflow') as mock_build:
            mock_workflow = Mock()
            mock_workflow.name = "integration_test"
            mock_build.return_value = mock_workflow
            
            if asyncio.iscoroutinefunction(builder.build):
                result = await builder.build(config)
            else:
                result = builder.build(config)
            
            assert result == mock_workflow


# Additional helper tests for imports
def test_core_imports():
    """Test that core imports work."""
    import aiq
    from aiq.builder import Builder
    from aiq.runtime import Runner, Session
    
    # These should not raise ImportError
    assert True


def test_cli_imports():
    """Test CLI imports."""
    from aiq.cli.entrypoint import aiq
    from aiq.cli.main import main
    
    assert aiq is not None
    assert main is not None