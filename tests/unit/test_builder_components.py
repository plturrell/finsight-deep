"""Tests for AIQ builder components."""

import pytest
from unittest.mock import Mock, patch
import uuid

from aiq.builder.builder import Builder
from aiq.builder.context import AIQContext, AIQContextState
from aiq.builder.workflow import Workflow
from aiq.builder.function import Function


class TestBuilder:
    """Test the Builder class."""
    
    def test_builder_creation(self):
        """Test Builder instance creation."""
        builder = Builder()
        assert builder is not None
        assert hasattr(builder, 'config')
    
    def test_builder_build_method(self):
        """Test Builder build method."""
        builder = Builder()
        
        # Mock config
        mock_config = Mock()
        mock_config.type = "workflow"
        
        with patch.object(builder, '_build_workflow') as mock_build:
            mock_build.return_value = Mock()
            result = builder.build(mock_config)
            mock_build.assert_called_once_with(mock_config)


class TestAIQContext:
    """Test the AIQContext class."""
    
    def test_context_creation(self):
        """Test AIQContext creation."""
        context_state = AIQContextState()
        context = AIQContext(context_state)
        
        assert context is not None
        assert context._context_state == context_state
    
    def test_context_properties(self):
        """Test AIQContext properties."""
        context = AIQContext.get()
        
        # Test basic properties exist
        assert hasattr(context, 'input_message')
        assert hasattr(context, 'user_manager')
        assert hasattr(context, 'metadata')
        assert hasattr(context, 'active_function')
    
    def test_push_active_function(self):
        """Test push_active_function context manager."""
        context = AIQContext.get()
        
        with context.push_active_function("test_function", {"input": "data"}) as manager:
            assert manager is not None
            # Function should be active within context
            active = context.active_function
            assert active.function_name == "test_function"
    
    def test_context_singleton(self):
        """Test that AIQContextState is a singleton."""
        state1 = AIQContextState.get()
        state2 = AIQContextState.get()
        assert state1 is state2


class TestWorkflow:
    """Test the Workflow class."""
    
    @pytest.fixture
    def sample_workflow_config(self):
        """Create a sample workflow configuration."""
        return {
            "name": "test_workflow",
            "type": "workflow",
            "version": "1.0",
            "functions": []
        }
    
    def test_workflow_creation(self, sample_workflow_config):
        """Test Workflow creation."""
        workflow = Workflow(sample_workflow_config)
        assert workflow is not None
        assert workflow.config == sample_workflow_config
    
    def test_workflow_string_representation(self, sample_workflow_config):
        """Test Workflow string representation."""
        workflow = Workflow(sample_workflow_config)
        str_repr = str(workflow)
        
        assert "test_workflow" in str_repr
        assert isinstance(str_repr, str)


class TestFunction:
    """Test the Function class."""
    
    @pytest.fixture
    def sample_function_config(self):
        """Create a sample function configuration."""
        return {
            "name": "test_function",
            "type": "function",
            "description": "A test function"
        }
    
    def test_function_creation(self, sample_function_config):
        """Test Function creation."""
        function = Function(sample_function_config)
        assert function is not None
        assert function.config == sample_function_config
    
    def test_function_call(self, sample_function_config):
        """Test Function __call__ method."""
        function = Function(sample_function_config)
        
        # Mock the internal call
        with patch.object(function, '_call') as mock_call:
            mock_call.return_value = {"result": "success"}
            result = function(input_data="test")
            
            mock_call.assert_called_once_with(input_data="test")
            assert result == {"result": "success"}
    
    def test_function_name_property(self, sample_function_config):
        """Test Function name property."""
        function = Function(sample_function_config)
        assert function.name == "test_function"


class TestBuilderIntegration:
    """Integration tests for builder components."""
    
    def test_builder_workflow_integration(self):
        """Test Builder creating a Workflow."""
        builder = Builder()
        
        config = Mock()
        config.type = "workflow"
        config.name = "integration_test"
        
        with patch.object(builder, '_build_workflow') as mock_build:
            workflow = Mock(spec=Workflow)
            mock_build.return_value = workflow
            
            result = builder.build(config)
            assert result == workflow
    
    def test_context_with_function(self):
        """Test context management with functions."""
        context = AIQContext.get()
        
        # Simulate function execution
        with context.push_active_function("integration_function", {"test": "data"}) as mgr:
            # Set output
            mgr.set_output({"result": "success"})
            
            # Verify function is active
            active = context.active_function
            assert active.function_name == "integration_function"
        
        # After context, function should no longer be active
        active_after = context.active_function
        assert active_after.function_name != "integration_function"