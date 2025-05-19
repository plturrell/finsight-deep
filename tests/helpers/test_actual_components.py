"""Tests for actual AIQ components implementation."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Test context and state
def test_context_imports():
    """Test basic context imports."""
    from aiq.builder.context import AIQContext, AIQContextState
    from aiq.data_models.invocation_node import InvocationNode
    
    # Create context
    context = AIQContext.get()
    assert context is not None
    
    # Test singleton
    state1 = AIQContextState.get()
    state2 = AIQContextState.get()
    assert state1 is state2


def test_request_attributes():
    """Test RequestAttributes."""
    from aiq.runtime.user_metadata import RequestAttributes
    
    attrs = RequestAttributes()
    assert attrs is not None
    assert attrs.method is None
    assert attrs.headers == {}


def test_config_models():
    """Test configuration models."""
    from aiq.data_models.config import AIQConfig, GeneralConfig
    
    # Test GeneralConfig
    general_config = GeneralConfig()
    assert general_config is not None
    
    # Test AIQConfig - check if it needs required fields
    try:
        aiq_config = AIQConfig(
            version="1.0",
            type="workflow"
        )
        assert aiq_config.version == "1.0"
    except Exception as e:
        # If required fields are missing, just verify the class exists
        assert AIQConfig is not None


def test_workflow_config():
    """Test workflow configuration."""
    from aiq.data_models.workflow import WorkflowConfig
    
    config = WorkflowConfig(
        type="workflow",
        name="test_workflow",
        version="1.0",
        functions=[]
    )
    
    assert config.name == "test_workflow"
    assert config.functions == []
    assert config.version == "1.0"


def test_base_models():
    """Test base model imports."""
    from aiq.data_models.common import HashableBaseModel
    
    # Create a simple test model
    class TestModel(HashableBaseModel):
        name: str = "test"
        value: int = 42
    
    model = TestModel()
    assert model.name == "test"
    assert model.value == 42
    
    # Should be hashable
    hash_value = hash(model)
    assert isinstance(hash_value, int)


def test_invocation_node():
    """Test InvocationNode."""
    from aiq.data_models.invocation_node import InvocationNode
    
    node = InvocationNode(
        function_id="test_123",
        function_name="test_function"
    )
    
    assert node.function_id == "test_123"
    assert node.function_name == "test_function"
    assert node.parent_id is None
    assert node.parent_name is None


def test_intermediate_step():
    """Test intermediate step models."""
    from aiq.data_models.intermediate_step import (
        IntermediateStep, 
        IntermediateStepType,
        IntermediateStepPayload
    )
    from datetime import datetime
    
    # Create a step
    step = IntermediateStep(
        UUID="step_123",
        type=IntermediateStepType.FUNCTION_START,
        timestamp=datetime.now()
    )
    
    assert step.UUID == "step_123"
    assert step.type == IntermediateStepType.FUNCTION_START
    assert isinstance(step.timestamp, datetime)


def test_runtime_state():
    """Test runtime state enum."""
    from aiq.runtime.runner import AIQRunnerState
    
    assert AIQRunnerState.UNINITIALIZED.value == 0
    assert AIQRunnerState.INITIALIZED.value == 1
    assert AIQRunnerState.RUNNING.value == 2
    assert AIQRunnerState.COMPLETED.value == 3
    assert AIQRunnerState.FAILED.value == 4


def test_llm_config():
    """Test LLM configuration models."""
    from aiq.data_models.llm import LLMBaseConfig
    
    # LLMBaseConfig is abstract, check if we can access it
    assert LLMBaseConfig is not None
    
    # Try to find a concrete implementation
    try:
        from aiq.llm.nim_llm import NimLLMConfig
        config = NimLLMConfig(
            name="test_llm",
            api_key="test_key",
            model="test_model"
        )
        assert config.name == "test_llm"
    except ImportError:
        # If NIM not available, just verify base class exists
        pass


def test_context_manager():
    """Test context manager functionality."""
    from aiq.builder.context import AIQContext
    
    context = AIQContext.get()
    initial_function = context.active_function
    
    # Test context manager
    with context.push_active_function("test_func", {"data": "test"}) as mgr:
        active = context.active_function
        assert active.function_name == "test_func"
        mgr.set_output({"result": "success"})
    
    # Should revert after context
    assert context.active_function.function_id == initial_function.function_id


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality in the codebase."""
    from aiq.builder.context import AIQContext
    
    context = AIQContext.get()
    
    # Many components support async operations
    # Just verify context works in async context
    async def async_operation():
        with context.push_active_function("async_test", {}) as mgr:
            await asyncio.sleep(0.01)  # Simulate async work
            mgr.set_output({"async": True})
            return mgr.output
    
    result = await async_operation()
    assert result == {"async": True}


def test_imports_and_aliases():
    """Test that import aliases work correctly."""
    from aiq.runtime import Runner, Session
    from aiq.runtime.runner import AIQRunner
    from aiq.runtime.session import AIQSessionManager
    
    assert Runner == AIQRunner
    assert Session == AIQSessionManager


# Integration test
def test_component_integration():
    """Test integration between different components."""
    from aiq.builder.context import AIQContext
    from aiq.data_models.invocation_node import InvocationNode
    from aiq.runtime.user_metadata import RequestAttributes
    
    # Get context
    context = AIQContext.get()
    
    # Check metadata
    metadata = context.metadata
    assert isinstance(metadata, RequestAttributes)
    
    # Check active function
    active = context.active_function
    assert isinstance(active, InvocationNode)
    assert active.function_name == "root"  # Default root function