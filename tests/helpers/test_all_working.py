"""Working tests for all AIQ components."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime
import os


class TestContextAndState:
    """Test context and state management."""
    
    def test_context_singleton(self):
        """Test AIQContextState singleton pattern."""
        from aiq.builder.context import AIQContextState
        
        state1 = AIQContextState.get()
        state2 = AIQContextState.get()
        assert state1 is state2
    
    def test_context_creation(self):
        """Test AIQContext creation and properties."""
        from aiq.builder.context import AIQContext
        
        context = AIQContext.get()
        assert context is not None
        assert hasattr(context, 'input_message')
        assert hasattr(context, 'metadata')
        assert hasattr(context, 'active_function')
    
    def test_context_manager(self):
        """Test context manager functionality."""
        from aiq.builder.context import AIQContext
        
        context = AIQContext.get()
        initial = context.active_function
        
        with context.push_active_function("test", {"data": 1}) as mgr:
            assert context.active_function.function_name == "test"
            mgr.set_output({"result": "ok"})
            assert mgr.output == {"result": "ok"}
        
        assert context.active_function.function_id == initial.function_id


class TestDataModels:
    """Test data models."""
    
    def test_invocation_node(self):
        """Test InvocationNode model."""
        from aiq.data_models.invocation_node import InvocationNode
        
        node = InvocationNode(
            function_id="test_id",
            function_name="test_func"
        )
        assert node.function_id == "test_id"
        assert node.function_name == "test_func"
    
    def test_config_models(self):
        """Test configuration models."""
        from aiq.data_models.config import AIQConfig, GeneralConfig
        
        # GeneralConfig should work with defaults
        general = GeneralConfig()
        assert general is not None
        
        # AIQConfig exists
        assert AIQConfig is not None
    
    def test_base_models(self):
        """Test base model functionality."""
        from aiq.data_models.common import HashableBaseModel
        
        class TestModel(HashableBaseModel):
            name: str = "test"
            value: int = 100
        
        model = TestModel()
        assert model.name == "test"
        assert model.value == 100
        
        # Should be hashable
        hash1 = hash(model)
        hash2 = hash(model)
        assert hash1 == hash2
    
    def test_intermediate_step_payload(self):
        """Test IntermediateStepPayload model."""
        from aiq.data_models.intermediate_step import (
            IntermediateStepPayload,
            IntermediateStepType
        )
        
        payload = IntermediateStepPayload(
            UUID="test_uuid",
            event_type=IntermediateStepType.FUNCTION_START,
            name="test_step",
            data={"test": "data"}
        )
        
        assert payload.UUID == "test_uuid"
        assert payload.event_type == IntermediateStepType.FUNCTION_START
        assert payload.name == "test_step"


class TestRuntime:
    """Test runtime components."""
    
    def test_runner_state(self):
        """Test AIQRunnerState enum."""
        from aiq.runtime.runner import AIQRunnerState
        
        assert AIQRunnerState.UNINITIALIZED.value == 0
        assert AIQRunnerState.INITIALIZED.value == 1
        assert AIQRunnerState.RUNNING.value == 2
        assert AIQRunnerState.COMPLETED.value == 3
        assert AIQRunnerState.FAILED.value == 4
    
    def test_request_attributes(self):
        """Test RequestAttributes model."""
        from aiq.runtime.user_metadata import RequestAttributes
        
        attrs = RequestAttributes()
        assert attrs is not None
        assert attrs._request is not None
        
        # Default values
        assert attrs.method is None
        assert attrs.body is None
    
    def test_import_aliases(self):
        """Test runtime import aliases."""
        from aiq.runtime import Runner, Session
        from aiq.runtime.runner import AIQRunner
        from aiq.runtime.session import AIQSessionManager
        
        assert Runner == AIQRunner
        assert Session == AIQSessionManager


class TestCLI:
    """Test CLI components."""
    
    def test_cli_imports(self):
        """Test CLI imports."""
        from aiq.cli.entrypoint import aiq
        from aiq.cli.main import main
        
        assert aiq is not None
        assert main is not None
    
    def test_cli_utils(self):
        """Test CLI utilities."""
        from aiq.cli.cli_utils.validation import validate_yaml
        
        # Function should exist
        assert validate_yaml is not None


class TestAsyncSupport:
    """Test async functionality."""
    
    @pytest.mark.asyncio
    async def test_async_context(self):
        """Test context in async environment."""
        from aiq.builder.context import AIQContext
        
        context = AIQContext.get()
        
        async def async_function():
            with context.push_active_function("async_func", {}) as mgr:
                await asyncio.sleep(0.001)
                mgr.set_output({"async": True})
                return mgr.output
        
        result = await async_function()
        assert result == {"async": True}
    
    @pytest.mark.asyncio
    async def test_multiple_async_contexts(self):
        """Test multiple async contexts."""
        from aiq.builder.context import AIQContext
        
        context = AIQContext.get()
        
        async def task1():
            with context.push_active_function("task1", {}) as mgr:
                await asyncio.sleep(0.001)
                mgr.set_output({"task": 1})
                return mgr.output
        
        async def task2():
            with context.push_active_function("task2", {}) as mgr:
                await asyncio.sleep(0.001)
                mgr.set_output({"task": 2})
                return mgr.output
        
        results = await asyncio.gather(task1(), task2())
        assert results[0] == {"task": 1}
        assert results[1] == {"task": 2}


class TestIntegration:
    """Integration tests."""
    
    def test_context_with_metadata(self):
        """Test context integration with metadata."""
        from aiq.builder.context import AIQContext
        from aiq.runtime.user_metadata import RequestAttributes
        
        context = AIQContext.get()
        metadata = context.metadata
        
        assert isinstance(metadata, RequestAttributes)
    
    def test_context_with_invocation(self):
        """Test context with invocation nodes."""
        from aiq.builder.context import AIQContext
        from aiq.data_models.invocation_node import InvocationNode
        
        context = AIQContext.get()
        active = context.active_function
        
        assert isinstance(active, InvocationNode)
        assert active.function_name == "root"
    
    def test_error_handling(self):
        """Test error handling in context."""
        from aiq.builder.context import AIQContext
        
        context = AIQContext.get()
        
        with pytest.raises(Exception):
            with context.push_active_function("error_func", {}) as mgr:
                raise Exception("Test error")
    
    @pytest.mark.asyncio  
    async def test_async_error_handling(self):
        """Test async error handling."""
        from aiq.builder.context import AIQContext
        
        context = AIQContext.get()
        
        async def failing_function():
            with context.push_active_function("failing", {}) as mgr:
                await asyncio.sleep(0.001)
                raise ValueError("Async error")
        
        with pytest.raises(ValueError, match="Async error"):
            await failing_function()


# Test that basic imports work
def test_core_imports():
    """Test core module imports."""
    import aiq
    from aiq.builder import Builder
    from aiq.runtime import Runner, Session
    
    assert aiq is not None
    
    
def test_module_structure():
    """Test module structure is correct."""
    import aiq.builder
    import aiq.runtime  
    import aiq.data_models
    import aiq.cli
    
    # All modules should be importable
    assert all([
        aiq.builder,
        aiq.runtime,
        aiq.data_models,
        aiq.cli
    ])