"""Tests for AIQ data models."""

import pytest
from datetime import datetime
from typing import Dict, Any
import json

from aiq.data_models.common import Component
from aiq.data_models.config import Config
from aiq.data_models.workflow import WorkflowConfig, FunctionSpec
from aiq.data_models.function import FunctionConfig, FunctionInput, FunctionOutput
from aiq.data_models.invocation_node import InvocationNode
from aiq.data_models.intermediate_step import IntermediateStep, IntermediateStepType, IntermediateStepPayload
from aiq.data_models.api_server import Request, Response


class TestCommonModels:
    """Test common data models."""
    
    def test_component_creation(self):
        """Test Component model creation."""
        component = Component(
            type="function",
            name="test_component",
            config={"key": "value"}
        )
        
        assert component.type == "function"
        assert component.name == "test_component"
        assert component.config == {"key": "value"}
    
    def test_component_validation(self):
        """Test Component validation."""
        # Should raise validation error for missing required fields
        with pytest.raises(Exception):
            Component()


class TestConfigModels:
    """Test configuration models."""
    
    def test_config_creation(self):
        """Test Config model creation."""
        config = Config(
            name="test_config",
            version="1.0.0",
            type="workflow"
        )
        
        assert config.name == "test_config"
        assert config.version == "1.0.0"
        assert config.type == "workflow"
    
    def test_config_with_metadata(self):
        """Test Config with metadata."""
        config = Config(
            name="test_config",
            version="1.0.0",
            type="workflow",
            metadata={"author": "test", "created": "2024-01-01"}
        )
        
        assert config.metadata["author"] == "test"
        assert config.metadata["created"] == "2024-01-01"


class TestWorkflowModels:
    """Test workflow data models."""
    
    def test_function_spec_creation(self):
        """Test FunctionSpec creation."""
        spec = FunctionSpec(
            type="function",
            name="test_function",
            description="A test function",
            config={"param": "value"}
        )
        
        assert spec.type == "function"
        assert spec.name == "test_function"
        assert spec.description == "A test function"
        assert spec.config["param"] == "value"
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig creation."""
        function_spec = FunctionSpec(
            type="function",
            name="step1",
            config={}
        )
        
        workflow_config = WorkflowConfig(
            name="test_workflow",
            version="1.0.0",
            description="Test workflow",
            functions=[function_spec]
        )
        
        assert workflow_config.name == "test_workflow"
        assert len(workflow_config.functions) == 1
        assert workflow_config.functions[0].name == "step1"
    
    def test_workflow_config_to_dict(self):
        """Test WorkflowConfig serialization."""
        spec = FunctionSpec(type="function", name="func1", config={})
        workflow = WorkflowConfig(
            name="workflow1",
            version="1.0",
            functions=[spec]
        )
        
        # Should be serializable
        workflow_dict = workflow.model_dump()
        assert workflow_dict["name"] == "workflow1"
        assert len(workflow_dict["functions"]) == 1


class TestFunctionModels:
    """Test function data models."""
    
    def test_function_input_creation(self):
        """Test FunctionInput creation."""
        func_input = FunctionInput(
            data={"key": "value"},
            context={"user": "test_user"}
        )
        
        assert func_input.data["key"] == "value"
        assert func_input.context["user"] == "test_user"
    
    def test_function_output_creation(self):
        """Test FunctionOutput creation."""
        func_output = FunctionOutput(
            result={"status": "success"},
            metadata={"duration": 1.5}
        )
        
        assert func_output.result["status"] == "success"
        assert func_output.metadata["duration"] == 1.5
    
    def test_function_config_creation(self):
        """Test FunctionConfig creation."""
        func_config = FunctionConfig(
            name="test_function",
            type="custom",
            description="Test function",
            parameters={"timeout": 30}
        )
        
        assert func_config.name == "test_function"
        assert func_config.type == "custom"
        assert func_config.parameters["timeout"] == 30


class TestInvocationModels:
    """Test invocation node models."""
    
    def test_invocation_node_creation(self):
        """Test InvocationNode creation."""
        node = InvocationNode(
            function_id="func_123",
            function_name="process_data"
        )
        
        assert node.function_id == "func_123"
        assert node.function_name == "process_data"
        assert node.parent_id is None
    
    def test_invocation_node_with_parent(self):
        """Test InvocationNode with parent."""
        node = InvocationNode(
            function_id="child_123",
            function_name="child_function",
            parent_id="parent_123",
            parent_name="parent_function"
        )
        
        assert node.parent_id == "parent_123"
        assert node.parent_name == "parent_function"
    
    def test_invocation_node_hierarchy(self):
        """Test InvocationNode hierarchy."""
        root = InvocationNode(function_id="root", function_name="root")
        child = InvocationNode(
            function_id="child",
            function_name="child",
            parent_id=root.function_id
        )
        
        assert child.parent_id == root.function_id


class TestIntermediateStepModels:
    """Test intermediate step models."""
    
    def test_intermediate_step_creation(self):
        """Test IntermediateStep creation."""
        step = IntermediateStep(
            UUID="step_123",
            type=IntermediateStepType.FUNCTION_START,
            timestamp=datetime.now(),
            data={"input": "test"}
        )
        
        assert step.UUID == "step_123"
        assert step.type == IntermediateStepType.FUNCTION_START
        assert step.data["input"] == "test"
    
    def test_intermediate_step_types(self):
        """Test different IntermediateStep types."""
        types = [
            IntermediateStepType.FUNCTION_START,
            IntermediateStepType.FUNCTION_END,
            IntermediateStepType.TOOL_CALL,
            IntermediateStepType.LLM_CALL
        ]
        
        for step_type in types:
            step = IntermediateStep(
                UUID=f"step_{step_type.value}",
                type=step_type,
                timestamp=datetime.now()
            )
            assert step.type == step_type
    
    def test_intermediate_step_payload(self):
        """Test IntermediateStepPayload."""
        payload = IntermediateStepPayload(
            UUID="payload_123",
            event_type=IntermediateStepType.TOOL_CALL,
            name="search_tool",
            data={"query": "test search"}
        )
        
        assert payload.UUID == "payload_123"
        assert payload.event_type == IntermediateStepType.TOOL_CALL
        assert payload.name == "search_tool"


class TestAPIServerModels:
    """Test API server models."""
    
    def test_request_creation(self):
        """Test Request model creation."""
        request = Request(
            method="POST",
            url="/api/v1/workflow",
            headers={"Content-Type": "application/json"},
            body={"workflow": "test"}
        )
        
        assert request.method == "POST"
        assert request.url == "/api/v1/workflow"
        assert request.headers["Content-Type"] == "application/json"
    
    def test_response_creation(self):
        """Test Response model creation."""
        response = Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"result": "success"}
        )
        
        assert response.status_code == 200
        assert response.body["result"] == "success"
    
    def test_request_with_query_params(self):
        """Test Request with query parameters."""
        request = Request(
            method="GET",
            url="/api/v1/data",
            query_params={"page": "1", "limit": "10"}
        )
        
        assert request.query_params["page"] == "1"
        assert request.query_params["limit"] == "10"


class TestModelSerialization:
    """Test model serialization/deserialization."""
    
    def test_workflow_config_json_serialization(self):
        """Test WorkflowConfig JSON serialization."""
        spec = FunctionSpec(type="function", name="func1", config={})
        workflow = WorkflowConfig(
            name="test_workflow",
            version="1.0",
            functions=[spec]
        )
        
        # Serialize to JSON
        json_str = workflow.model_dump_json()
        
        # Deserialize from JSON
        parsed = WorkflowConfig.model_validate_json(json_str)
        assert parsed.name == "test_workflow"
        assert len(parsed.functions) == 1
    
    def test_nested_model_serialization(self):
        """Test nested model serialization."""
        node = InvocationNode(
            function_id="123",
            function_name="test",
            metadata={"nested": {"key": "value"}}
        )
        
        # Should handle nested structures
        data = node.model_dump()
        assert data["metadata"]["nested"]["key"] == "value"
    
    def test_datetime_serialization(self):
        """Test datetime serialization."""
        now = datetime.now()
        step = IntermediateStep(
            UUID="step_123",
            type=IntermediateStepType.FUNCTION_START,
            timestamp=now
        )
        
        # Should serialize datetime properly
        json_str = step.model_dump_json()
        parsed = json.loads(json_str)
        
        # Timestamp should be serialized as string
        assert isinstance(parsed["timestamp"], str)