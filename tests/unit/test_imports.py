"""Test that all the main imports work correctly."""

import pytest


def test_core_imports():
    """Test that core AIQ imports work."""
    import aiq
    from aiq.builder import Builder
    from aiq.runtime import Runner, Session
    
    assert hasattr(aiq, 'Builder')
    assert hasattr(aiq, 'Runner')  
    assert hasattr(aiq, 'Session')


def test_builder_imports():
    """Test builder module imports."""
    from aiq.builder.builder import Builder
    from aiq.builder.context import AIQContext
    from aiq.builder.workflow import Workflow
    from aiq.builder.function import Function
    
    # Basic assertions to ensure classes exist
    assert AIQContext is not None
    assert Builder is not None
    assert Workflow is not None
    assert Function is not None


def test_data_models():
    """Test data model imports."""
    from aiq.data_models.common import Component
    from aiq.data_models.config import Config
    from aiq.data_models.workflow import WorkflowConfig


def test_runtime_imports():
    """Test runtime module imports."""
    from aiq.runtime.runner import AIQRunner
    from aiq.runtime.session import AIQSessionManager
    
    # Test aliases work
    from aiq.runtime import Runner, Session
    assert Runner == AIQRunner
    assert Session == AIQSessionManager


def test_llm_imports():
    """Test LLM provider imports."""
    try:
        from aiq.llm.nim_llm import NimLLM
        from aiq.llm.openai_llm import OpenaiLLM
    except ImportError as e:
        # These might have optional dependencies
        pytest.skip(f"Optional dependency not installed: {e}")


def test_embedder_imports():
    """Test embedder imports."""
    try:
        from aiq.embedder.nim_embedder import NimEmbedder  
        from aiq.embedder.openai_embedder import OpenaiEmbedder
    except ImportError as e:
        # These might have optional dependencies
        pytest.skip(f"Optional dependency not installed: {e}")


def test_tool_imports():
    """Test tool imports."""
    from aiq.tool.datetime_tools import DatetimeTool, DatetimeToolSchema
    from aiq.tool.document_search import DocumentSearchTool
    from aiq.tool.server_tools import ListToolsTool, RunToolTool


def test_cli_imports():
    """Test CLI imports."""
    from aiq.cli.entrypoint import aiq
    from aiq.cli.main import main
    assert aiq is not None
    assert main is not None