"""Tests for AIQ runtime components."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from aiq.runtime.runner import AIQRunner, AIQRunnerState
from aiq.runtime.session import AIQSessionManager
from aiq.runtime.user_metadata import RequestAttributes


class TestAIQRunner:
    """Test the AIQRunner class."""
    
    @pytest.fixture
    def mock_function(self):
        """Create a mock function."""
        func = Mock()
        func.name = "test_function"
        func.config = {"name": "test_function", "type": "function"}
        func.return_value = {"result": "success"}
        return func
    
    def test_runner_creation(self):
        """Test AIQRunner instance creation."""
        runner = AIQRunner()
        assert runner is not None
        assert runner.state == AIQRunnerState.UNINITIALIZED
    
    def test_runner_initialization(self, mock_function):
        """Test runner initialization."""
        runner = AIQRunner()
        result = runner.initialize(mock_function)
        
        assert result is True
        assert runner.state == AIQRunnerState.INITIALIZED
        assert runner.function == mock_function
    
    def test_runner_run_sync(self, mock_function):
        """Test synchronous run method."""
        runner = AIQRunner()
        runner.initialize(mock_function)
        
        with patch.object(runner, '_run_sync') as mock_run:
            mock_run.return_value = {"result": "success"}
            result = runner.run("test input")
            
            mock_run.assert_called_once_with("test input")
            assert result == {"result": "success"}
            assert runner.state == AIQRunnerState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_runner_run_async(self, mock_function):
        """Test asynchronous run method."""
        runner = AIQRunner()
        runner.initialize(mock_function)
        
        async_func = AsyncMock(return_value={"result": "async success"})
        mock_function.side_effect = async_func
        
        result = await runner.arun("test input")
        
        async_func.assert_called_once_with("test input")
        assert result == {"result": "async success"}
        assert runner.state == AIQRunnerState.COMPLETED
    
    def test_runner_error_handling(self, mock_function):
        """Test runner error handling."""
        runner = AIQRunner()
        runner.initialize(mock_function)
        
        mock_function.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            runner.run("test input")
        
        assert runner.state == AIQRunnerState.FAILED
    
    def test_runner_state_transitions(self, mock_function):
        """Test runner state transitions."""
        runner = AIQRunner()
        
        # Initial state
        assert runner.state == AIQRunnerState.UNINITIALIZED
        
        # After initialization
        runner.initialize(mock_function)
        assert runner.state == AIQRunnerState.INITIALIZED
        
        # During run
        with patch.object(runner, '_run_sync') as mock_run:
            def set_running(*args):
                assert runner.state == AIQRunnerState.RUNNING
                return {"result": "success"}
            
            mock_run.side_effect = set_running
            runner.run("test")
        
        # After completion
        assert runner.state == AIQRunnerState.COMPLETED


class TestAIQSessionManager:
    """Test the AIQSessionManager class."""
    
    def test_session_manager_creation(self):
        """Test SessionManager creation."""
        manager = AIQSessionManager()
        assert manager is not None
        assert hasattr(manager, '_sessions')
    
    def test_create_session(self):
        """Test session creation."""
        manager = AIQSessionManager()
        
        session_id = manager.create_session()
        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_id in manager._sessions
    
    def test_get_session(self):
        """Test getting a session."""
        manager = AIQSessionManager()
        
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        
        assert session is not None
        assert session['id'] == session_id
    
    def test_update_session(self):
        """Test updating a session."""
        manager = AIQSessionManager()
        
        session_id = manager.create_session()
        update_data = {"status": "active", "user": "test_user"}
        
        manager.update_session(session_id, update_data)
        session = manager.get_session(session_id)
        
        assert session['status'] == "active"
        assert session['user'] == "test_user"
    
    def test_delete_session(self):
        """Test deleting a session."""
        manager = AIQSessionManager()
        
        session_id = manager.create_session()
        manager.delete_session(session_id)
        
        assert session_id not in manager._sessions
        assert manager.get_session(session_id) is None
    
    def test_list_sessions(self):
        """Test listing all sessions."""
        manager = AIQSessionManager()
        
        # Create multiple sessions
        session_ids = [manager.create_session() for _ in range(3)]
        
        sessions = manager.list_sessions()
        assert len(sessions) == 3
        
        for session_id in session_ids:
            assert session_id in sessions


class TestRequestAttributes:
    """Test the RequestAttributes class."""
    
    def test_request_attributes_creation(self):
        """Test RequestAttributes creation."""
        attrs = RequestAttributes()
        assert attrs is not None
        assert hasattr(attrs, '_request')
    
    def test_method_property(self):
        """Test method property."""
        attrs = RequestAttributes()
        
        # Mock the request
        with patch.object(attrs._request, 'method', 'POST'):
            assert attrs.method == 'POST'
    
    def test_headers_property(self):
        """Test headers property."""
        attrs = RequestAttributes()
        
        # Mock headers
        mock_headers = {"Content-Type": "application/json"}
        with patch.object(attrs._request, 'headers', mock_headers):
            assert attrs.headers == mock_headers
    
    def test_query_params_property(self):
        """Test query params property."""
        attrs = RequestAttributes()
        
        # Mock query params
        mock_params = {"key": "value", "page": "1"}
        with patch.object(attrs._request, 'query_params', mock_params):
            assert attrs.query_params == mock_params
    
    def test_set_custom_attributes(self):
        """Test setting custom attributes."""
        attrs = RequestAttributes()
        
        # Should be able to set custom attributes
        attrs.custom_field = "custom_value"
        assert attrs.custom_field == "custom_value"


class TestRuntimeIntegration:
    """Integration tests for runtime components."""
    
    def test_runner_with_context(self):
        """Test runner with context integration."""
        from aiq.builder.context import AIQContext
        
        runner = AIQRunner()
        mock_function = Mock()
        mock_function.name = "context_test"
        mock_function.return_value = {"status": "ok"}
        
        runner.initialize(mock_function)
        
        # Get context
        context = AIQContext.get()
        
        # Run with context
        result = runner.run("test input")
        
        assert result == {"status": "ok"}
        assert runner.state == AIQRunnerState.COMPLETED
    
    def test_session_with_runner(self):
        """Test session manager with runner integration."""
        manager = AIQSessionManager()
        runner = AIQRunner()
        
        # Create session
        session_id = manager.create_session()
        
        # Associate runner with session
        manager.update_session(session_id, {"runner_id": id(runner)})
        
        session = manager.get_session(session_id)
        assert session["runner_id"] == id(runner)
    
    @pytest.mark.asyncio
    async def test_async_integration(self):
        """Test async components integration."""
        runner = AIQRunner()
        
        async_func = AsyncMock()
        async_func.name = "async_integration"
        async_func.return_value = {"async": True}
        
        runner.initialize(async_func)
        result = await runner.arun("async input")
        
        assert result == {"async": True}
        async_func.assert_called_once_with("async input")