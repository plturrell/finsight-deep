"""Tests for AIQ tools."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pytz

from aiq.tool.datetime_tools import DatetimeTool, DatetimeToolSchema
from aiq.tool.document_search import DocumentSearchTool
from aiq.tool.server_tools import ListToolsTool, RunToolTool


class TestDatetimeTools:
    """Test datetime tools."""
    
    def test_datetime_tool_creation(self):
        """Test DatetimeTool creation."""
        tool = DatetimeTool()
        assert tool is not None
        assert tool.name == "datetime_tool"
        assert tool.description is not None
    
    def test_datetime_tool_schema(self):
        """Test DatetimeToolSchema."""
        schema = DatetimeToolSchema(
            date="2024-01-01",
            timezone="UTC",
            format="%Y-%m-%d %H:%M:%S"
        )
        
        assert schema.date == "2024-01-01"
        assert schema.timezone == "UTC"
        assert schema.format == "%Y-%m-%d %H:%M:%S"
    
    def test_get_current_time(self):
        """Test getting current time."""
        tool = DatetimeTool()
        
        # Test default (UTC)
        result = tool._call(DatetimeToolSchema())
        assert "current_time" in result
        assert isinstance(result["current_time"], str)
    
    def test_get_time_with_timezone(self):
        """Test getting time with specific timezone."""
        tool = DatetimeTool()
        
        schema = DatetimeToolSchema(timezone="America/New_York")
        result = tool._call(schema)
        
        assert "current_time" in result
        assert "timezone" in result
        assert result["timezone"] == "America/New_York"
    
    def test_parse_specific_date(self):
        """Test parsing specific date."""
        tool = DatetimeTool()
        
        schema = DatetimeToolSchema(
            date="2024-01-01 12:00:00",
            format="%Y-%m-%d %H:%M:%S"
        )
        result = tool._call(schema)
        
        assert "parsed_time" in result
        assert "2024-01-01" in result["parsed_time"]
    
    def test_invalid_timezone(self):
        """Test invalid timezone handling."""
        tool = DatetimeTool()
        
        schema = DatetimeToolSchema(timezone="Invalid/Timezone")
        
        with pytest.raises(Exception):
            tool._call(schema)
    
    def test_invalid_date_format(self):
        """Test invalid date format handling."""
        tool = DatetimeTool()
        
        schema = DatetimeToolSchema(
            date="invalid date",
            format="%Y-%m-%d"
        )
        
        with pytest.raises(Exception):
            tool._call(schema)


class TestDocumentSearchTool:
    """Test document search tool."""
    
    @pytest.fixture
    def mock_documents(self):
        """Create mock documents."""
        return [
            {"id": "1", "content": "Python programming guide", "metadata": {"type": "tutorial"}},
            {"id": "2", "content": "JavaScript basics", "metadata": {"type": "tutorial"}},
            {"id": "3", "content": "Python advanced features", "metadata": {"type": "advanced"}}
        ]
    
    def test_document_search_tool_creation(self):
        """Test DocumentSearchTool creation."""
        tool = DocumentSearchTool()
        assert tool is not None
        assert tool.name == "document_search"
    
    def test_search_documents(self, mock_documents):
        """Test searching documents."""
        tool = DocumentSearchTool()
        
        # Mock the search functionality
        with patch.object(tool, '_search_documents') as mock_search:
            mock_search.return_value = [mock_documents[0], mock_documents[2]]
            
            result = tool._call({"query": "Python"})
            
            assert len(result["results"]) == 2
            assert all("Python" in doc["content"] for doc in result["results"])
    
    def test_search_with_filters(self, mock_documents):
        """Test searching with filters."""
        tool = DocumentSearchTool()
        
        with patch.object(tool, '_search_documents') as mock_search:
            mock_search.return_value = [mock_documents[0]]
            
            result = tool._call({
                "query": "Python",
                "filters": {"type": "tutorial"}
            })
            
            assert len(result["results"]) == 1
            assert result["results"][0]["metadata"]["type"] == "tutorial"
    
    def test_search_with_limit(self, mock_documents):
        """Test searching with result limit."""
        tool = DocumentSearchTool()
        
        with patch.object(tool, '_search_documents') as mock_search:
            mock_search.return_value = mock_documents[:1]
            
            result = tool._call({
                "query": "programming",
                "limit": 1
            })
            
            assert len(result["results"]) == 1
    
    def test_empty_search_results(self):
        """Test empty search results."""
        tool = DocumentSearchTool()
        
        with patch.object(tool, '_search_documents') as mock_search:
            mock_search.return_value = []
            
            result = tool._call({"query": "nonexistent"})
            
            assert result["results"] == []
            assert result["count"] == 0


class TestServerTools:
    """Test server tools."""
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        return {
            "datetime_tool": DatetimeTool(),
            "document_search": DocumentSearchTool(),
            "list_tools": ListToolsTool()
        }
    
    def test_list_tools_creation(self):
        """Test ListToolsTool creation."""
        tool = ListToolsTool()
        assert tool is not None
        assert tool.name == "list_tools"
    
    def test_list_available_tools(self, mock_tool_registry):
        """Test listing available tools."""
        tool = ListToolsTool()
        
        with patch('aiq.tool.server_tools.get_tool_registry') as mock_registry:
            mock_registry.return_value = mock_tool_registry
            
            result = tool._call({})
            
            assert "tools" in result
            assert len(result["tools"]) == 3
            assert "datetime_tool" in result["tools"]
    
    def test_list_tools_with_filter(self, mock_tool_registry):
        """Test listing tools with filter."""
        tool = ListToolsTool()
        
        with patch('aiq.tool.server_tools.get_tool_registry') as mock_registry:
            mock_registry.return_value = mock_tool_registry
            
            result = tool._call({"filter": "date"})
            
            assert "tools" in result
            assert "datetime_tool" in result["tools"]
            assert "document_search" not in result["tools"]
    
    def test_run_tool_creation(self):
        """Test RunToolTool creation."""
        tool = RunToolTool()
        assert tool is not None
        assert tool.name == "run_tool"
    
    def test_run_specific_tool(self, mock_tool_registry):
        """Test running a specific tool."""
        tool = RunToolTool()
        
        with patch('aiq.tool.server_tools.get_tool_registry') as mock_registry:
            mock_registry.return_value = mock_tool_registry
            
            result = tool._call({
                "tool_name": "datetime_tool",
                "tool_input": {"timezone": "UTC"}
            })
            
            assert "result" in result
            assert result["tool_name"] == "datetime_tool"
    
    def test_run_nonexistent_tool(self, mock_tool_registry):
        """Test running a nonexistent tool."""
        tool = RunToolTool()
        
        with patch('aiq.tool.server_tools.get_tool_registry') as mock_registry:
            mock_registry.return_value = mock_tool_registry
            
            with pytest.raises(Exception, match="Tool not found"):
                tool._call({
                    "tool_name": "nonexistent_tool",
                    "tool_input": {}
                })
    
    def test_run_tool_error_handling(self, mock_tool_registry):
        """Test error handling when running tools."""
        tool = RunToolTool()
        
        # Mock a tool that raises an error
        mock_tool = Mock()
        mock_tool._call.side_effect = Exception("Tool execution error")
        mock_tool_registry["error_tool"] = mock_tool
        
        with patch('aiq.tool.server_tools.get_tool_registry') as mock_registry:
            mock_registry.return_value = mock_tool_registry
            
            with pytest.raises(Exception, match="Tool execution error"):
                tool._call({
                    "tool_name": "error_tool",
                    "tool_input": {}
                })


class TestToolIntegration:
    """Integration tests for tools."""
    
    def test_tool_chaining(self):
        """Test chaining multiple tools."""
        list_tool = ListToolsTool()
        run_tool = RunToolTool()
        
        # Mock registry
        registry = {
            "datetime_tool": DatetimeTool(),
            "list_tools": list_tool
        }
        
        with patch('aiq.tool.server_tools.get_tool_registry') as mock_registry:
            mock_registry.return_value = registry
            
            # First, list tools
            list_result = list_tool._call({})
            available_tools = list_result["tools"]
            
            # Then run one of them
            if "datetime_tool" in available_tools:
                run_result = run_tool._call({
                    "tool_name": "datetime_tool",
                    "tool_input": {"timezone": "UTC"}
                })
                
                assert run_result["tool_name"] == "datetime_tool"
                assert "result" in run_result
    
    def test_tool_with_complex_input(self):
        """Test tool with complex input structure."""
        tool = DocumentSearchTool()
        
        complex_input = {
            "query": "machine learning",
            "filters": {
                "type": "research",
                "date_range": {
                    "start": "2023-01-01",
                    "end": "2024-01-01"
                }
            },
            "options": {
                "fuzzy_match": True,
                "relevance_threshold": 0.8
            }
        }
        
        with patch.object(tool, '_search_documents') as mock_search:
            mock_search.return_value = []
            
            result = tool._call(complex_input)
            
            # Should handle complex input without errors
            assert "results" in result
            assert isinstance(result["results"], list)