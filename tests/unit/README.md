# Unit Tests

This directory contains general unit tests for AIQToolkit components.

## What Goes Here

- Tests for individual classes and functions
- Tests that don't require external dependencies
- Fast, isolated tests that test single units of functionality
- Core functionality tests

## Test Files

- `test_imports.py` - Verify module imports work correctly
- `test_conftest.py` - Test suite configuration tests
- `test_data_models.py` - Data model validation tests
- `test_builder_components.py` - Builder component tests
- `test_runtime_components.py` - Runtime component tests
- `test_simplified_components.py` - Simplified component tests
- `test_tools.py` - Tool functionality tests
- `test_basic_functionality.py` - Basic system functionality
- `test_continuous.py` - Continuous integration tests
- `test_isolated_components.py` - Isolated component testing
- `test_plugins.py` - Plugin system tests

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_data_models.py

# Run with verbose output
pytest -v tests/unit/
```