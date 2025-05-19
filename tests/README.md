# AIQToolkit Test Suite

This directory contains all tests for the AIQToolkit project, organized into subdirectories based on test types and purposes.

## Directory Structure

- **`aiq/`** - Unit tests mirroring the source code structure
  - Tests for individual modules and components
  - Organized to match `src/aiq/` structure
  
- **`unit/`** - General unit tests
  - Core component tests
  - Basic functionality tests
  - Import and utility tests

- **`integration/`** - Integration tests
  - Tests that verify interaction between multiple components
  - Distributed system tests
  - Service integration tests

- **`e2e/`** - End-to-end tests
  - Complete workflow tests
  - NVIDIA GPU/infrastructure tests
  - Full system verification tests

- **`performance/`** - Performance and benchmark tests
  - Scalability tests
  - Load testing
  - Performance regression tests

- **`helpers/`** - Test utilities and helpers
  - Shared test utilities
  - Test fixtures
  - Meta-tests for test suite health

- **`reports/`** - Test reports and documentation
  - Test execution reports
  - Coverage reports
  - Test summary documentation

- **`test_data/`** - Test data and fixtures
  - Configuration files for testing
  - Sample data files
  - Mock data

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test category
```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Run tests for specific module
```bash
pytest tests/aiq/agent/
pytest tests/aiq/distributed/
```

### Run with coverage
```bash
pytest --cov=aiq tests/
```

### Run with specific markers
```bash
pytest -m "not slow" tests/  # Skip slow tests
pytest -m "gpu" tests/      # Only GPU tests
```

## Writing Tests

1. **Unit Tests**: Place in `unit/` or appropriate subdirectory in `aiq/`
2. **Integration Tests**: Place in `integration/`
3. **End-to-End Tests**: Place in `e2e/`
4. **Performance Tests**: Place in `performance/`

Follow the existing test patterns and naming conventions:
- Test files start with `test_`
- Test functions start with `test_`
- Use descriptive names that explain what is being tested

## Test Configuration

- `conftest.py` - Shared pytest fixtures and configuration
- Test data should be placed in `test_data/`
- Mock configurations should be in appropriate test directories