# AIQToolkit Plugin Packages

This directory contains plugin packages that extend AIQToolkit with integrations for popular AI frameworks.

## Available Plugins

- `aiqtoolkit_agno/` - Agno framework integration
- `aiqtoolkit_crewai/` - CrewAI integration
- `aiqtoolkit_langchain/` - LangChain integration
- `aiqtoolkit_llama_index/` - LlamaIndex integration
- `aiqtoolkit_mem0ai/` - Mem0 AI memory system integration
- `aiqtoolkit_semantic_kernel/` - Microsoft Semantic Kernel integration
- `aiqtoolkit_test/` - Test utilities for plugin development
- `aiqtoolkit_weave/` - Weights & Biases Weave integration
- `aiqtoolkit_zep_cloud/` - Zep Cloud memory integration

## Compatibility

The `compat/` directory contains backward compatibility packages for the legacy `agentiq` naming.

## Installation

Install individual plugins:
```bash
pip install aiqtoolkit-langchain
pip install aiqtoolkit-llama-index
```

## Development

Each plugin follows the same structure:
- `pyproject.toml` - Package metadata
- `src/aiq/plugins/` - Plugin implementation
- `tests/` - Plugin tests

## Creating a Plugin

See the test plugin (`aiqtoolkit_test`) for a minimal example of plugin structure.