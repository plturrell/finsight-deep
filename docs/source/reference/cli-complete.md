# Complete CLI Reference

## Overview

The AIQ Toolkit Command Line Interface (CLI) provides comprehensive access to all toolkit features, from workflow management to component configuration. This reference covers all available commands and options.

## Global Options

```bash
aiq [OPTIONS] COMMAND [ARGS]...

Options:
  --version              Show version and exit
  --help                 Show this message and exit
  --verbose, -v          Enable verbose logging
  --debug                Enable debug mode
  --config PATH          Path to configuration file
  --no-color             Disable colored output
  --output FORMAT        Output format (json, yaml, table)
```

## Commands

### aiq configure

Configure AIQ Toolkit settings and channels.

```bash
aiq configure [OPTIONS] SUBCOMMAND
```

#### Subcommands

##### channel

Manage registry channels.

```bash
aiq configure channel [OPTIONS] COMMAND
```

**Add Channel**
```bash
aiq configure channel add [OPTIONS] NAME URL

Options:
  --priority INTEGER     Channel priority (lower = higher priority)
  --auth-token TEXT      Authentication token for private channels
  --timeout INTEGER      Request timeout in seconds
  --verify-ssl / --no-verify-ssl  Verify SSL certificates

Example:
  aiq configure channel add nvidia https://registry.nvidia.com/aiq
```

**Remove Channel**
```bash
aiq configure channel remove NAME

Example:
  aiq configure channel remove nvidia
```

**Update Channel**
```bash
aiq configure channel update [OPTIONS] NAME

Options:
  --url TEXT             New channel URL
  --priority INTEGER     New priority
  --auth-token TEXT      New authentication token
  --timeout INTEGER      New timeout

Example:
  aiq configure channel update nvidia --priority 1
```

**List Channels**
```bash
aiq configure channel list

Example output:
  Configured channels:
  1. nvidia (priority: 1) - https://registry.nvidia.com/aiq
  2. pypi (priority: 2) - https://pypi.org/simple
  3. local (priority: 3) - file:///home/user/.aiq/local
```

### aiq eval

Evaluate workflows and components.

```bash
aiq eval [OPTIONS] CONFIG_FILE

Options:
  --output PATH          Output directory for results
  --dataset PATH         Dataset to use for evaluation
  --metrics TEXT         Comma-separated list of metrics
  --batch-size INTEGER   Evaluation batch size
  --num-workers INTEGER  Number of parallel workers
  --device TEXT          Device to use (cpu, cuda, cuda:0)
  --save-predictions     Save individual predictions
  --profile              Enable profiling during evaluation

Example:
  aiq eval config/eval.yaml --output results/ --metrics accuracy,f1 --device cuda:0
```

### aiq info

Display information about components and system.

```bash
aiq info [OPTIONS] SUBCOMMAND
```

#### Subcommands

##### channels

Show configured channels.

```bash
aiq info channels [OPTIONS]

Options:
  --format TEXT          Output format (table, json, yaml)
  --show-auth           Show authentication status

Example:
  aiq info channels --format json
```

##### components

List available components.

```bash
aiq info components [OPTIONS]

Options:
  --type TEXT            Filter by component type
  --channel TEXT         Filter by channel
  --search TEXT          Search components by name
  --show-versions        Show all available versions
  --format TEXT          Output format (table, json, yaml)

Example:
  aiq info components --type agent --search react
```

##### list-mcp

List MCP servers and tools.

```bash
aiq info list-mcp [OPTIONS]

Options:
  --servers              List MCP servers only
  --tools                List MCP tools only
  --format TEXT          Output format (table, json, yaml)

Example:
  aiq info list-mcp --tools --format json
```

### aiq registry

Manage component registry operations.

```bash
aiq registry [OPTIONS] SUBCOMMAND
```

#### Subcommands

##### publish

Publish a component to registry.

```bash
aiq registry publish [OPTIONS] PATH

Options:
  --channel TEXT         Target channel for publishing
  --version TEXT         Component version
  --force                Force overwrite existing version
  --dry-run              Perform dry run without publishing
  --sign                 Sign package with GPG

Example:
  aiq registry publish ./my-component --channel nvidia --version 1.0.0
```

##### pull

Pull a component from registry.

```bash
aiq registry pull [OPTIONS] COMPONENT

Options:
  --version TEXT         Specific version to pull
  --channel TEXT         Channel to pull from
  --output PATH          Output directory
  --no-deps              Skip dependencies
  --verify               Verify package signature

Example:
  aiq registry pull aiq/react-agent --version 2.1.0 --output ./components
```

##### remove

Remove a component from local registry.

```bash
aiq registry remove [OPTIONS] COMPONENT

Options:
  --version TEXT         Specific version to remove
  --all-versions         Remove all versions
  --force                Force removal without confirmation

Example:
  aiq registry remove aiq/old-component --all-versions
```

##### search

Search for components in registry.

```bash
aiq registry search [OPTIONS] QUERY

Options:
  --type TEXT            Filter by component type
  --channel TEXT         Search specific channel
  --limit INTEGER        Maximum results to return
  --sort TEXT            Sort by (name, popularity, updated)
  --format TEXT          Output format (table, json, yaml)

Example:
  aiq registry search "llm agent" --type agent --limit 10
```

### aiq run

Run a workflow from configuration.

```bash
aiq run [OPTIONS] CONFIG_FILE

Options:
  --input PATH           Input data file
  --output PATH          Output directory
  --params TEXT          Parameter overrides (key=value)
  --profile              Enable profiling
  --debug                Enable debug mode
  --max-steps INTEGER    Maximum workflow steps
  --timeout INTEGER      Execution timeout in seconds
  --device TEXT          Device to use (cpu, cuda)
  --stream               Stream output in real-time
  --save-state           Save workflow state for resumption

Example:
  aiq run workflow.yaml --input data.json --output results/ --params temperature=0.7 --device cuda:0
```

### aiq serve

Start API server for workflows.

```bash
aiq serve [OPTIONS] CONFIG_FILE

Options:
  --host TEXT            Server host address
  --port INTEGER         Server port
  --workers INTEGER      Number of worker processes
  --reload               Enable auto-reload
  --ssl-cert PATH        SSL certificate file
  --ssl-key PATH         SSL key file
  --cors                 Enable CORS
  --auth-token TEXT      Required authentication token
  --max-requests INT     Maximum concurrent requests
  --timeout INTEGER      Request timeout in seconds

Example:
  aiq serve api_config.yaml --host 0.0.0.0 --port 8000 --workers 4 --cors
```

### aiq start

Start frontend interfaces.

```bash
aiq start [OPTIONS] FRONTEND
```

#### Frontends

##### console

Start console interface.

```bash
aiq start console [OPTIONS]

Options:
  --workflow PATH        Workflow configuration file
  --history-file PATH    Command history file
  --prompt TEXT          Custom prompt text

Example:
  aiq start console --workflow config.yaml
```

##### fastapi

Start FastAPI server.

```bash
aiq start fastapi [OPTIONS]

Options:
  --config PATH          Server configuration file
  --host TEXT            Server host
  --port INTEGER         Server port
  --workers INTEGER      Number of workers
  --reload               Enable auto-reload
  --docs                 Enable API documentation

Example:
  aiq start fastapi --host 0.0.0.0 --port 8000 --docs
```

##### mcp

Start MCP server.

```bash
aiq start mcp [OPTIONS]

Options:
  --config PATH          MCP configuration file
  --port INTEGER         Server port
  --tools PATH           Tools configuration
  --auth TEXT            Authentication type

Example:
  aiq start mcp --config mcp.yaml --port 5000
```

##### ui

Start web UI (Streamlit interface).

```bash
aiq start ui [OPTIONS]

Options:
  --config PATH          UI configuration file
  --port INTEGER         Server port
  --theme TEXT           UI theme (light, dark)
  --no-browser           Don't open browser automatically

Example:
  aiq start ui --port 8501 --theme dark
```

### aiq uninstall

Uninstall components.

```bash
aiq uninstall [OPTIONS] COMPONENT

Options:
  --version TEXT         Specific version to uninstall
  --all-versions         Uninstall all versions
  --force                Force uninstall without confirmation
  --keep-deps            Keep dependencies

Example:
  aiq uninstall aiq/old-agent --all-versions --force
```

### aiq validate

Validate workflow configurations.

```bash
aiq validate [OPTIONS] CONFIG_FILE

Options:
  --schema PATH          Custom validation schema
  --strict               Enable strict validation
  --format TEXT          Output format for errors
  --fix                  Attempt to fix common issues

Example:
  aiq validate workflow.yaml --strict --format json
```

### aiq workflow

Workflow management commands.

```bash
aiq workflow [OPTIONS] SUBCOMMAND
```

#### Subcommands

##### new

Create a new workflow from template.

```bash
aiq workflow new [OPTIONS] NAME

Options:
  --template TEXT        Template to use
  --output PATH          Output directory
  --config TEXT          Initial configuration values
  --interactive          Interactive configuration mode

Example:
  aiq workflow new my-workflow --template react-agent --output ./workflows
```

##### list

List available workflows.

```bash
aiq workflow list [OPTIONS]

Options:
  --path PATH            Directory to search
  --format TEXT          Output format
  --filter TEXT          Filter by name pattern

Example:
  aiq workflow list --path ./workflows --filter "test*"
```

##### convert

Convert workflow between formats.

```bash
aiq workflow convert [OPTIONS] INPUT OUTPUT

Options:
  --from-format TEXT     Input format
  --to-format TEXT       Output format
  --validate             Validate after conversion

Example:
  aiq workflow convert workflow.json workflow.yaml --validate
```

## Configuration Files

### Global Configuration

Location: `~/.aiq/config.yaml`

```yaml
# Global AIQ configuration
aiq:
  # Default settings
  defaults:
    device: cuda
    log_level: info
    output_format: table
    
  # Channel configuration
  channels:
    - name: nvidia
      url: https://registry.nvidia.com/aiq
      priority: 1
      auth_token: ${NVIDIA_API_KEY}
      
    - name: pypi
      url: https://pypi.org/simple
      priority: 2
      
    - name: local
      url: file:///home/user/.aiq/local
      priority: 3
      
  # CLI preferences
  cli:
    color: true
    editor: vim
    history_size: 1000
    
  # Performance settings
  performance:
    max_workers: 8
    cache_size: 1000
    timeout: 300
```

### Workflow Configuration

```yaml
# Example workflow configuration
name: example_workflow
version: 1.0.0

components:
  llm:
    type: openai
    model: gpt-4
    temperature: 0.7
    
  agent:
    type: react
    max_steps: 10
    tools:
      - document_search
      - web_search
      
  retriever:
    type: milvus
    collection: knowledge_base
    top_k: 5
    
workflow:
  steps:
    - name: process_query
      component: agent
      input: ${user_query}
      
    - name: generate_response
      component: llm
      input: ${process_query.output}
```

## Environment Variables

```bash
# API Keys
export OPENAI_API_KEY=your_key
export NVIDIA_API_KEY=your_key

# Configuration
export AIQ_CONFIG_PATH=/path/to/config
export AIQ_LOG_LEVEL=debug
export AIQ_CACHE_DIR=/path/to/cache

# Performance
export AIQ_MAX_WORKERS=16
export AIQ_GPU_MEMORY_FRACTION=0.8
export AIQ_ENABLE_PROFILING=true

# Registry
export AIQ_REGISTRY_URL=https://custom.registry.com
export AIQ_REGISTRY_TOKEN=your_token
```

## Advanced Usage

### Command Chaining

```bash
# Build and run workflow
aiq validate workflow.yaml && aiq run workflow.yaml --profile

# Pull component and start server
aiq registry pull aiq/api-server && aiq serve config.yaml
```

### Scripting

```bash
#!/bin/bash
# deploy.sh - Deploy AIQ workflow

# Validate configuration
if ! aiq validate config.yaml; then
    echo "Invalid configuration"
    exit 1
fi

# Run with monitoring
aiq run config.yaml \
    --output results/ \
    --profile \
    --device cuda:0 \
    --stream | tee workflow.log
    
# Generate report
aiq eval results/ --metrics all --format html > report.html
```

### Aliases

```bash
# .bashrc or .zshrc
alias aiq-dev="aiq --config ~/.aiq/dev-config.yaml"
alias aiq-prod="aiq --config ~/.aiq/prod-config.yaml"
alias aiq-validate="aiq validate --strict"
alias aiq-profile="aiq run --profile --device cuda:0"
```

## Troubleshooting

### Common Issues

1. **Command not found**
   ```bash
   # Ensure AIQ is in PATH
   export PATH=$PATH:/path/to/aiq/bin
   ```

2. **Permission denied**
   ```bash
   # Fix permissions
   chmod +x $(which aiq)
   ```

3. **Configuration errors**
   ```bash
   # Validate configuration
   aiq validate --strict config.yaml
   ```

4. **Registry connection issues**
   ```bash
   # Test registry connection
   aiq registry search test --debug
   ```

### Debug Mode

```bash
# Enable debug logging
export AIQ_LOG_LEVEL=debug
aiq --debug run workflow.yaml

# Verbose output
aiq -vvv info components
```

### Log Files

```bash
# Default log location
~/.aiq/logs/aiq.log

# Custom log location
export AIQ_LOG_FILE=/path/to/custom.log
```

## Examples

### Complete Workflow Example

```bash
# 1. Create new workflow
aiq workflow new research-assistant --template react-agent

# 2. Configure components
cd research-assistant
vim config.yaml

# 3. Validate configuration
aiq validate config.yaml --strict

# 4. Test locally
aiq run config.yaml --input test.json --debug

# 5. Start API server
aiq serve config.yaml --host 0.0.0.0 --port 8000

# 6. Deploy to production
aiq registry publish . --channel internal --version 1.0.0
```

### Development Workflow

```bash
# Start development environment
aiq start console --workflow dev-config.yaml

# In another terminal, start UI
aiq start ui --port 8501 --no-browser

# Monitor logs
tail -f ~/.aiq/logs/aiq.log
```

## See Also

- [Workflow Configuration](../workflows/workflow-configuration.md)
- [API Server Reference](api-server-endpoints.md)
- [Component Development](../extend/functions.md)