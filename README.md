<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

![NVIDIA Agent Intelligence Toolkit](./docs/source/_static/aiqtoolkit_banner.png "AIQ Toolkit banner image")

# NVIDIA Agent Intelligence Toolkit

Agent Intelligence Toolkit (AIQ Toolkit) is a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework.

> Note: Agent Intelligence Toolkit was previously known as <!-- vale off -->AgentIQ<!-- vale on -->, however the API has not changed and is fully compatible with previous releases. Users should update their dependencies to depend on `aiqtoolkit` instead of `agentiq`. I transitional package named `agentiq` is available for backwards compatibility, but will be removed in the future.

## Key Features

- [**Framework Agnostic:**](./docs/source/quick-start/installing.md#framework-integrations) AIQ Toolkit works side-by-side and around existing agentic frameworks, such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), and [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), as well as customer enterprise frameworks and simple Python agents. This allows you to use your current technology stack without replatforming. AIQ Toolkit complements any existing agentic framework or memory tool you're using and isn't tied to any specific agentic framework, long-term memory, or data source.

- [**Reusability:**](./docs/source/extend/sharing-components.md) Every agent, tool, and agentic workflow in this library exists as a function call that works together in complex software applications. The composability between these agents, tools, and workflows allows you to build once and reuse in different scenarios.

- [**Rapid Development:**](docs/source/tutorials/customize-a-workflow.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs. This allows you and your development teams to move quickly if you're already developing with agents.

- [**Profiling:**](./docs/source/workflows/profiler.md) Use the profiler to profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks. While we encourage you to wrap (decorate) every tool and agent to get the most out of the profiler, you have the freedom to integrate your tools, agents, and workflows to whatever level you want. You start small and go to where you believe you'll see the most value and expand from there.

- [**Observability:**](./docs/source/workflows/observe/index.md) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool, with examples using [Phoenix](./docs/source/workflows/observe/observe-workflow-with-phoenix.md) and [W&B Weave](./docs/source/workflows/observe/observe-workflow-with-weave.md).

- [**Evaluation System:**](./docs/source/workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

- [**User Interface:**](./docs/source/quick-start/launching-ui.md) Use the AIQ Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.

- [**Full MCP Support:**](./docs/source/workflows/mcp/index.md) Compatible with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can use AIQ Toolkit as an [MCP client](./docs/source/workflows/mcp/mcp-client.md) to connect to and use tools served by remote MCP servers. You can also use AIQ Toolkit as an [MCP server](./docs/source/workflows/mcp/mcp-server.md) to publish tools via MCP.

- **Local GPU Acceleration:** Leverage NVIDIA GPUs for faster LLM inference when using compatible providers like NIM. Includes custom CUDA kernels for specific operations like similarity calculations.

- **Digital Human UI:** Interactive chat interface with visualization capabilities for workflow outputs. Includes avatar-based conversation system for enhanced user experience.

- **Multiple Retrieval Backends:** Support for various vector databases and retrieval systems including local embeddings, cloud services, and NVIDIA's NeMo Retriever integration.

- **Flexible LLM Integration:** Works with multiple LLM providers including OpenAI, Anthropic, NVIDIA NIM, and local models. Easy to switch between providers via configuration.

- **Workflow Orchestration:** Define complex multi-step workflows with conditional logic, tool integration, and state management. Supports various agent architectures including ReAct, ReWOO, and tool-calling agents.

## Project Organization

The project has been reorganized for better structure:

- **Test files**: All test files have been moved to `tests/`
- **Scripts**: Deployment and utility scripts are in `scripts/`
  - Docker-related scripts are in `scripts/docker/`
- **Configuration examples**: Example config files are in `config/examples/`  
- **Docker files**: Docker-compose files are in `docker/`
- **Requirements**: Additional requirements files are in `requirements/`

With AIQ Toolkit, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

## Component Overview

The following diagram illustrates the key components of AIQ Toolkit and how they interact. It provides a high-level view of the architecture, including agents, plugins, workflows, and user interfaces. Use this as a reference to understand how to integrate and extend AIQ Toolkit in your projects.

![AIQ Toolkit Components Diagram](docs/source/_static/aiqtoolkit_gitdiagram.png)

## Links

 * [Documentation](https://docs.nvidia.com/aiqtoolkit): Explore the full documentation for AIQ Toolkit.
 * [Get Started Guide](./docs/source/quick-start/installing.md): Set up your environment and start building with AIQ Toolkit.
 * [Examples](./examples/README.md): Explore examples of AIQ Toolkit workflows located in the [`examples`](./examples) directory of the source repository.
 * [Create and Customize AIQ Toolkit Workflows](docs/source/tutorials/customize-a-workflow.md): Learn how to create and customize AIQ Toolkit workflows.
 * [Evaluate with AIQ Toolkit](./docs/source/workflows/evaluate.md): Learn how to evaluate your AIQ Toolkit workflows.
 * [Troubleshooting](./docs/source/troubleshooting.md): Get help with common issues.


## Get Started

### Prerequisites

Before you begin using AIQ Toolkit, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install [Python (3.11 or 3.12)](https://www.python.org/downloads/)

### Install From Source

1. Clone the AIQ Toolkit repository to your local machine.
    ```bash
    git clone git@github.com:NVIDIA/AIQToolkit.git aiqtoolkit
    cd aiqtoolkit
    ```

2. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

3. Fetch the data sets by downloading the LFS files.
    ```bash
    git lfs install
    git lfs fetch
    git lfs pull
    ```

4. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    ```
    Make sure the environment is built with Python version `3.11` or `3.12`. If you have multiple Python versions installed,
    you can specify the desired version using the `--python` flag. For example, to use Python 3.11:
    ```bash
    uv venv --seed .venv --python 3.11
    ```
    You can replace `--python 3.11` with any other Python version (`3.11` or `3.12`) that you have installed.

5. Install the AIQ Toolkit library.
    To install the AIQ Toolkit library along with all of the optional dependencies. Including developer tools (`--all-groups`) and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

    Alternatively to install just the core AIQ Toolkit without any plugins, run the following:
    ```bash
    uv sync
    ```

    At this point individual plugins, which are located under the `packages` directory, can be installed with the following command `uv pip install -e '.[<plugin_name>]'`.
    For example, to install the `langchain` plugin, run the following:
    ```bash
    uv pip install -e '.[langchain]'
    ```

    > [!NOTE]
    > Many of the example workflows require plugins, and following the documented steps in one of these examples will in turn install the necessary plugins. For example following the steps in the `examples/simple/README.md` guide will install the `aiqtoolkit-langchain` plugin if you haven't already done so.


    In addition to plugins, there are optional dependencies needed for profiling. To install these dependencies, run the following:
    ```bash
    uv pip install -e '.[profiling]'
    ```

6. Verify the installation using the AIQ Toolkit CLI

   ```bash
   aiq --version
   ```

   This should output the AIQ Toolkit version which is currently installed.

## New Advanced Features

### Self-Correcting AI Systems

AIQ Toolkit now includes autonomous error detection and correction:

```python
from aiq import SelfCorrectingResearchSystem, ContentType

# Initialize self-correcting system
system = SelfCorrectingResearchSystem(
    enable_gpu=True,
    correction_strategy=CorrectionStrategy.POST_GENERATION
)

# Process with automatic error correction
result = await system.process_query(
    query="Explain quantum computing",
    content_type=ContentType.FACTUAL_REPORT,
    enable_self_correction=True
)

print(f"Errors corrected: {result.error_count}")
print(f"Confidence score: {result.confidence_score}")
```

### GPU-Accelerated Research Execution

Advanced research task execution with GPU optimization:

```python
from aiq.research import ResearchTaskExecutor, ResearchTask, TaskType

# Initialize executor with GPU optimization
executor = ResearchTaskExecutor(
    num_gpus=torch.cuda.device_count(),
    enable_optimization=True
)

# Create research task
task = ResearchTask(
    task_id="research_1",
    task_type=TaskType.RETRIEVAL,
    query="Latest advances in transformer architectures",
    target_latency_ms=100
)

# Execute with GPU acceleration
result = await executor.execute_task(task)
```

### Real-Time Verification

```python
from aiq.verification import VerificationSystem

# Initialize verification system
verifier = VerificationSystem({
    'enable_source_validation': True,
    'confidence_methods': ['bayesian', 'fuzzy', 'dempster_shafer']
})

# Verify claims
result = verifier.verify_claim(
    claim="GPT-3 has 175 billion parameters",
    sources=[{'url': 'https://arxiv.org/abs/2005.14165', 'type': 'paper'}]
)
```

### Hardware Optimization

```python
from aiq.hardware import TensorCoreOptimizer, ResourcePredictor

# Optimize model for Tensor Cores
optimizer = TensorCoreOptimizer()
optimized_model = optimizer.optimize_model(your_model)

# Predict resource requirements
predictor = ResourcePredictor()
requirements = predictor.predict_requirements(
    model,
    input_shape=(32, 128, 768),
    target_batch_size=64
)
```

## Hello World Example

1. Ensure you have set the `NVIDIA_API_KEY` environment variable to allow the example to use NVIDIA NIMs. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

   ```bash
   export NVIDIA_API_KEY=<your_api_key>
   ```

2. Create the AIQ Toolkit workflow configuration file. This file will define the agents, tools, and workflows that will be used in the example. Save the following as `workflow.yaml`:

   ```yaml
   functions:
      # Add a tool to search wikipedia
      wikipedia_search:
         _type: wiki_search
         max_results: 2

   llms:
      # Tell AIQ Toolkit which LLM to use for the agent
      nim_llm:
         _type: nim
         model_name: meta/llama-3.1-70b-instruct
         temperature: 0.0

   workflow:
      # Use an agent that 'reasons' and 'acts'
      _type: react_agent
      # Give it access to our wikipedia search tool
      tool_names: [wikipedia_search]
      # Tell it which LLM to use
      llm_name: nim_llm
      # Make it verbose
      verbose: true
      # Retry parsing errors because LLMs are non-deterministic
      retry_parsing_errors: true
      # Retry up to 3 times
      max_retries: 3
   ```

3. Run the Hello World example using the `aiq` CLI and the `workflow.yaml` file.

   ```bash
   aiq run --config_file workflow.yaml --input "List five subspecies of Aardvarks"
   ```

   This will run the workflow and output the results to the console.

   ```console
   Workflow Result:
   ['Here are five subspecies of Aardvarks:\n\n1. Orycteropus afer afer (Southern aardvark)\n2. O. a. adametzi  Grote, 1921 (Western aardvark)\n3. O. a. aethiopicus  Sundevall, 1843\n4. O. a. angolensis  Zukowsky & Haltenorth, 1957\n5. O. a. erikssoni  Lönnberg, 1906']
   ```

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AIQToolkit/issues) if you have any feedback or feature requests.

## Acknowledgements

We would like to thank the following open source projects that made AIQ Toolkit possible:

- [CrewAI](https://github.com/crewAIInc/crewAI)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Llama-Index](https://github.com/run-llama/llama_index)
- [Mem0ai](https://github.com/mem0ai/mem0)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [uv](https://github.com/astral-sh/uv)
