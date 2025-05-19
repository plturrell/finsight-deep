<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


<!-- This role is needed at the index to set the default backtick role -->
```{eval-rst}
.. role:: py(code)
   :language: python
   :class: highlight
```

![NVIDIA Agent Intelligence Toolkit](./_static/aiqtoolkit_banner.png "AIQ Toolkit banner image")

# NVIDIA Agent Intelligence Toolkit Overview

Agent Intelligence Toolkit (AIQ Toolkit) is a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework.


:::{note}
Agent Intelligence Toolkit was previously known as <!-- vale off -->AgentIQ<!-- vale on -->, however the API has not changed and is fully compatible with previous releases. Users should update their dependencies to depend on `aiqtoolkit` instead of `agentiq`. I transitional package named `agentiq` is available for backwards compatibility, but will be removed in the future.
:::

## Key Features

- [**Framework Agnostic:**](./quick-start/installing.md#framework-integrations) AIQ Toolkit works side-by-side and around existing agentic frameworks, such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), and [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), as well as customer enterprise frameworks and simple Python agents. This allows you to use your current technology stack without replatforming. AIQ Toolkit complements any existing agentic framework or memory tool you're using and isn't tied to any specific agentic framework, long-term memory, or data source.

- [**Reusability:**](./extend/sharing-components.md) Every agent, tool, and agentic workflow in this library exists as a function call that works together in complex software applications. The composability between these agents, tools, and workflows allows you to build once and reuse in different scenarios.

- [**Rapid Development:**](./tutorials/index.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs. This allows you and your development teams to move quickly if you're already developing with agents.

- [**Profiling:**](./workflows/profiler.md) Use the profiler to profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.

- [**Observability:**](./workflows/observe/index.md) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool, with examples using [Phoenix](./workflows/observe/observe-workflow-with-phoenix.md) and [W&B Weave](./workflows/observe/observe-workflow-with-weave.md).

- [**Evaluation System:**](./workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

- [**User Interface:**](./quick-start/launching-ui.md) Use the AIQ Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.

- [**Full MCP Support:**](./workflows/mcp/index.md) Compatible with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). You can use AIQ Toolkit as an [MCP client](./workflows/mcp/mcp-client.md) to connect to and use tools served by remote MCP servers. You can also use AIQ Toolkit as an [MCP server](./workflows/mcp/mcp-server.md) to publish tools via MCP.

- [**GPU Acceleration:**](./workflows/performance/gpu-optimization.md) CUDA tensor core optimization achieving **12.8x speedup** over CPU baseline with parallel processing for verification and consensus.

- [**Nash-Ethereum Consensus:**](./workflows/consensus/index.md) Game-theoretic blockchain consensus with smart contracts for decentralized trust and on-chain verification proofs.

- [**Real-time Verification:**](./workflows/verification/index.md) W3C PROV standard compliance with multi-method confidence scoring and source attribution tracking.

- [**Digital Human Interface:**](./workflows/digital-human/index.md) NVIDIA Audio2Face-3D integration providing natural conversation interface with real-time emotion rendering.

## What AIQ Toolkit Is

- A **lightweight, unifying library** that makes every agent, tool, and workflow you already have work together, just as simple function calls work together in complex software applications.
- An **end-to-end agentic profiler**, allowing you to track input/output tokens and timings at a granular level for every tool and agent, regardless of the amount of nesting.
- A way to accomplish **end-to-end evaluation and observability**. With the potential to wrap and hook into every function call, AIQ Toolkit can output observability data to your platform of choice. It also includes an end-to-end evaluation system, allowing you to consistently evaluate your complex, multi-framework workflows in the exact same way as you develop and deploy them.
- A **compliment to existing agentic frameworks** and memory tools, not a replacement.
- **100% opt in.** While we encourage users to wrap (decorate) every tool and agent to get the most out of the profiler, you have the freedom to integrate to whatever level you want - tool level, agent level, or entire workflow level. You have the freedom to start small and where you believe you’ll see the most value and expand from there.


## What AIQ Toolkit Is Not

- **An agentic framework.** AIQ Toolkit is designed to work alongside, not replace, your existing agentic frameworks — whether they are enterprise-grade systems or simple Python-based agents.
- **An attempt to solve agent-to-agent communication.** Agent communication is best handled over existing protocols, such as HTTP, gRPC, and sockets.
- **An observability platform.** While AIQ Toolkit is able to collect and transmit fine-grained telemetry to help with optimization and evaluation, it does not replace your preferred observability platform and data collection application.


## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AIQToolkit/issues) if you have any feedback or feature requests.

```{toctree}
:hidden:
:caption: About Agent Intelligence Toolkit
Overview <self>
Release Notes <./release-notes.md>
```

```{toctree}
:hidden:
:caption: Get Started

Quick Start Guide <./quick-start/index.md>
Tutorials <./tutorials/index.md>
```

```{toctree}
:hidden:
:caption: Manage Workflows

About Workflows <./workflows/about/index.md>
./workflows/run-workflows.md
Workflow Configuration <./workflows/workflow-configuration.md>
Functions <./workflows/functions/index.md>
./workflows/mcp/index.md
Evaluate Workflows <./workflows/evaluate.md>
Profiling Workflows <./workflows/profiler.md>
./workflows/using-local-llms.md
./workflows/observe/index.md
```

```{toctree}
:hidden:
:caption: Core Components

Verification System <./workflows/verification/index.md>
Nash-Ethereum Consensus <./workflows/consensus/index.md>
Research Framework <./workflows/research/index.md>
Digital Human <./workflows/digital-human/index.md>
Digital Human Technical Guide <./workflows/digital-human/technical-guide.md>
Digital Human Deployment <./workflows/digital-human/deployment.md>
```

```{toctree}
:hidden:
:caption: Integration

Integration Overview <./workflows/integration/index.md>
Verification-Consensus <./workflows/integration/verification-consensus.md>
Knowledge Graph <./workflows/integration/knowledge-graph.md>
Architecture <./workflows/integration/architecture.md>
```

```{toctree}
:hidden:
:caption: Store and Retrieve

Memory Module <./store-and-retrieve/memory.md>
./store-and-retrieve/retrievers.md
```

```{toctree}
:hidden:
:caption: Extend

Writing Custom Functions <./extend/functions.md>
Extending the AIQ Toolkit Using Plugins <./extend/plugins.md>
Sharing Components <./extend/sharing-components.md>
Adding a Custom Evaluator <./extend/custom-evaluator.md>
./extend/adding-a-retriever.md
./extend/memory.md
Adding an LLM Provider <./extend/adding-an-llm-provider.md>
```

```{toctree}
:hidden:
:caption: Reference

./api/index.rst
./reference/interactive-models.md
API Server Endpoints <./reference/api-server-endpoints.md>
./reference/websockets.md
Command Line Interface (CLI) <./reference/cli.md>
Evaluation <./reference/evaluate.md>
Evaluation Endpoints <./reference/evaluate-api.md>
Troubleshooting <./troubleshooting.md>
```

```{toctree}
:hidden:
:caption: Resources

Code of Conduct <./resources/code-of-conduct.md>
Contributing <./resources/contributing.md>
./resources/running-ci-locally.md
./support.md
./resources/licensing
```
