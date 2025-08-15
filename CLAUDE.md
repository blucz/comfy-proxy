# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyProxy is a Python client library for interacting with ComfyUI servers. It provides:
- Async workflow execution using websockets and HTTP
- Support for single or multiple ComfyUI instance management  
- Built-in workflow implementations: Flux and Qwen Image models with LoRA support
- Flexible address configuration (single, multiple, port ranges)

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies using Poetry
poetry install

# Run tests
poetry run pytest

# Run a specific test file
poetry run pytest tests/test_workflow.py

# Run a specific test
poetry run pytest tests/test_workflow.py::test_workflow
```

### Testing
The test suite includes unit tests for address parsing, workflow construction, and multi-instance management. Tests use pytest with asyncio support for testing async code.

Test files:
- `test_workflow.py` - Basic workflow construction
- `test_qwen_image.py` - Qwen Image workflow tests
- `test_address.py` - Address parsing tests
- `test_comfy_proxy.py` - SingleComfy client tests
- `test_multi_comfy.py` - Multi-instance management tests
- `test_generate_image.py` - End-to-end generation tests (requires running ComfyUI)

## Architecture

### Core Components

**SingleComfy** (comfy_proxy/comfy.py:23-155)
- Handles communication with a single ComfyUI instance
- Manages websocket connections for receiving generated images
- Queues prompts via HTTP POST to /prompt endpoint
- Tracks execution via websocket messages

**Comfy** (comfy_proxy/comfy.py:157-235)  
- Manages multiple SingleComfy instances for parallel generation
- Uses asyncio.Queue for work distribution
- Worker tasks handle generation requests with instance locking
- Supports various address formats via parse_addresses()

**ComfyWorkflow** (comfy_proxy/workflow.py:43-96)
- Base class for building ComfyUI node graphs
- Nodes are operations (model loading, encoding, sampling, etc.)
- Provides add_node() for graph construction
- to_dict() converts to ComfyUI JSON format

**FluxWorkflow** (comfy_proxy/workflows/flux.py:41-150)
- Concrete workflow implementation for Flux models
- Builds complete generation pipeline with ~15 nodes
- Supports LoRA chaining - each LoRA connects to previous in sequence
- Key nodes: UNETLoader, DualCLIPLoader, VAEDecode, SaveImageWebsocket

**QwenImageWorkflow** (comfy_proxy/workflows/qwen_image.py:44-136)
- Workflow implementation for Qwen Image models
- Uses ModelSamplingAuraFlow for Qwen-specific sampling
- Supports model-only LoRAs via LoraLoaderModelOnly
- Includes positive and negative prompt conditioning
- Key nodes: UNETLoader, CLIPLoader, KSampler, VAEDecode

### Key Design Patterns

1. **Node Graph Construction**: Workflows build directed graphs where nodes reference each other via [node_id, output_index] tuples
2. **Async Generators**: Image generation uses async generators to yield results as they become available
3. **Worker Pool**: Multiple ComfyUI instances are managed via worker tasks that pull from a shared queue
4. **Websocket Protocol**: Images are received via websocket with special handling for 'save_image_websocket_node'

### Address Parsing
The parse_addresses() function (comfy_proxy/address.py) handles multiple formats:
- Single: "127.0.0.1:8188"
- List: ["127.0.0.1:8188", "127.0.0.1:8189"]  
- Comma-separated: "127.0.0.1:8188,127.0.0.1:8189"
- Port range: "127.0.0.1:8188-8192"

### Communication Flow
1. Workflow converted to JSON dict via to_dict()
2. JSON posted to ComfyUI /prompt endpoint with client_id
3. Websocket connection established with same client_id
4. Execution messages received indicating node progress
5. Image data received as binary websocket messages
6. Images yielded back to caller as PNG bytes