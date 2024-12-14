# ComfyProxy

A Python client library for interacting with ComfyUI servers and managing image generation workflows.

## Features

- Single and multi-instance ComfyUI server management
- Async workflow execution
- Flexible address configuration (single, multiple, port ranges)
- Built-in workflow templates
- Customizable image sizes and parameters
- Websocket-based image generation and retrieval

## Installation

ComfyProxy requires Python 3.11 or later. Install using pip:

```bash
pip install comfy-proxy
```

## Quick Start

```python
import asyncio
from comfy_proxy.comfy import SingleComfy
from comfy_proxy.workflows.flux import FluxModel, FluxWorkflowParams, FluxWorkflow
from comfy_proxy.workflow import Sizes

async def generate_image():
    # Initialize a ComfyUI instance
    comfy = Comfy("127.0.0.1:7821")
    
    # Configure model and parameters
    model = FluxModel()
    params = FluxWorkflowParams(
        prompt="a beautiful sunset over mountains",
        model=model,
        size=Sizes.LANDSCAPE_16_9,
        seed=42,
        batch_size=1
    )
    
    # Create and execute workflow
    workflow = FluxWorkflow(params)
    async for image_data in comfy.generate(workflow):
        with open("generated_image.png", "wb") as f:
            f.write(image_data)
        break  # Save first image only

# Run the example
asyncio.run(generate_image())
```

## Multiple ComfyUI Instances

ComfyProxy can distribute work across multiple ComfyUI servers:

```python
from comfy_proxy.comfy import Comfy

# Initialize with multiple instances
comfy = Comfy("127.0.0.1:8188-8192")  # Port range
# or
comfy = Comfy(["127.0.0.1:8188", "127.0.0.1:8189"])  # List
# or
comfy = Comfy("127.0.0.1:8188,127.0.0.1:8189")  # Comma-separated
```

## Development

Requirements:
- Python 3.11+
- Poetry for dependency management

Setup:
```bash
# Clone repository
git clone https://github.com/yourusername/comfy-proxy.git
cd comfy-proxy

# Install dependencies
poetry install

# Run tests
poetry run pytest
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
