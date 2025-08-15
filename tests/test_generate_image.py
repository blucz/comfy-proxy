import pytest
import asyncio
import random
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.flux import FluxModel, FluxWorkflowParams, FluxWorkflow
from comfy_proxy.workflow import Sizes

@pytest.mark.asyncio
async def test_generate_to_file() -> None:
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")
    
    # Create a model configuration
    model = FluxModel()
    
    # Set up workflow parameters
    params = FluxWorkflowParams(
        prompt="a beautiful sunset over mountains, masterpiece, highly detailed, photorealistic, in utah",
        model=model,
        size=Sizes.LANDSCAPE_16_9,
        seed=random.randint(0, 2**32),
        batch_size=1
    )
    
    # Create the workflow
    workflow = FluxWorkflow(params)

    # Generate and save the first image
    async for image_data in comfy.generate(workflow):
        with open("image.png", "wb") as f:
            f.write(image_data)
        break  # Only save the first image
        
    # Allow time for cleanup
    await asyncio.sleep(0.1)
