import pytest
import asyncio
import aiohttp
import random
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.flux import FluxModel, FluxWorkflowParams, FluxWorkflow
from comfy_proxy.workflow import Sizes

@pytest.mark.asyncio
async def test_parallel_generation() -> None:
    # Create test prompts
    prompts = [
        "a beautiful sunset over mountains",
        "a serene lake surrounded by pine trees",
        "a bustling cityscape at night",
        "a peaceful garden with blooming flowers"
    ]
    
    # Create workflows for each prompt
    model = FluxModel()
    workflows = []
    for i, prompt in enumerate(prompts):
        params = FluxWorkflowParams(
            prompt=prompt,
            model=model,
            size=Sizes.LANDSCAPE_16_9,
            seed=random.randint(0, 2**32),
            batch_size=1
        )
        workflows.append(FluxWorkflow(params))
    
    # Initialize MultiComfy with port range format
    comfy = Comfy("127.0.0.1:7821-7824")
    
    # Create tasks for parallel generation
    async def generate_and_save(workflow: FluxWorkflow, index: int) -> bool:
        async for image_data in comfy.generate(workflow):
            with open(f"test_image_{index}.png", "wb") as f:
                f.write(image_data)
            return True  # Successfully generated
        return False
    
    # Run all generations in parallel
    tasks = [generate_and_save(workflow, i) for i, workflow in enumerate(workflows)]
    results = await asyncio.gather(*tasks)
    
    # Verify all generations completed successfully
    assert all(results), "Not all image generations completed successfully"
    
    # Cleanup
    await comfy.stop()
