import pytest
import asyncio
import aiohttp
from comfy_proxy.comfy import SingleComfy
from comfy_proxy.workflows.flux import FluxModel, FluxWorkflowParams, FluxWorkflow
from comfy_proxy.workflow import Sizes

@pytest.mark.asyncio
async def test_workflow() -> None:
    async with aiohttp.ClientSession() as session:
        # Initialize the Comfy client
        comfy = SingleComfy("127.0.0.1:7821")
        
        # Create the default model configuration 
        model = FluxModel()
        
        # Set up workflow parameters
        params = FluxWorkflowParams(
            prompt="a beautiful sunset over mountains",
            model=model,
            size=Sizes.LANDSCAPE_16_9,
            seed=42,
            batch_size=1
        )
        
        # Create the workflow
        workflow = FluxWorkflow(params)
        
        # Generate the prompt dictionary
        prompt_dict = workflow.to_dict()
        
        # Verify the basic structure requirements
        assert isinstance(prompt_dict, dict)
        assert "save_image_websocket_node" in prompt_dict
        
        # Allow time for cleanup
        await asyncio.sleep(0.1)


