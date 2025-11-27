import pytest
import asyncio
import random
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.z_image_turbo import ZImageTurboModel, ZImageTurboWorkflowParams, ZImageTurboWorkflow

@pytest.mark.asyncio
async def test_z_image_turbo_generate() -> None:
    """Test Z-Image Turbo generation against a running ComfyUI instance"""
    comfy = Comfy("deepmonster2:8191")

    try:
        model = ZImageTurboModel()

        params = ZImageTurboWorkflowParams(
            prompt="a beautiful sunset over mountains, masterpiece, highly detailed, photorealistic",
            model=model,
            seed=random.randint(0, 2**32),
        )

        workflow = ZImageTurboWorkflow(params)

        async for image_data in comfy.generate(workflow):
            # Save the generated image
            with open("z_image_turbo_test.png", "wb") as f:
                f.write(image_data)
            print(f"Generated image saved, size: {len(image_data)} bytes")
            break
    finally:
        await comfy.stop()
