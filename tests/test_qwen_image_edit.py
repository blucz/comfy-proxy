import pytest
import asyncio
import random
import os
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.qwen_image_edit_2509 import (
    QwenImageEditPlusModel,
    QwenImageEditPlusLightningModel,
    QwenImageEditPlusWorkflowParams,
    QwenImageEditPlusLightningWorkflowParams,
    QwenImageEditPlusWorkflow,
    QwenImageEditPlusLightningWorkflow
)
from comfy_proxy.workflow import Sizes

@pytest.mark.asyncio
async def test_qwen_image_edit_plus() -> None:
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")

    # Create a model configuration
    model = QwenImageEditPlusModel.default()

    # Set up workflow parameters (image will be uploaded)
    params = QwenImageEditPlusWorkflowParams(
        prompt="obtain the side view",
        image="input.jpg",  # Placeholder, will be replaced by uploaded filename
        model=model,
        megapixels=1.0,
        seed=random.randint(0, 2**64)
    )

    # Create the workflow
    workflow = QwenImageEditPlusWorkflow(params)

    # Get the path to input.jpg in tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(test_dir, "input.jpg")

    # Generate and save the first image with image upload
    async for image_data in comfy.generate(workflow, image_uploads={"image": input_image_path}):
        with open("qwen_edit_output.png", "wb") as f:
            f.write(image_data)
        break  # Only save the first image

    # Allow time for cleanup
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_qwen_image_edit_plus_lightning() -> None:
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")

    # Create a Plus Lightning model configuration (automatically includes Lightning LoRA)
    model = QwenImageEditPlusLightningModel.default()

    # Set up workflow parameters (steps=8, cfg=1.0 by default, image will be uploaded)
    params = QwenImageEditPlusLightningWorkflowParams(
        prompt="obtain the side view",
        image="input.jpg",  # Placeholder, will be replaced by uploaded filename
        model=model,
        megapixels=1.0,
        seed=random.randint(0, 2**64)
    )

    # Create the workflow
    workflow = QwenImageEditPlusLightningWorkflow(params)

    # Get the path to input.jpg in tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(test_dir, "input.jpg")

    # Generate and save the first image with image upload
    async for image_data in comfy.generate(workflow, image_uploads={"image": input_image_path}):
        with open("qwen_edit_plus_lightning_output.png", "wb") as f:
            f.write(image_data)
        break  # Only save the first image

    # Allow time for cleanup
    await asyncio.sleep(0.1)
