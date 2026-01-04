import pytest
import asyncio
import random
import os
from PIL import Image
import io
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.flux_fill import (
    FluxFillModel,
    FluxFillInpaintWorkflowParams,
    FluxFillInpaintWorkflow,
    FluxFillOutpaintWorkflowParams,
    FluxFillOutpaintWorkflow
)


def create_test_mask(input_path: str, output_path: str) -> None:
    """Create a simple mask image - white rectangle in center."""
    img = Image.open(input_path)
    width, height = img.size

    # Create black image (preserve area)
    mask = Image.new("L", (width, height), 0)

    # Draw white rectangle in center (inpaint area)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    margin_x = width // 4
    margin_y = height // 4
    draw.rectangle(
        [margin_x, margin_y, width - margin_x, height - margin_y],
        fill=255
    )

    mask.save(output_path)


@pytest.mark.asyncio
async def test_flux_fill_inpaint() -> None:
    """Test Flux Fill inpainting with actual ComfyUI generation."""
    comfy = Comfy("deepmonster2:8188")

    try:
        model = FluxFillModel.default()

        # Get path to input image
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(test_dir, "input.jpg")
        mask_path = os.path.join(test_dir, "test_mask.png")

        # Create a test mask
        create_test_mask(input_image_path, mask_path)

        params = FluxFillInpaintWorkflowParams(
            prompt="a red sports car",
            image="input.jpg",  # Placeholder
            mask="test_mask.png",  # Placeholder
            model=model,
            guidance=30.0,
            steps=20,
            seed=random.randint(0, 2**32)
        )

        workflow = FluxFillInpaintWorkflow(params)

        async for image_data, workflow_dict in comfy.generate(
            workflow,
            image_uploads={
                "input.jpg": input_image_path,
                "test_mask.png": mask_path
            }
        ):
            with open("flux_fill_inpaint_output.png", "wb") as f:
                f.write(image_data)
            break
    finally:
        await comfy.stop()
        # Clean up test mask
        if os.path.exists(mask_path):
            os.remove(mask_path)


@pytest.mark.asyncio
async def test_flux_fill_outpaint() -> None:
    """Test Flux Fill outpainting with actual ComfyUI generation."""
    comfy = Comfy("deepmonster2:8188")

    try:
        model = FluxFillModel.default()

        # Get path to input image
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(test_dir, "input.jpg")

        params = FluxFillOutpaintWorkflowParams(
            prompt="expansive landscape with mountains and forest",
            image="input.jpg",  # Placeholder
            model=model,
            left=256,
            right=256,
            top=0,
            bottom=0,
            feathering=40,
            guidance=30.0,
            steps=20,
            seed=random.randint(0, 2**32)
        )

        workflow = FluxFillOutpaintWorkflow(params)

        async for image_data, workflow_dict in comfy.generate(
            workflow,
            image_uploads={"input.jpg": input_image_path}
        ):
            with open("flux_fill_outpaint_output.png", "wb") as f:
                f.write(image_data)
            break
    finally:
        await comfy.stop()


# --- Structure tests (no ComfyUI required) ---

@pytest.mark.asyncio
async def test_flux_fill_inpaint_workflow_structure() -> None:
    """Test that FluxFillInpaintWorkflow generates valid workflow structure."""
    model = FluxFillModel.default()

    params = FluxFillInpaintWorkflowParams(
        prompt="a red sports car",
        image="input.png",
        mask="mask.png",
        model=model,
        seed=42
    )

    workflow = FluxFillInpaintWorkflow(params)
    prompt_dict = workflow.to_dict()

    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "DualCLIPLoader" in node_types
    assert "UNETLoader" in node_types
    assert "VAELoader" in node_types
    assert "InpaintModelConditioning" in node_types
    assert "DifferentialDiffusion" in node_types
    assert "KSampler" in node_types
    assert "FluxGuidance" in node_types


@pytest.mark.asyncio
async def test_flux_fill_outpaint_workflow_structure() -> None:
    """Test that FluxFillOutpaintWorkflow generates valid workflow structure."""
    model = FluxFillModel.default()

    params = FluxFillOutpaintWorkflowParams(
        prompt="expansive landscape",
        image="input.png",
        model=model,
        left=256,
        right=256,
        seed=42
    )

    workflow = FluxFillOutpaintWorkflow(params)
    prompt_dict = workflow.to_dict()

    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "ImagePadForOutpaint" in node_types
    assert "InpaintModelConditioning" in node_types
    assert "DifferentialDiffusion" in node_types
    assert "KSampler" in node_types


@pytest.mark.asyncio
async def test_flux_fill_model_defaults() -> None:
    """Test that FluxFillModel has correct default values."""
    model = FluxFillModel.default()

    # Note: clip_l first, then t5xxl (matching official workflow)
    assert model.clip_name1 == "clip_l.safetensors"
    assert model.clip_name2 == "t5xxl_fp16.safetensors"
    assert model.vae_name == "ae.safetensors"
    assert model.unet_name == "flux1-fill-dev.safetensors"
    assert model.weight_dtype == "default"
