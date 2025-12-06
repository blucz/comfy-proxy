"""Tests for SeedVR2 Image Upscale Workflow."""

import pytest
import random
import os
from comfy_proxy.workflows.seedvr2_upscale import (
    SeedVR2UpscaleModel,
    SeedVR2UpscaleWorkflowParams,
    SeedVR2UpscaleWorkflow
)


@pytest.mark.asyncio
async def test_seedvr2_upscale_workflow_structure() -> None:
    """Test that the upscale workflow generates a valid structure (no ComfyUI required)."""
    # Create a model configuration
    model = SeedVR2UpscaleModel.default()

    # Create workflow parameters
    params = SeedVR2UpscaleWorkflowParams(
        input_image="test_image.png",
        model=model,
        resolution=1080,
        seed=42
    )

    # Create the workflow
    workflow = SeedVR2UpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Verify basic structure
    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    # Check that required node types exist
    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "LoadImage" in node_types
    assert "SeedVR2LoadDiTModel" in node_types
    assert "SeedVR2LoadVAEModel" in node_types
    assert "SeedVR2VideoUpscaler" in node_types
    assert "SaveImageWebsocket" in node_types


@pytest.mark.asyncio
async def test_seedvr2_upscale_default_params() -> None:
    """Test that default parameters are applied correctly."""
    params = SeedVR2UpscaleWorkflowParams(
        input_image="test.png"
    )

    # Check defaults
    assert params.resolution == 1080
    assert params.max_resolution == 0
    assert params.batch_size == 1
    assert params.color_correction == "lab"
    assert params.seed is not None  # Should be auto-generated
    assert params.model is not None
    assert params.model.dit_name == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    assert params.model.vae_name == "ema_vae_fp16.safetensors"


@pytest.mark.asyncio
async def test_seedvr2_upscale_custom_resolution() -> None:
    """Test workflow with custom resolution settings."""
    model = SeedVR2UpscaleModel.default()

    params = SeedVR2UpscaleWorkflowParams(
        input_image="test.png",
        model=model,
        resolution=2160,  # 4K
        max_resolution=4096,
        seed=12345
    )

    workflow = SeedVR2UpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Find the upscaler node and check its inputs
    upscaler_node = None
    for node in prompt_dict.values():
        if node["class_type"] == "SeedVR2VideoUpscaler":
            upscaler_node = node
            break

    assert upscaler_node is not None
    assert upscaler_node["inputs"]["resolution"] == 2160
    assert upscaler_node["inputs"]["max_resolution"] == 4096
    assert upscaler_node["inputs"]["seed"] == 12345


@pytest.mark.asyncio
async def test_seedvr2_upscale_input_image_node_tracking() -> None:
    """Test that the input image node ID is tracked for updates."""
    params = SeedVR2UpscaleWorkflowParams(
        input_image="placeholder.png",
        seed=42
    )

    workflow = SeedVR2UpscaleWorkflow(params)

    # Verify the input_image_node_id is set
    assert workflow.input_image_node_id is not None

    # Verify we can update the image reference
    workflow._update_image_reference("image", "uploaded_image.png", workflow.input_image_node_id)

    prompt_dict = workflow.to_dict()
    load_image_node = prompt_dict[workflow.input_image_node_id]
    assert load_image_node["inputs"]["image"] == "uploaded_image.png"


@pytest.mark.asyncio
async def test_seedvr2_upscale_color_correction_options() -> None:
    """Test that color correction option is applied."""
    params = SeedVR2UpscaleWorkflowParams(
        input_image="test.png",
        color_correction="none",
        seed=42
    )

    workflow = SeedVR2UpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Find the upscaler node
    for node in prompt_dict.values():
        if node["class_type"] == "SeedVR2VideoUpscaler":
            assert node["inputs"]["color_correction"] == "none"
            break


# End-to-end test (commented out - requires running ComfyUI with SeedVR2)
"""
@pytest.mark.asyncio
async def test_seedvr2_upscale_e2e() -> None:
    '''End-to-end test that actually runs the upscale workflow.'''
    from comfy_proxy.comfy import Comfy

    comfy = Comfy("127.0.0.1:8188")

    try:
        model = SeedVR2UpscaleModel.default()

        params = SeedVR2UpscaleWorkflowParams(
            input_image="test.png",  # Placeholder, will be replaced
            model=model,
            resolution=1080,
            seed=random.randint(0, 2**32)
        )

        workflow = SeedVR2UpscaleWorkflow(params)

        # Get the path to a test image
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(test_dir, "test_image.jpg")

        # Generate with image upload
        async for image_data, workflow_dict in comfy.generate(workflow, image_uploads={"image": test_image_path}):
            with open("seedvr2_upscale_output.png", "wb") as f:
                f.write(image_data)
            break  # Only save the first image
    finally:
        await comfy.stop()
"""
