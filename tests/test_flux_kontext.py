import pytest
import asyncio
import random
import os
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.flux_kontext import (
    FluxKontextModel,
    FluxKontextWorkflowParams,
    FluxKontextWorkflow
)
from comfy_proxy.workflow import Sizes


@pytest.mark.asyncio
async def test_flux_kontext_edit() -> None:
    """Test Flux Kontext image editing with actual ComfyUI generation."""
    comfy = Comfy("deepmonster2:8188")

    try:
        model = FluxKontextModel.default()

        # Get path to input image
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(test_dir, "input.jpg")

        params = FluxKontextWorkflowParams(
            prompt="change the background to a beach sunset",
            reference_images=["input.jpg"],  # Placeholder
            model=model,
            size=Sizes.SQUARE_1K,
            guidance=2.5,  # Kontext uses lower guidance
            steps=20,
            seed=random.randint(0, 2**32)
        )

        workflow = FluxKontextWorkflow(params)

        async for image_data, workflow_dict in comfy.generate(
            workflow,
            image_uploads={"input.jpg": input_image_path}
        ):
            with open("flux_kontext_output.png", "wb") as f:
                f.write(image_data)
            break
    finally:
        await comfy.stop()


@pytest.mark.skip(reason="Multi-ref image upload needs workflow-level support")
@pytest.mark.asyncio
async def test_flux_kontext_multi_ref() -> None:
    """Test Flux Kontext with multiple reference images."""
    comfy = Comfy("deepmonster2:8188")

    try:
        model = FluxKontextModel.default()

        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(test_dir, "input.jpg")

        # Use same image twice as a test
        params = FluxKontextWorkflowParams(
            prompt="combine elements from both images",
            reference_images=["ref1.jpg", "ref2.jpg"],
            model=model,
            size=Sizes.SQUARE_1K,
            guidance=3.5,
            steps=20,
            seed=random.randint(0, 2**32)
        )

        workflow = FluxKontextWorkflow(params)

        async for image_data, workflow_dict in comfy.generate(
            workflow,
            image_uploads={
                "ref1.jpg": input_image_path,
                "ref2.jpg": input_image_path
            }
        ):
            with open("flux_kontext_multi_output.png", "wb") as f:
                f.write(image_data)
            break
    finally:
        await comfy.stop()


# --- Structure tests (no ComfyUI required) ---

@pytest.mark.asyncio
async def test_flux_kontext_workflow_structure() -> None:
    """Test that FluxKontextWorkflow generates valid workflow structure."""
    model = FluxKontextModel.default()

    params = FluxKontextWorkflowParams(
        prompt="change the color to blue",
        reference_images=["ref1.png"],
        model=model,
        size=Sizes.SQUARE_1K,
        seed=42
    )

    workflow = FluxKontextWorkflow(params)
    prompt_dict = workflow.to_dict()

    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "DualCLIPLoader" in node_types
    assert "UNETLoader" in node_types
    assert "VAELoader" in node_types
    assert "ReferenceLatent" in node_types
    assert "FluxKontextImageScale" in node_types
    assert "ConditioningZeroOut" in node_types
    assert "KSampler" in node_types
    assert "FluxGuidance" in node_types


@pytest.mark.asyncio
async def test_flux_kontext_workflow_multiple_refs() -> None:
    """Test that FluxKontextWorkflow supports multiple reference images with ImageStitch."""
    model = FluxKontextModel.default()

    params = FluxKontextWorkflowParams(
        prompt="combine these elements",
        reference_images=["ref1.png", "ref2.png", "ref3.png"],
        model=model,
        size=Sizes.LANDSCAPE_16_9,
        seed=42
    )

    workflow = FluxKontextWorkflow(params)
    prompt_dict = workflow.to_dict()

    node_type_list = [node["class_type"] for node in prompt_dict.values()]
    assert node_type_list.count("LoadImage") == 3
    # Multiple refs use ImageStitch, single ReferenceLatent
    assert node_type_list.count("ImageStitch") == 2  # 3 images = 2 stitches
    assert node_type_list.count("ReferenceLatent") == 1
    assert len(workflow.reference_image_node_ids) == 3


@pytest.mark.asyncio
async def test_flux_kontext_workflow_validation() -> None:
    """Test that FluxKontextWorkflowParams validates reference images."""
    model = FluxKontextModel.default()

    with pytest.raises(ValueError, match="At least 1 reference image"):
        FluxKontextWorkflowParams(
            prompt="test",
            reference_images=[],
            model=model,
            size=Sizes.SQUARE_1K,
            seed=42
        )

    with pytest.raises(ValueError, match="Maximum 10 reference images"):
        FluxKontextWorkflowParams(
            prompt="test",
            reference_images=[f"ref{i}.png" for i in range(11)],
            model=model,
            size=Sizes.SQUARE_1K,
            seed=42
        )


@pytest.mark.asyncio
async def test_flux_kontext_model_defaults() -> None:
    """Test that FluxKontextModel has correct default values."""
    model = FluxKontextModel.default()

    # Note: clip_l first, then t5xxl (matching official workflow)
    assert model.clip_name1 == "clip_l.safetensors"
    assert model.clip_name2 == "t5xxl_fp16.safetensors"
    assert model.vae_name == "ae.safetensors"
    assert model.unet_name == "flux1-dev-kontext_fp8_scaled.safetensors"
    assert model.weight_dtype == "default"
    assert model.loras == []
