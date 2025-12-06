import pytest
import asyncio
import random
import os
from comfy_proxy.comfy import Comfy, SingleComfy
from comfy_proxy.workflows.flux2 import (
    Flux2Model,
    Flux2WorkflowParams,
    Flux2Workflow,
    Flux2EditModel,
    Flux2EditWorkflowParams,
    Flux2EditWorkflow
)
from comfy_proxy.workflow import Sizes


@pytest.mark.asyncio
async def test_flux2_workflow_structure() -> None:
    """Test that Flux2Workflow generates valid workflow structure (no ComfyUI required)."""
    # Create the default model configuration
    model = Flux2Model.default()

    # Set up workflow parameters
    params = Flux2WorkflowParams(
        prompt="a beautiful sunset over mountains",
        model=model,
        size=Sizes.LANDSCAPE_16_9,
        seed=42,
        batch_size=1
    )

    # Create the workflow
    workflow = Flux2Workflow(params)

    # Generate the prompt dictionary
    prompt_dict = workflow.to_dict()

    # Verify the basic structure requirements
    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    # Verify key Flux2-specific nodes exist
    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "CLIPLoader" in node_types  # Flux2 uses single CLIPLoader, not DualCLIPLoader
    assert "EmptyFlux2LatentImage" in node_types
    assert "Flux2Scheduler" in node_types
    assert "FluxGuidance" in node_types
    assert "BasicGuider" in node_types
    assert "SamplerCustomAdvanced" in node_types

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_flux2_edit_workflow_structure() -> None:
    """Test that Flux2EditWorkflow generates valid workflow structure (no ComfyUI required)."""
    # Create the default model configuration
    model = Flux2EditModel.default()

    # Set up workflow parameters with 2 reference images
    params = Flux2EditWorkflowParams(
        prompt="a dog sitting in a field",
        model=model,
        reference_images=["ref1.png", "ref2.png"],  # Placeholders
        size=Sizes.PORTRAIT_3_4,
        seed=42,
        batch_size=1
    )

    # Create the workflow
    workflow = Flux2EditWorkflow(params)

    # Generate the prompt dictionary
    prompt_dict = workflow.to_dict()

    # Verify the basic structure requirements
    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    # Verify key nodes exist
    node_types = [node["class_type"] for node in prompt_dict.values()]
    assert "CLIPLoader" in node_types
    assert "EmptyFlux2LatentImage" in node_types
    assert "Flux2Scheduler" in node_types
    assert "FluxGuidance" in node_types
    assert "BasicGuider" in node_types
    assert "SamplerCustomAdvanced" in node_types

    # Verify reference image chain nodes exist (2 of each for 2 reference images)
    assert node_types.count("LoadImage") == 2
    assert node_types.count("ImageScaleToTotalPixels") == 2
    assert node_types.count("VAEEncode") == 2
    assert node_types.count("ReferenceLatent") == 2

    # Verify reference image node IDs are tracked
    assert len(workflow.reference_image_node_ids) == 2

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_flux2_edit_workflow_max_images() -> None:
    """Test that Flux2EditWorkflow supports up to 10 reference images."""
    model = Flux2EditModel.default()

    # Test with maximum 10 reference images
    params = Flux2EditWorkflowParams(
        prompt="test",
        model=model,
        reference_images=[f"ref{i}.png" for i in range(10)],
        size=Sizes.SQUARE_1K,
        seed=42
    )

    workflow = Flux2EditWorkflow(params)
    prompt_dict = workflow.to_dict()

    node_types = [node["class_type"] for node in prompt_dict.values()]
    assert node_types.count("ReferenceLatent") == 10
    assert len(workflow.reference_image_node_ids) == 10


@pytest.mark.asyncio
async def test_flux2_edit_workflow_validation() -> None:
    """Test that Flux2EditWorkflowParams validates reference images."""
    model = Flux2EditModel.default()

    # Should raise error with no reference images
    with pytest.raises(ValueError, match="At least 1 reference image"):
        Flux2EditWorkflowParams(
            prompt="test",
            model=model,
            reference_images=[],
            size=Sizes.SQUARE_1K,
            seed=42
        )

    # Should raise error with more than 10 reference images
    with pytest.raises(ValueError, match="Maximum 10 reference images"):
        Flux2EditWorkflowParams(
            prompt="test",
            model=model,
            reference_images=[f"ref{i}.png" for i in range(11)],
            size=Sizes.SQUARE_1K,
            seed=42
        )


@pytest.mark.asyncio
async def test_flux2_generate() -> None:
    """Test actual image generation with Flux2 (requires running ComfyUI)."""
    comfy = Comfy("deepmonster2:8190")

    try:
        model = Flux2Model.default()

        params = Flux2WorkflowParams(
            prompt="a beautiful mountain landscape at sunset, photorealistic",
            model=model,
            size=Sizes.LANDSCAPE_16_9,
            guidance=3.5,
            steps=20,
            seed=random.randint(0, 2**32)
        )

        workflow = Flux2Workflow(params)

        # Generate and save the first image
        async for image_data, workflow_dict in comfy.generate(workflow):
            with open("flux2_t2i_output.png", "wb") as f:
                f.write(image_data)
            assert workflow_dict is not None  # Verify workflow dict is returned
            break  # Only save the first image
    finally:
        await comfy.stop()


@pytest.mark.asyncio
async def test_flux2_edit_generate() -> None:
    """Test actual image editing with Flux2EditWorkflow (requires running ComfyUI)."""
    comfy = Comfy("deepmonster2:8190")

    try:
        model = Flux2EditModel.default()

        # Get the path to input.jpg in tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(test_dir, "input.jpg")

        # Use placeholder names that will be replaced during upload
        params = Flux2EditWorkflowParams(
            prompt="a golden retriever dog sitting in a sunny meadow",
            model=model,
            reference_images=["ref_image_1.png"],  # Placeholder
            size=Sizes.LANDSCAPE_16_9,
            guidance=3.5,
            steps=20,
            megapixels=1.0,
            seed=random.randint(0, 2**32)
        )

        workflow = Flux2EditWorkflow(params)

        # Upload image and update the specific LoadImage node
        from comfy_proxy.comfy import SingleComfy
        single_comfy = SingleComfy("deepmonster2:8190")

        uploaded_filename = await single_comfy.upload_image(input_image_path)
        workflow._update_image_reference("image", uploaded_filename, workflow.reference_image_node_ids[0])

        # Generate without image_uploads since we manually uploaded
        async for image_data, workflow_dict in comfy.generate(workflow):
            with open("flux2_edit_output.png", "wb") as f:
                f.write(image_data)
            break
    finally:
        await comfy.stop()


@pytest.mark.asyncio
async def test_flux2_edit_multiple_refs_generate() -> None:
    """Test Flux2 edit with multiple reference images (requires running ComfyUI)."""
    comfy = Comfy("deepmonster2:8190")

    try:
        model = Flux2EditModel.default()

        # Get the path to input.jpg in tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(test_dir, "input.jpg")

        # Use same image twice as both references (for testing)
        params = Flux2EditWorkflowParams(
            prompt="a cat and a dog playing together in a garden",
            model=model,
            reference_images=["ref1.png", "ref2.png"],  # Placeholders
            size=Sizes.LANDSCAPE_16_9,
            guidance=3.5,
            steps=20,
            megapixels=1.0,
            seed=random.randint(0, 2**32)
        )

        workflow = Flux2EditWorkflow(params)

        # Upload image and update each LoadImage node by its ID
        from comfy_proxy.comfy import SingleComfy
        single_comfy = SingleComfy("deepmonster2:8190")

        uploaded_filename = await single_comfy.upload_image(input_image_path)
        # Update both reference image nodes with the same uploaded file
        for node_id in workflow.reference_image_node_ids:
            workflow._update_image_reference("image", uploaded_filename, node_id)

        # Generate without image_uploads since we manually uploaded
        async for image_data, workflow_dict in comfy.generate(workflow):
            with open("flux2_edit_multi_ref_output.png", "wb") as f:
                f.write(image_data)
            break
    finally:
        await comfy.stop()
