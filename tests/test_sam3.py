"""Tests for SAM3 Segmentation Workflow."""

import pytest
from comfy_proxy.workflows.sam3 import (
    SAM3Model,
    SAM3WorkflowParams,
    SAM3Workflow,
    SAM3Detection,
    SAM3Result
)


@pytest.mark.asyncio
async def test_sam3_workflow_structure() -> None:
    """Test that the SAM3 workflow generates a valid structure (no ComfyUI required)."""
    # Create a model configuration
    model = SAM3Model.default()

    # Create workflow parameters
    params = SAM3WorkflowParams(
        input_image="test_image.png",
        prompt="animal",
        model=model,
        confidence_threshold=0.2
    )

    # Create the workflow
    workflow = SAM3Workflow(params)
    prompt_dict = workflow.to_dict()

    # Verify basic structure
    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    # Check that required node types exist
    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "LoadImage" in node_types
    assert "LoadSAM3Model" in node_types
    assert "SAM3Grounding" in node_types
    assert "SaveImageWebsocket" in node_types
    assert "MaskToImage" in node_types
    assert "SaveImage" in node_types


@pytest.mark.asyncio
async def test_sam3_default_params() -> None:
    """Test that default parameters are applied correctly."""
    params = SAM3WorkflowParams(
        input_image="test.png",
        prompt="person"
    )

    # Check defaults
    assert params.confidence_threshold == 0.2
    assert params.max_detections == -1  # Unlimited
    assert params.multimask_output is False
    assert params.model is not None
    assert params.model.model_path == "models/sam3/sam3.pt"
    assert params.output_prefix == "sam3_"


@pytest.mark.asyncio
async def test_sam3_custom_params() -> None:
    """Test workflow with custom parameters."""
    model = SAM3Model(model_path="custom/sam3.pt")

    params = SAM3WorkflowParams(
        input_image="test.png",
        prompt="car",
        model=model,
        confidence_threshold=0.5,
        max_detections=10,
        multimask_output=True
    )

    workflow = SAM3Workflow(params)
    prompt_dict = workflow.to_dict()

    # Find the SAM3Grounding node and check its inputs
    grounding_node = None
    for node in prompt_dict.values():
        if node["class_type"] == "SAM3Grounding":
            grounding_node = node
            break

    assert grounding_node is not None
    assert grounding_node["inputs"]["text_prompt"] == "car"
    assert grounding_node["inputs"]["confidence_threshold"] == 0.5
    assert grounding_node["inputs"]["max_detections"] == 10


@pytest.mark.asyncio
async def test_sam3_input_image_node_tracking() -> None:
    """Test that the input image node ID is tracked for updates."""
    params = SAM3WorkflowParams(
        input_image="placeholder.png",
        prompt="object"
    )

    workflow = SAM3Workflow(params)

    # Verify the input_image_node_id is set
    assert workflow.input_image_node_id is not None

    # Verify we can update the image reference
    workflow._update_image_reference("image", "uploaded_image.png", workflow.input_image_node_id)

    prompt_dict = workflow.to_dict()
    load_image_node = prompt_dict[workflow.input_image_node_id]
    assert load_image_node["inputs"]["image"] == "uploaded_image.png"


@pytest.mark.asyncio
async def test_sam3_model_config() -> None:
    """Test SAM3Model configuration options."""
    # Test default model
    default_model = SAM3Model.default()
    assert default_model.model_path == "models/sam3/sam3.pt"
    assert default_model.config_path == ""

    # Test custom model
    custom_model = SAM3Model(
        model_path="custom/path/sam3_large.pt",
        config_path="custom/config.yaml"
    )
    assert custom_model.model_path == "custom/path/sam3_large.pt"
    assert custom_model.config_path == "custom/config.yaml"


@pytest.mark.asyncio
async def test_sam3_grounding_node_outputs() -> None:
    """Test that the workflow correctly connects to SAM3Grounding outputs."""
    params = SAM3WorkflowParams(
        input_image="test.png",
        prompt="person"
    )

    workflow = SAM3Workflow(params)
    prompt_dict = workflow.to_dict()

    # Find SaveImageWebsocket node
    save_node = prompt_dict["save_image_websocket_node"]

    # It should reference the visualization output (index 1) of SAM3Grounding
    images_input = save_node["inputs"]["images"]
    assert isinstance(images_input, list)
    assert images_input[1] == 1  # Output index 1 = visualization

    # Verify the referenced node is SAM3Grounding
    grounding_node_id = images_input[0]
    grounding_node = prompt_dict[grounding_node_id]
    assert grounding_node["class_type"] == "SAM3Grounding"


@pytest.mark.asyncio
async def test_sam3_mask_save_nodes() -> None:
    """Test that masks are saved via SaveImage node."""
    params = SAM3WorkflowParams(
        input_image="test.png",
        prompt="cat",
        output_prefix="test_prefix_"
    )

    workflow = SAM3Workflow(params)
    prompt_dict = workflow.to_dict()

    # Find the SaveImage node for masks
    save_image_nodes = [
        node for node in prompt_dict.values()
        if node["class_type"] == "SaveImage"
    ]
    assert len(save_image_nodes) == 1
    save_node = save_image_nodes[0]

    # Check filename prefix is set correctly
    assert save_node["inputs"]["filename_prefix"] == "test_prefix_mask"

    # Find MaskToImage node
    mask_to_image_nodes = [
        node for node in prompt_dict.values()
        if node["class_type"] == "MaskToImage"
    ]
    assert len(mask_to_image_nodes) == 1

    # Verify MaskToImage connects to SAM3Grounding output 0 (masks)
    mask_node = mask_to_image_nodes[0]
    mask_input = mask_node["inputs"]["mask"]
    assert isinstance(mask_input, list)
    assert mask_input[1] == 0  # Output index 0 = masks


@pytest.mark.asyncio
async def test_sam3_result_dataclasses() -> None:
    """Test SAM3Detection and SAM3Result dataclasses."""
    # Test SAM3Detection
    detection = SAM3Detection(
        bbox={"x": 10, "y": 20, "width": 100, "height": 200},
        score=0.95,
        mask_filename="sam3_mask_00001.png"
    )
    assert detection.bbox["x"] == 10
    assert detection.score == 0.95
    assert detection.mask_filename == "sam3_mask_00001.png"

    # Test SAM3Result
    result = SAM3Result(
        detections=[detection],
        visualization=b"test_image_data",
        original_width=1024,
        original_height=768
    )
    assert len(result.detections) == 1
    assert result.visualization == b"test_image_data"
    assert result.original_width == 1024
    assert result.original_height == 768

    # Test empty result
    empty_result = SAM3Result()
    assert len(empty_result.detections) == 0
    assert empty_result.visualization is None
