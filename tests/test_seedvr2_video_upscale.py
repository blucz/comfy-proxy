"""Tests for SeedVR2 Video Upscale Workflow."""

import pytest
from comfy_proxy.workflows.seedvr2_video_upscale import (
    SeedVR2VideoUpscaleModel,
    SeedVR2VideoUpscaleWorkflowParams,
    SeedVR2VideoUpscaleWorkflow
)


@pytest.mark.asyncio
async def test_seedvr2_video_upscale_workflow_structure() -> None:
    """Test that the video upscale workflow generates a valid structure (no ComfyUI required)."""
    # Create a model configuration
    model = SeedVR2VideoUpscaleModel.default()

    # Create workflow parameters
    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="test_video.mp4",
        model=model,
        resolution=1080,
        seed=42
    )

    # Create the workflow
    workflow = SeedVR2VideoUpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Verify basic structure
    assert isinstance(prompt_dict, dict)
    assert "save_video_node" in prompt_dict

    # Check that required node types exist
    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "LoadVideo" in node_types
    assert "GetVideoComponents" in node_types
    assert "SeedVR2LoadDiTModel" in node_types
    assert "SeedVR2LoadVAEModel" in node_types
    assert "SeedVR2VideoUpscaler" in node_types
    assert "CreateVideo" in node_types
    assert "SaveVideo" in node_types


@pytest.mark.asyncio
async def test_seedvr2_video_upscale_default_params() -> None:
    """Test that default parameters are applied correctly."""
    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="test.mp4"
    )

    # Check defaults
    assert params.resolution == 1080
    assert params.max_resolution == 0
    assert params.batch_size == 33
    assert params.uniform_batch_size is True
    assert params.color_correction == "lab"
    assert params.temporal_overlap == 3
    assert params.seed is not None  # Should be auto-generated
    assert params.model is not None
    assert params.model.dit_name == "seedvr2_ema_3b_fp16.safetensors"
    assert params.model.vae_name == "ema_vae_fp16.safetensors"
    # Check video-specific model defaults
    assert params.model.dit_blocks_to_swap == 32
    assert params.model.vae_encode_tiled is True
    assert params.model.vae_decode_tiled is True


@pytest.mark.asyncio
async def test_seedvr2_video_upscale_custom_resolution() -> None:
    """Test workflow with custom resolution settings."""
    model = SeedVR2VideoUpscaleModel.default()

    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="test.mp4",
        model=model,
        resolution=2160,  # 4K
        max_resolution=4096,
        seed=12345
    )

    workflow = SeedVR2VideoUpscaleWorkflow(params)
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
async def test_seedvr2_video_upscale_input_video_node_tracking() -> None:
    """Test that the input video node ID is tracked for updates."""
    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="placeholder.mp4",
        seed=42
    )

    workflow = SeedVR2VideoUpscaleWorkflow(params)

    # Verify the input_video_node_id is set
    assert workflow.input_video_node_id is not None

    # Verify we can update the video reference using _update_video_reference
    workflow._update_video_reference("file", "uploaded_video.mp4", workflow.input_video_node_id)

    prompt_dict = workflow.to_dict()
    load_video_node = prompt_dict[workflow.input_video_node_id]
    assert load_video_node["inputs"]["file"] == "uploaded_video.mp4"


@pytest.mark.asyncio
async def test_seedvr2_video_upscale_temporal_overlap() -> None:
    """Test that temporal overlap is properly set for video processing."""
    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="test.mp4",
        temporal_overlap=5,
        batch_size=64,
        seed=42
    )

    workflow = SeedVR2VideoUpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Find the upscaler node
    for node in prompt_dict.values():
        if node["class_type"] == "SeedVR2VideoUpscaler":
            assert node["inputs"]["temporal_overlap"] == 5
            assert node["inputs"]["batch_size"] == 64
            assert node["inputs"]["uniform_batch_size"] is True
            break


@pytest.mark.asyncio
async def test_seedvr2_video_upscale_video_output_settings() -> None:
    """Test that video output settings are properly configured."""
    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="test.mp4",
        output_prefix="upscaled/video",
        output_format="mp4",
        output_codec="h264",
        seed=42
    )

    workflow = SeedVR2VideoUpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Check SaveVideo node settings
    save_node = prompt_dict["save_video_node"]
    assert save_node["class_type"] == "SaveVideo"
    assert save_node["inputs"]["filename_prefix"] == "upscaled/video"
    assert save_node["inputs"]["format"] == "mp4"
    assert save_node["inputs"]["codec"] == "h264"


@pytest.mark.asyncio
async def test_seedvr2_video_upscale_audio_preservation() -> None:
    """Test that audio is connected from input to output."""
    params = SeedVR2VideoUpscaleWorkflowParams(
        input_video="test.mp4",
        seed=42
    )

    workflow = SeedVR2VideoUpscaleWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Find CreateVideo node and verify audio input is connected
    for node in prompt_dict.values():
        if node["class_type"] == "CreateVideo":
            # Audio should come from GetVideoComponents output 1
            audio_input = node["inputs"]["audio"]
            assert isinstance(audio_input, list)
            assert len(audio_input) == 2
            assert audio_input[1] == 1  # audio is output index 1
            break
