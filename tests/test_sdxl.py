"""Tests for SDXL text-to-image workflow."""

import pytest
from comfy_proxy.workflows.sdxl import (
    SDXLModel,
    SDXLWorkflowParams,
    SDXLWorkflow
)
from comfy_proxy.workflow import Lora, Sizes


@pytest.mark.asyncio
async def test_sdxl_workflow_structure() -> None:
    """Test that the SDXL workflow generates a valid structure."""
    model = SDXLModel.default()

    params = SDXLWorkflowParams(
        prompt="a beautiful sunset over mountains",
        model=model,
        seed=42
    )

    workflow = SDXLWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Verify basic structure
    assert isinstance(prompt_dict, dict)
    assert "save_image_websocket_node" in prompt_dict

    # Check that required node types exist
    node_types = {node["class_type"] for node in prompt_dict.values()}
    assert "CheckpointLoaderSimple" in node_types
    assert "CLIPTextEncode" in node_types
    assert "EmptyLatentImage" in node_types
    assert "KSamplerAdvanced" in node_types
    assert "VAEDecode" in node_types
    assert "SaveImageWebsocket" in node_types


@pytest.mark.asyncio
async def test_sdxl_default_params() -> None:
    """Test that default parameters are applied correctly."""
    params = SDXLWorkflowParams(
        prompt="test prompt"
    )

    # Check defaults
    assert params.negative_prompt == "text, watermark"
    assert params.size == Sizes.SQUARE_1K
    assert params.cfg == 8.0
    assert params.steps == 20
    assert params.scheduler == "normal"
    assert params.sampler == "euler"
    assert params.batch_size == 1
    assert params.seed is not None  # Should be auto-generated
    assert params.model is not None
    assert params.model.checkpoint_name == "sd_xl_base_1.0.safetensors"


@pytest.mark.asyncio
async def test_sdxl_custom_model() -> None:
    """Test workflow with custom model checkpoint."""
    model = SDXLModel(
        checkpoint_name="juggernautXL_v8Rundiffusion.safetensors"
    )

    params = SDXLWorkflowParams(
        prompt="a fantasy landscape",
        model=model,
        size=Sizes.LANDSCAPE_16_9,
        cfg=7.0,
        steps=30,
        seed=12345
    )

    workflow = SDXLWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Find checkpoint node and verify model
    for node in prompt_dict.values():
        if node["class_type"] == "CheckpointLoaderSimple":
            assert node["inputs"]["ckpt_name"] == "juggernautXL_v8Rundiffusion.safetensors"
            break

    # Find sampler node and verify settings
    for node in prompt_dict.values():
        if node["class_type"] == "KSamplerAdvanced":
            assert node["inputs"]["cfg"] == 7.0
            assert node["inputs"]["steps"] == 30
            assert node["inputs"]["noise_seed"] == 12345
            break

    # Find latent node and verify size
    for node in prompt_dict.values():
        if node["class_type"] == "EmptyLatentImage":
            assert node["inputs"]["width"] == 1344
            assert node["inputs"]["height"] == 768
            break


@pytest.mark.asyncio
async def test_sdxl_with_loras() -> None:
    """Test workflow with LoRA models."""
    model = SDXLModel(
        checkpoint_name="sd_xl_base_1.0.safetensors",
        loras=[
            Lora(name="detail_xl.safetensors", weight=0.8),
            Lora(name="style_xl.safetensors", weight=0.5)
        ]
    )

    params = SDXLWorkflowParams(
        prompt="detailed portrait",
        model=model,
        seed=42
    )

    workflow = SDXLWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Check that LoRA nodes exist
    node_types = [node["class_type"] for node in prompt_dict.values()]
    lora_count = node_types.count("LoraLoader")
    assert lora_count == 2

    # Find LoRA nodes and verify weights
    lora_nodes = [node for node in prompt_dict.values() if node["class_type"] == "LoraLoader"]
    lora_names = [node["inputs"]["lora_name"] for node in lora_nodes]
    assert "detail_xl.safetensors" in lora_names
    assert "style_xl.safetensors" in lora_names


@pytest.mark.asyncio
async def test_sdxl_prompts() -> None:
    """Test that prompts are correctly encoded."""
    params = SDXLWorkflowParams(
        prompt="a majestic dragon",
        negative_prompt="blurry, ugly, deformed",
        seed=42
    )

    workflow = SDXLWorkflow(params)
    prompt_dict = workflow.to_dict()

    # Find CLIP encode nodes
    clip_nodes = [node for node in prompt_dict.values() if node["class_type"] == "CLIPTextEncode"]
    assert len(clip_nodes) == 2

    texts = [node["inputs"]["text"] for node in clip_nodes]
    assert "a majestic dragon" in texts
    assert "blurry, ugly, deformed" in texts


@pytest.mark.asyncio
async def test_sdxl_sampler_settings() -> None:
    """Test various sampler and scheduler settings."""
    params = SDXLWorkflowParams(
        prompt="test",
        sampler="dpmpp_2m",
        scheduler="karras",
        seed=42
    )

    workflow = SDXLWorkflow(params)
    prompt_dict = workflow.to_dict()

    for node in prompt_dict.values():
        if node["class_type"] == "KSamplerAdvanced":
            assert node["inputs"]["sampler_name"] == "dpmpp_2m"
            assert node["inputs"]["scheduler"] == "karras"
            assert node["inputs"]["add_noise"] == "enable"
            assert node["inputs"]["return_with_leftover_noise"] == "disable"
            assert node["inputs"]["start_at_step"] == 0
            assert node["inputs"]["end_at_step"] == params.steps
            break
