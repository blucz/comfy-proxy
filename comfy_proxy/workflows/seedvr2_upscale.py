"""SeedVR2 Image Upscale Workflow for ComfyUI.

This workflow uses SeedVR2 to upscale images using AI-based upscaling.
Based on SeedVR2_simple_image_upscale.json workflow.
"""

from dataclasses import dataclass
from typing import Optional
import random
from ..workflow import ComfyWorkflow


@dataclass
class SeedVR2UpscaleModel:
    """Configuration for SeedVR2 upscale model components"""
    dit_name: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    vae_name: str = "ema_vae_fp16.safetensors"
    dit_device: str = "cuda:0"
    vae_device: str = "cuda:0"
    attention_mode: str = "sdpa"
    # DiT offload settings
    dit_blocks_to_swap: int = 0
    dit_swap_io_components: bool = False
    dit_offload_device: str = "none"
    dit_cache_model: bool = False
    # VAE settings
    vae_encode_tiled: bool = False
    vae_encode_tile_size: int = 1024
    vae_encode_tile_overlap: int = 128
    vae_decode_tiled: bool = False
    vae_decode_tile_size: int = 1024
    vae_decode_tile_overlap: int = 128
    vae_offload_device: str = "none"
    vae_cache_model: bool = False

    @classmethod
    def default(cls) -> 'SeedVR2UpscaleModel':
        """Returns the default SeedVR2 upscale model configuration"""
        return cls()


@dataclass
class SeedVR2UpscaleWorkflowParams:
    """Parameters for configuring a SeedVR2 upscale workflow"""
    input_image: str  # Path/filename of image to upscale
    model: SeedVR2UpscaleModel = None
    resolution: int = 1080  # Target resolution (short edge)
    max_resolution: int = 0  # 0 = no limit
    batch_size: int = 1
    color_correction: str = "lab"  # "lab", "none", or other options
    seed: Optional[int] = None
    # Advanced parameters (usually left at defaults)
    temporal_overlap: int = 0
    prepend_frames: int = 0
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    offload_device: str = "cpu"
    enable_debug: bool = False

    def __post_init__(self):
        if self.model is None:
            self.model = SeedVR2UpscaleModel.default()
        if self.seed is None:
            self.seed = random.randint(0, 2**32)


class SeedVR2UpscaleWorkflow(ComfyWorkflow):
    """A workflow for upscaling images using SeedVR2."""

    def __init__(self, params: SeedVR2UpscaleWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: SeedVR2UpscaleWorkflowParams object containing all upscale parameters
        """
        super().__init__()
        self.params = params
        self.input_image_node_id = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.input_image
        }, title="Load Image")
        self.input_image_node_id = load_image

        # Load DiT model
        dit = self.add_node("SeedVR2LoadDiTModel", {
            "model": self.params.model.dit_name,
            "device": self.params.model.dit_device,
            "blocks_to_swap": self.params.model.dit_blocks_to_swap,
            "swap_io_components": self.params.model.dit_swap_io_components,
            "offload_device": self.params.model.dit_offload_device,
            "cache_model": self.params.model.dit_cache_model,
            "attention_mode": self.params.model.attention_mode
        }, title="Load SeedVR2 DiT Model")

        # Load VAE model
        vae = self.add_node("SeedVR2LoadVAEModel", {
            "model": self.params.model.vae_name,
            "device": self.params.model.vae_device,
            "encode_tiled": self.params.model.vae_encode_tiled,
            "encode_tile_size": self.params.model.vae_encode_tile_size,
            "encode_tile_overlap": self.params.model.vae_encode_tile_overlap,
            "decode_tiled": self.params.model.vae_decode_tiled,
            "decode_tile_size": self.params.model.vae_decode_tile_size,
            "decode_tile_overlap": self.params.model.vae_decode_tile_overlap,
            "tile_debug": "false",
            "offload_device": self.params.model.vae_offload_device,
            "cache_model": self.params.model.vae_cache_model
        }, title="Load SeedVR2 VAE Model")

        # Run upscaler
        upscaler = self.add_node("SeedVR2VideoUpscaler", {
            "image": [load_image, 0],
            "dit": [dit, 0],
            "vae": [vae, 0],
            "seed": self.params.seed,
            "resolution": self.params.resolution,
            "max_resolution": self.params.max_resolution,
            "batch_size": self.params.batch_size,
            "uniform_batch_size": False,
            "color_correction": self.params.color_correction,
            "temporal_overlap": self.params.temporal_overlap,
            "prepend_frames": self.params.prepend_frames,
            "input_noise_scale": self.params.input_noise_scale,
            "latent_noise_scale": self.params.latent_noise_scale,
            "offload_device": self.params.offload_device,
            "enable_debug": self.params.enable_debug
        }, title="SeedVR2 Video Upscaler")

        # Save via websocket for streaming output
        self.add_node("SaveImageWebsocket", {
            "images": [upscaler, 0]
        }, node_id="save_image_websocket_node")
