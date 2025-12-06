"""SeedVR2 Video Upscale Workflow for ComfyUI.

This workflow uses SeedVR2 to upscale videos using AI-based upscaling.
Based on SeedVR2_HD_video_upscale.json workflow.
"""

from dataclasses import dataclass
from typing import Optional
import random
from ..workflow import ComfyWorkflow


@dataclass
class SeedVR2VideoUpscaleModel:
    """Configuration for SeedVR2 video upscale model components"""
    dit_name: str = "seedvr2_ema_3b_fp16.safetensors"
    vae_name: str = "ema_vae_fp16.safetensors"
    dit_device: str = "cuda:0"
    vae_device: str = "cuda:0"
    attention_mode: str = "sdpa"
    # DiT offload settings
    dit_blocks_to_swap: int = 32
    dit_swap_io_components: bool = False
    dit_offload_device: str = "cpu"
    dit_cache_model: bool = False
    # VAE settings
    vae_encode_tiled: bool = True
    vae_encode_tile_size: int = 1024
    vae_encode_tile_overlap: int = 128
    vae_decode_tiled: bool = True
    vae_decode_tile_size: int = 768
    vae_decode_tile_overlap: int = 128
    vae_offload_device: str = "cpu"
    vae_cache_model: bool = False

    @classmethod
    def default(cls) -> 'SeedVR2VideoUpscaleModel':
        """Returns the default SeedVR2 video upscale model configuration"""
        return cls()


@dataclass
class SeedVR2VideoUpscaleWorkflowParams:
    """Parameters for configuring a SeedVR2 video upscale workflow"""
    input_video: str  # Path/filename of video to upscale
    model: SeedVR2VideoUpscaleModel = None
    resolution: int = 1080  # Target resolution (short edge)
    max_resolution: int = 0  # 0 = no limit
    batch_size: int = 33  # Higher for video processing
    uniform_batch_size: bool = True
    color_correction: str = "lab"  # "lab", "none", or other options
    seed: Optional[int] = None
    # Video-specific parameters
    temporal_overlap: int = 3  # Frames to overlap between batches for smooth transitions
    prepend_frames: int = 0
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    offload_device: str = "cpu"
    enable_debug: bool = False
    # Output settings
    output_prefix: str = "video/ComfyUI"
    output_format: str = "auto"
    output_codec: str = "auto"

    def __post_init__(self):
        if self.model is None:
            self.model = SeedVR2VideoUpscaleModel.default()
        if self.seed is None:
            self.seed = random.randint(0, 2**32)


class SeedVR2VideoUpscaleWorkflow(ComfyWorkflow):
    """A workflow for upscaling videos using SeedVR2."""

    def __init__(self, params: SeedVR2VideoUpscaleWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: SeedVR2VideoUpscaleWorkflowParams object containing all upscale parameters
        """
        super().__init__()
        self.params = params
        self.input_video_node_id = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load input video
        load_video = self.add_node("LoadVideo", {
            "file": self.params.input_video
        }, title="Load Video")
        self.input_video_node_id = load_video

        # Extract video components (frames, audio, fps)
        get_components = self.add_node("GetVideoComponents", {
            "video": [load_video, 0]
        }, title="Get Video Components")

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

        # Run upscaler on frames
        upscaler = self.add_node("SeedVR2VideoUpscaler", {
            "image": [get_components, 0],  # frames output
            "dit": [dit, 0],
            "vae": [vae, 0],
            "seed": self.params.seed,
            "resolution": self.params.resolution,
            "max_resolution": self.params.max_resolution,
            "batch_size": self.params.batch_size,
            "uniform_batch_size": self.params.uniform_batch_size,
            "color_correction": self.params.color_correction,
            "temporal_overlap": self.params.temporal_overlap,
            "prepend_frames": self.params.prepend_frames,
            "input_noise_scale": self.params.input_noise_scale,
            "latent_noise_scale": self.params.latent_noise_scale,
            "offload_device": self.params.offload_device,
            "enable_debug": self.params.enable_debug
        }, title="SeedVR2 Video Upscaler")

        # Create video from upscaled frames with original audio and fps
        create_video = self.add_node("CreateVideo", {
            "images": [upscaler, 0],
            "audio": [get_components, 1],  # audio output
            "fps": [get_components, 2]  # fps output
        }, title="Create Video")

        # Save the video
        self.add_node("SaveVideo", {
            "video": [create_video, 0],
            "filename_prefix": self.params.output_prefix,
            "format": self.params.output_format,
            "codec": self.params.output_codec
        }, node_id="save_video_node")
