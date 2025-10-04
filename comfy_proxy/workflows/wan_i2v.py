from dataclasses import dataclass
from typing import Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size


@dataclass
class WanI2VModel:
    """Configuration for a Wan 2.2 I2V model and its components"""
    unet_high_noise_name: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    unet_low_noise_name: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
    clip_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_name: str = "wan_2.1_vae.safetensors"
    weight_dtype: str = "fp8_e4m3fn"
    clip_type: str = "wan"
    clip_device: str = "default"

    @classmethod
    def default(cls) -> 'WanI2VModel':
        """Returns the default Wan I2V model configuration"""
        return cls()


@dataclass
class WanI2VLightningModel:
    """Configuration for Wan 2.2 I2V Lightning (4 steps) model"""
    unet_high_noise_name: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    unet_low_noise_name: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
    clip_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_name: str = "wan_2.1_vae.safetensors"
    lora_high_noise_name: str = "Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors"
    lora_low_noise_name: str = "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors"
    weight_dtype: str = "fp8_e4m3fn"
    clip_type: str = "wan"
    clip_device: str = "default"

    @classmethod
    def default(cls) -> 'WanI2VLightningModel':
        """Returns the default Wan I2V Lightning model configuration"""
        return cls()


@dataclass
class WanI2VWorkflowParams:
    """Parameters for configuring a Wan I2V workflow"""
    prompt: str
    start_image: str  # Path to start frame image
    end_image: Optional[str] = None  # Path to end frame image (optional)
    model: WanI2VModel = None
    negative_prompt: str = ""
    size: Size = Sizes.VIDEO_480P_LANDSCAPE
    frame_count: int = 81
    fps: int = 16
    cfg: float = 4.0
    steps: int = 20
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 5.0  # ModelSamplingSD3 shift parameter
    sage_attention: str = "auto"  # "auto" or "disabled"
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.model is None:
            self.model = WanI2VModel.default()
        if self.seed is None:
            self.seed = random.randint(0, 2**64)


@dataclass
class WanI2VLightningWorkflowParams:
    """Parameters for configuring a Wan I2V Lightning workflow (4 steps, cfg=1.0)"""
    prompt: str
    start_image: str  # Path to start frame image
    end_image: Optional[str] = None  # Path to end frame image (optional)
    model: WanI2VLightningModel = None
    negative_prompt: str = ""
    size: Size = Sizes.VIDEO_480P_LANDSCAPE
    frame_count: int = 81
    fps: int = 16
    cfg: float = 1.0  # Lightning uses cfg=1.0
    steps: int = 4  # Lightning uses 4 steps
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 5.0  # ModelSamplingSD3 shift parameter
    sage_attention: str = "auto"  # "auto" or "disabled"
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.model is None:
            self.model = WanI2VLightningModel.default()
        if self.seed is None:
            self.seed = random.randint(0, 2**64)


class WanI2VWorkflow(ComfyWorkflow):
    """A workflow for the Wan 2.2 I2V model with image-to-video generation."""

    def __init__(self, params: WanI2VWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: WanI2VWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
        self.start_image_node_id = None
        self.end_image_node_id = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load models
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        }, title="Load VAE")

        clip = self.add_node("CLIPLoader", {
            "clip_name": self.params.model.clip_name,
            "type": self.params.model.clip_type,
            "device": self.params.model.clip_device
        }, title="Load CLIP")

        unet_high = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_high_noise_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        unet_low = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_low_noise_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        # Apply Sage Attention patches
        sage_high = self.add_node("PathchSageAttentionKJ", {
            "sage_attention": self.params.sage_attention,
            "model": [unet_high, 0]
        }, title="Patch Sage Attention KJ")

        sage_low = self.add_node("PathchSageAttentionKJ", {
            "sage_attention": self.params.sage_attention,
            "model": [unet_low, 0]
        }, title="Patch Sage Attention KJ")

        # Apply model sampling
        model_sampling_high = self.add_node("ModelSamplingSD3", {
            "shift": self.params.shift,
            "model": [sage_high, 0]
        }, title="ModelSamplingSD3")

        model_sampling_low = self.add_node("ModelSamplingSD3", {
            "shift": self.params.shift,
            "model": [sage_low, 0]
        }, title="ModelSamplingSD3")

        # Encode prompts
        positive_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Positive Prompt)")

        negative_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.negative_prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Load start image
        load_start_image = self.add_node("LoadImage", {
            "image": self.params.start_image
        }, title="Load Image")
        self.start_image_node_id = load_start_image

        # Determine which node to use based on whether end_image is provided
        if self.params.end_image:
            # Load end image
            load_end_image = self.add_node("LoadImage", {
                "image": self.params.end_image
            }, title="Load Image")
            self.end_image_node_id = load_end_image

            # Use WanFirstLastFrameToVideo
            wan_video = self.add_node("WanFirstLastFrameToVideo", {
                "width": self.params.size[0],
                "height": self.params.size[1],
                "length": self.params.frame_count,
                "batch_size": self.params.batch_size,
                "positive": [positive_prompt, 0],
                "negative": [negative_prompt, 0],
                "vae": [vae, 0],
                "start_image": [load_start_image, 0],
                "end_image": [load_end_image, 0]
            }, title="WanFirstLastFrameToVideo")
        else:
            # Use WanImageToVideo (only start image)
            wan_video = self.add_node("WanImageToVideo", {
                "width": self.params.size[0],
                "height": self.params.size[1],
                "length": self.params.frame_count,
                "batch_size": self.params.batch_size,
                "positive": [positive_prompt, 0],
                "negative": [negative_prompt, 0],
                "vae": [vae, 0],
                "start_image": [load_start_image, 0]
            }, title="WanImageToVideo")

        # Calculate step split (high noise: 0 to steps/2, low noise: steps/2 to steps)
        high_noise_end_step = self.params.steps // 2
        low_noise_start_step = high_noise_end_step

        # High noise sampler
        sampler_high = self.add_node("KSamplerAdvanced", {
            "add_noise": "enable",
            "noise_seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "start_at_step": 0,
            "end_at_step": high_noise_end_step,
            "return_with_leftover_noise": "enable",
            "model": [model_sampling_high, 0],
            "positive": [wan_video, 0],
            "negative": [wan_video, 1],
            "latent_image": [wan_video, 2]
        }, title="KSampler (Advanced)")

        # Low noise sampler
        sampler_low = self.add_node("KSamplerAdvanced", {
            "add_noise": "disable",
            "noise_seed": 42,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "start_at_step": low_noise_start_step,
            "end_at_step": self.params.steps,
            "return_with_leftover_noise": "disable",
            "model": [model_sampling_low, 0],
            "positive": [wan_video, 0],
            "negative": [wan_video, 1],
            "latent_image": [sampler_high, 0]
        }, title="KSampler (Advanced)")

        # Decode
        decode = self.add_node("VAEDecode", {
            "samples": [sampler_low, 0],
            "vae": [vae, 0]
        }, title="VAE Decode")

        # Create video
        create_video = self.add_node("CreateVideo", {
            "fps": self.params.fps,
            "images": [decode, 0]
        }, title="Create Video")

        # Save video to disk (ComfyUI doesn't have SaveVideoWebsocket)
        self.add_node("SaveVideo", {
            "filename_prefix": "wan_i2v/video",
            "format": "mp4",
            "codec": "h264",
            "video": [create_video, 0]
        }, title="Save Video")

        # Also save first frame via websocket for preview/completion detection
        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")


class WanI2VLightningWorkflow(ComfyWorkflow):
    """A workflow for the Wan 2.2 I2V Lightning model (4 steps, cfg=1.0) with image-to-video generation."""

    def __init__(self, params: WanI2VLightningWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: WanI2VLightningWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
        self.start_image_node_id = None
        self.end_image_node_id = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load models
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        }, title="Load VAE")

        clip = self.add_node("CLIPLoader", {
            "clip_name": self.params.model.clip_name,
            "type": self.params.model.clip_type,
            "device": self.params.model.clip_device
        }, title="Load CLIP")

        unet_high = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_high_noise_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        unet_low = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_low_noise_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        # Apply Sage Attention patches
        sage_high = self.add_node("PathchSageAttentionKJ", {
            "sage_attention": self.params.sage_attention,
            "model": [unet_high, 0]
        }, title="Patch Sage Attention KJ")

        sage_low = self.add_node("PathchSageAttentionKJ", {
            "sage_attention": self.params.sage_attention,
            "model": [unet_low, 0]
        }, title="Patch Sage Attention KJ")

        # Load LoRAs for Lightning
        lora_high = self.add_node("LoraLoaderModelOnly", {
            "lora_name": self.params.model.lora_high_noise_name,
            "strength_model": 1.0,
            "model": [sage_high, 0]
        }, title="LoraLoaderModelOnly")

        lora_low = self.add_node("LoraLoaderModelOnly", {
            "lora_name": self.params.model.lora_low_noise_name,
            "strength_model": 1.0,
            "model": [sage_low, 0]
        }, title="LoraLoaderModelOnly")

        # Apply model sampling
        model_sampling_high = self.add_node("ModelSamplingSD3", {
            "shift": self.params.shift,
            "model": [lora_high, 0]
        }, title="ModelSamplingSD3")

        model_sampling_low = self.add_node("ModelSamplingSD3", {
            "shift": self.params.shift,
            "model": [lora_low, 0]
        }, title="ModelSamplingSD3")

        # Encode prompts
        positive_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Positive Prompt)")

        negative_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.negative_prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Load start image
        load_start_image = self.add_node("LoadImage", {
            "image": self.params.start_image
        }, title="Load Image")
        self.start_image_node_id = load_start_image

        # Determine which node to use based on whether end_image is provided
        if self.params.end_image:
            # Load end image
            load_end_image = self.add_node("LoadImage", {
                "image": self.params.end_image
            }, title="Load Image")
            self.end_image_node_id = load_end_image

            # Use WanFirstLastFrameToVideo
            wan_video = self.add_node("WanFirstLastFrameToVideo", {
                "width": self.params.size[0],
                "height": self.params.size[1],
                "length": self.params.frame_count,
                "batch_size": self.params.batch_size,
                "positive": [positive_prompt, 0],
                "negative": [negative_prompt, 0],
                "vae": [vae, 0],
                "start_image": [load_start_image, 0],
                "end_image": [load_end_image, 0]
            }, title="WanFirstLastFrameToVideo")
        else:
            # Use WanImageToVideo (only start image)
            wan_video = self.add_node("WanImageToVideo", {
                "width": self.params.size[0],
                "height": self.params.size[1],
                "length": self.params.frame_count,
                "batch_size": self.params.batch_size,
                "positive": [positive_prompt, 0],
                "negative": [negative_prompt, 0],
                "vae": [vae, 0],
                "start_image": [load_start_image, 0]
            }, title="WanImageToVideo")

        # Calculate step split (high noise: 0 to 2, low noise: 2 to 4)
        high_noise_end_step = 2
        low_noise_start_step = 2

        # High noise sampler
        sampler_high = self.add_node("KSamplerAdvanced", {
            "add_noise": "enable",
            "noise_seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "start_at_step": 0,
            "end_at_step": high_noise_end_step,
            "return_with_leftover_noise": "enable",
            "model": [model_sampling_high, 0],
            "positive": [wan_video, 0],
            "negative": [wan_video, 1],
            "latent_image": [wan_video, 2]
        }, title="KSampler (Advanced)")

        # Low noise sampler
        sampler_low = self.add_node("KSamplerAdvanced", {
            "add_noise": "disable",
            "noise_seed": 42,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "start_at_step": low_noise_start_step,
            "end_at_step": self.params.steps,
            "return_with_leftover_noise": "disable",
            "model": [model_sampling_low, 0],
            "positive": [wan_video, 0],
            "negative": [wan_video, 1],
            "latent_image": [sampler_high, 0]
        }, title="KSampler (Advanced)")

        # Decode
        decode = self.add_node("VAEDecode", {
            "samples": [sampler_low, 0],
            "vae": [vae, 0]
        }, title="VAE Decode")

        # Create video
        create_video = self.add_node("CreateVideo", {
            "fps": self.params.fps,
            "images": [decode, 0]
        }, title="Create Video")

        # Save video to disk (ComfyUI doesn't have SaveVideoWebsocket)
        self.add_node("SaveVideo", {
            "filename_prefix": "wan_i2v/video",
            "format": "mp4",
            "codec": "h264",
            "video": [create_video, 0]
        }, title="Save Video")

        # Also save first frame via websocket for preview/completion detection
        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")
