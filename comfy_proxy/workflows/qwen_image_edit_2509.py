from dataclasses import dataclass
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora


@dataclass
class QwenImageEditPlusModel:
    """Configuration for a Qwen Image Edit 2509 model and its components"""
    unet_name: str = "Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "fp8_e4m3fn"
    clip_type: str = "qwen_image"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'QwenImageEditPlusModel':
        """Returns the default Qwen Image Edit model configuration"""
        return cls()


@dataclass
class QwenImageEditPlusLightningModel:
    """Configuration for Qwen Image Edit Plus Lightning (8 steps) model"""
    unet_name: str = "Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "fp8_e4m3fn"
    clip_type: str = "qwen_image"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []
        # Always include the lightning LoRA
        lightning_lora = Lora(name="Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors", weight=1.0)
        if lightning_lora not in self.loras:
            self.loras.insert(0, lightning_lora)

    @classmethod
    def default(cls) -> 'QwenImageEditPlusLightningModel':
        """Returns the default Qwen Image Edit Plus Lightning model configuration"""
        return cls()


@dataclass
class QwenImageEditPlusWorkflowParams:
    """Parameters for configuring a Qwen Image Edit workflow"""
    prompt: str
    image: str  # Path to input image or image data
    model: QwenImageEditPlusModel
    negative_prompt: str = "cartoon, anime, ugly"
    size: Size = Sizes.SQUARE_1K  # Target megapixels, will be scaled
    megapixels: float = 1.0  # Target megapixels for scaling
    cfg: float = 4.0
    cfg_norm_strength: float = 1.0  # CFGNorm strength
    steps: int = 30
    scheduler: str = "simple"
    sampler: str = "euler"
    denoise: float = 1.0
    shift: float = 3.0  # ModelSamplingAuraFlow shift parameter
    upscale_method: str = "lanczos"
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**64)


@dataclass
class QwenImageEditPlusLightningWorkflowParams:
    """Parameters for configuring a Qwen Image Edit Plus Lightning workflow (8 steps, cfg=1.0)"""
    prompt: str
    image: str  # Path to input image or image data
    model: QwenImageEditPlusLightningModel
    negative_prompt: str = "cartoon, anime, ugly"
    size: Size = Sizes.SQUARE_1K  # Target megapixels, will be scaled
    megapixels: float = 1.0  # Target megapixels for scaling
    cfg: float = 1.0  # Lightning uses cfg=1.0
    cfg_norm_strength: float = 1.0  # CFGNorm strength
    steps: int = 8  # Lightning uses 8 steps
    scheduler: str = "simple"
    sampler: str = "euler"
    denoise: float = 1.0
    shift: float = 3.0  # ModelSamplingAuraFlow shift parameter
    upscale_method: str = "lanczos"
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**64)


class QwenImageEditPlusWorkflow(ComfyWorkflow):
    """A workflow for the Qwen Image Edit 2509 model with image input support and LoRA chaining."""

    def __init__(self, params: QwenImageEditPlusWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: QwenImageEditPlusWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
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

        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        # Chain LoRAs if present (using LoraLoaderModelOnly for model-only LoRAs)
        current_model = unet

        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoaderModelOnly", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "model": [current_model, 0]
            }, title="LoraLoaderModelOnly")
            current_model = lora

        # Load and scale input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.image
        }, title="Load Image")

        scale_image = self.add_node("ImageScaleToTotalPixels", {
            "upscale_method": self.params.upscale_method,
            "megapixels": self.params.megapixels,
            "image": [load_image, 0]
        }, title="Scale Image to Total Pixels")

        # Encode the input image to latent space
        vae_encode = self.add_node("VAEEncode", {
            "pixels": [scale_image, 0],
            "vae": [vae, 0]
        }, title="VAE Encode")

        # Apply model sampling
        model_sampling = self.add_node("ModelSamplingAuraFlow", {
            "shift": self.params.shift,
            "model": [current_model, 0]
        }, title="ModelSamplingAuraFlow")

        # Apply CFG normalization
        cfg_norm = self.add_node("CFGNorm", {
            "strength": self.params.cfg_norm_strength,
            "model": [model_sampling, 0]
        }, title="CFGNorm")

        # Encode prompts using TextEncodeQwenImageEditPlus
        positive_prompt = self.add_node("TextEncodeQwenImageEditPlus", {
            "prompt": self.params.prompt,
            "clip": [clip, 0],
            "vae": [vae, 0],
            "image1": [scale_image, 0]
        }, title="TextEncodeQwenImageEditPlus")

        negative_prompt = self.add_node("TextEncodeQwenImageEditPlus", {
            "prompt": self.params.negative_prompt,
            "clip": [clip, 0],
            "vae": [vae, 0],
            "image1": [scale_image, 0]
        }, title="TextEncodeQwenImageEditPlus")

        # Sample
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [cfg_norm, 0],
            "positive": [positive_prompt, 0],
            "negative": [negative_prompt, 0],
            "latent_image": [vae_encode, 0]
        }, title="KSampler")

        # Decode
        decode = self.add_node("VAEDecode", {
            "samples": [sampler, 0],
            "vae": [vae, 0]
        }, title="VAE Decode")

        # Save via websocket
        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")


class QwenImageEditPlusLightningWorkflow(ComfyWorkflow):
    """A workflow for the Qwen Image Edit Plus Lightning model (8 steps, cfg=1.0) with LoRA support."""

    def __init__(self, params: QwenImageEditPlusLightningWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: QwenImageEditPlusLightningWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
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

        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        # Chain LoRAs (including lightning LoRA which is auto-added in model __post_init__)
        current_model = unet

        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoaderModelOnly", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "model": [current_model, 0]
            }, title="LoraLoaderModelOnly")
            current_model = lora

        # Load and scale input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.image
        }, title="Load Image")

        scale_image = self.add_node("ImageScaleToTotalPixels", {
            "upscale_method": self.params.upscale_method,
            "megapixels": self.params.megapixels,
            "image": [load_image, 0]
        }, title="Scale Image to Total Pixels")

        # Encode the input image to latent space
        vae_encode = self.add_node("VAEEncode", {
            "pixels": [scale_image, 0],
            "vae": [vae, 0]
        }, title="VAE Encode")

        # Apply model sampling
        model_sampling = self.add_node("ModelSamplingAuraFlow", {
            "shift": self.params.shift,
            "model": [current_model, 0]
        }, title="ModelSamplingAuraFlow")

        # Apply CFG normalization
        cfg_norm = self.add_node("CFGNorm", {
            "strength": self.params.cfg_norm_strength,
            "model": [model_sampling, 0]
        }, title="CFGNorm")

        # Encode prompts using TextEncodeQwenImageEditPlus
        positive_prompt = self.add_node("TextEncodeQwenImageEditPlus", {
            "prompt": self.params.prompt,
            "clip": [clip, 0],
            "vae": [vae, 0],
            "image1": [scale_image, 0]
        }, title="TextEncodeQwenImageEditPlus")

        negative_prompt = self.add_node("TextEncodeQwenImageEditPlus", {
            "prompt": self.params.negative_prompt,
            "clip": [clip, 0],
            "vae": [vae, 0],
            "image1": [scale_image, 0]
        }, title="TextEncodeQwenImageEditPlus")

        # Sample
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [cfg_norm, 0],
            "positive": [positive_prompt, 0],
            "negative": [negative_prompt, 0],
            "latent_image": [vae_encode, 0]
        }, title="KSampler")

        # Decode
        decode = self.add_node("VAEDecode", {
            "samples": [sampler, 0],
            "vae": [vae, 0]
        }, title="VAE Decode")

        # Save via websocket
        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")
