from dataclasses import dataclass
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora


@dataclass
class QwenImageInpaintModel:
    """Configuration for Qwen Image Inpaint model and its components"""
    unet_name: str = "qwen_image_fp8_e4m3fn.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    controlnet_name: str = "Qwen-Image-Controlnet-Inpainting.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"
    clip_type: str = "qwen_image"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'QwenImageInpaintModel':
        """Returns the default Qwen Image Inpaint model configuration"""
        return cls()


@dataclass
class QwenImageInpaintLightningModel:
    """Configuration for Qwen Image Inpaint Lightning (8 steps, cfg=1) model"""
    unet_name: str = "qwen_image_fp8_e4m3fn.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    controlnet_name: str = "Qwen-Image-Controlnet-Inpainting.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"
    clip_type: str = "qwen_image"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []
        # Always include the lightning LoRA
        lightning_lora = Lora(name="qwen/Qwen-Image-Lightning-8steps-V2.0.safetensors", weight=1.0)
        if lightning_lora not in self.loras:
            self.loras.insert(0, lightning_lora)

    @classmethod
    def default(cls) -> 'QwenImageInpaintLightningModel':
        """Returns the default Qwen Image Inpaint Lightning model configuration"""
        return cls()


@dataclass
class QwenImageInpaintWorkflowParams:
    """Parameters for configuring a Qwen Image Inpaint workflow"""
    prompt: str
    image: str  # Path to input image
    mask: str  # Path to mask image (white = inpaint area)
    model: QwenImageInpaintModel
    negative_prompt: str = " "
    strength: float = 0.8  # ControlNet inpainting strength (AliMama node)
    cfg: float = 2.5
    steps: int = 20
    scheduler: str = "beta"
    sampler: str = "dpmpp_sde"
    denoise: float = 1.0
    shift: float = 3.1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**64)


@dataclass
class QwenImageInpaintLightningWorkflowParams:
    """Parameters for configuring a Qwen Image Inpaint Lightning workflow (8 steps, cfg=1)"""
    prompt: str
    image: str  # Path to input image
    mask: str  # Path to mask image (white = inpaint area)
    model: QwenImageInpaintLightningModel
    negative_prompt: str = " "
    strength: float = 0.8  # ControlNet inpainting strength (AliMama node)
    cfg: float = 1.0  # Lightning uses cfg=1
    steps: int = 8  # Lightning uses 8 steps
    scheduler: str = "beta"
    sampler: str = "dpmpp_sde"
    denoise: float = 1.0
    shift: float = 3.1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**64)


class QwenImageInpaintWorkflow(ComfyWorkflow):
    """A workflow for Qwen Image inpainting with mask provided by client."""

    def __init__(self, params: QwenImageInpaintWorkflowParams):
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

        controlnet = self.add_node("ControlNetLoader", {
            "control_net_name": self.params.model.controlnet_name
        }, title="Load ControlNet")

        # Chain LoRAs if present
        current_model = unet
        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoaderModelOnly", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "model": [current_model, 0]
            }, title="LoraLoaderModelOnly")
            current_model = lora

        # Apply model sampling
        model_sampling = self.add_node("ModelSamplingAuraFlow", {
            "shift": self.params.shift,
            "model": [current_model, 0]
        }, title="ModelSamplingAuraFlow")

        # Load input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.image
        }, title="Load Image")

        # Load mask
        load_mask = self.add_node("LoadImage", {
            "image": self.params.mask
        }, title="Load Mask")

        # Encode prompts
        positive_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Positive Prompt)")

        negative_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.negative_prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Apply ControlNet inpainting (AliMama node)
        controlnet_apply = self.add_node("ControlNetInpaintingAliMamaApply", {
            "positive": [positive_prompt, 0],
            "negative": [negative_prompt, 0],
            "control_net": [controlnet, 0],
            "vae": [vae, 0],
            "image": [load_image, 0],
            "mask": [load_mask, 1],  # Mask output from LoadImage
            "strength": self.params.strength,
            "start_percent": 0,
            "end_percent": 1
        }, title="ControlNetInpaintingAliMamaApply")

        # Encode image to latent
        vae_encode = self.add_node("VAEEncode", {
            "pixels": [load_image, 0],
            "vae": [vae, 0]
        }, title="VAE Encode")

        # Sample
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [model_sampling, 0],
            "positive": [controlnet_apply, 0],
            "negative": [controlnet_apply, 1],
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


class QwenImageInpaintLightningWorkflow(ComfyWorkflow):
    """A workflow for Qwen Image inpainting with Lightning (8 steps, cfg=1)."""

    def __init__(self, params: QwenImageInpaintLightningWorkflowParams):
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

        controlnet = self.add_node("ControlNetLoader", {
            "control_net_name": self.params.model.controlnet_name
        }, title="Load ControlNet")

        # Chain LoRAs (including lightning LoRA which is auto-added)
        current_model = unet
        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoaderModelOnly", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "model": [current_model, 0]
            }, title="LoraLoaderModelOnly")
            current_model = lora

        # Apply model sampling
        model_sampling = self.add_node("ModelSamplingAuraFlow", {
            "shift": self.params.shift,
            "model": [current_model, 0]
        }, title="ModelSamplingAuraFlow")

        # Load input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.image
        }, title="Load Image")

        # Load mask
        load_mask = self.add_node("LoadImage", {
            "image": self.params.mask
        }, title="Load Mask")

        # Encode prompts
        positive_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Positive Prompt)")

        negative_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.negative_prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Apply ControlNet inpainting (AliMama node)
        controlnet_apply = self.add_node("ControlNetInpaintingAliMamaApply", {
            "positive": [positive_prompt, 0],
            "negative": [negative_prompt, 0],
            "control_net": [controlnet, 0],
            "vae": [vae, 0],
            "image": [load_image, 0],
            "mask": [load_mask, 1],  # Mask output from LoadImage
            "strength": self.params.strength,
            "start_percent": 0,
            "end_percent": 1
        }, title="ControlNetInpaintingAliMamaApply")

        # Encode image to latent
        vae_encode = self.add_node("VAEEncode", {
            "pixels": [load_image, 0],
            "vae": [vae, 0]
        }, title="VAE Encode")

        # Sample
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [model_sampling, 0],
            "positive": [controlnet_apply, 0],
            "negative": [controlnet_apply, 1],
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
