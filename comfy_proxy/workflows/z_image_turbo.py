from dataclasses import dataclass
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Size, Lora

@dataclass
class ZImageTurboModel:
    """Configuration for a Z-Image Turbo model and its components"""
    unet_name: str = "z_image_turbo_bf16.safetensors"
    clip_name: str = "qwen_3_4b.safetensors"
    vae_name: str = "ae.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"
    clip_type: str = "lumina2"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'ZImageTurboModel':
        """Returns the default Z-Image Turbo model configuration"""
        return cls()

@dataclass
class ZImageTurboWorkflowParams:
    """Parameters for configuring a Z-Image Turbo workflow"""
    prompt: str
    model: ZImageTurboModel
    negative_prompt: str = "blurry ugly bad"
    size: Size = (1536, 1024)  # Default as in example
    cfg: float = 1.0  # Z-Image Turbo uses cfg=1
    steps: int = 10  # Z-Image Turbo uses 10 steps
    scheduler: str = "simple"
    sampler: str = "euler"
    denoise: float = 1.0
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)

class ZImageTurboWorkflow(ComfyWorkflow):
    """A workflow for the Z-Image Turbo model with LoRA support."""

    def __init__(self, params: ZImageTurboWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: ZImageTurboWorkflowParams object containing all generation parameters
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

        # Encode prompts
        positive_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Positive Prompt)")

        negative_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.negative_prompt,
            "clip": [clip, 0]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Create empty latent
        latent = self.add_node("EmptySD3LatentImage", {
            "width": self.params.size[0],
            "height": self.params.size[1],
            "batch_size": self.params.batch_size
        }, title="EmptySD3LatentImage")

        # Sample (no ModelSamplingAuraFlow - it's bypassed in the example)
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [current_model, 0],
            "positive": [positive_prompt, 0],
            "negative": [negative_prompt, 0],
            "latent_image": [latent, 0]
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
