from dataclasses import dataclass
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora

@dataclass
class QwenImageModel:
    """Configuration for a Qwen Image model and its components"""
    unet_name: str = "qwen_image_fp8_e4m3fn.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"
    clip_type: str = "qwen_image"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'QwenImageModel':
        """Returns the default Qwen Image model configuration"""
        return cls()

@dataclass
class QwenImageWorkflowParams:
    """Parameters for configuring a Qwen Image workflow"""
    prompt: str
    model: QwenImageModel
    negative_prompt: str = "ugly, cartoon, 3d, video game, cg"
    size: Size = Sizes.PORTRAIT_3_4  # Default 848x1152 as in example
    cfg: float = 3.5
    steps: int = 20
    scheduler: str = "simple"
    sampler: str = "euler"
    denoise: float = 1.0
    shift: float = 3.1  # ModelSamplingAuraFlow shift parameter
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)

class QwenImageWorkflow(ComfyWorkflow):
    """A workflow for the Qwen Image model with LoRA support."""
    
    def __init__(self, params: QwenImageWorkflowParams):
        """Create a workflow based on the provided parameters
        
        Args:
            params: QwenImageWorkflowParams object containing all generation parameters
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

        # Apply model sampling
        model_sampling = self.add_node("ModelSamplingAuraFlow", {
            "shift": self.params.shift,
            "model": [current_model, 0]
        }, title="ModelSamplingAuraFlow")

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

        # Sample
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [model_sampling, 0],
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
