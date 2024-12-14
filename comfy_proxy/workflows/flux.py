from dataclasses import dataclass
from typing import List, Optional, Dict
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora

@dataclass
class FluxModel:
    """Configuration for a Flux model and its components"""
    clip_name1: str = "t5xxl_fp16.safetensors"
    clip_name2: str = "clip_l.safetensors"
    vae_name: str = "ae.sft"
    unet_name: str = "flux1-dev.sft"
    loras: List['Lora'] = None  # Forward reference since Lora is defined later

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'FluxModel':
        """Returns the default Flux model configuration"""
        return cls()

@dataclass
class FluxWorkflowParams:
    """Parameters for configuring a ComfyUI workflow"""
    prompt: str
    model: FluxModel
    size: Size = Sizes.SQUARE_1K  # type: tuple[int, int]
    guidance: float = 3.5
    steps: int = 20
    scheduler: str = "normal"
    sampler: str = "euler"
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)

class FluxWorkflow(ComfyWorkflow):
    """A workflow for the Flux model with lora support."""
    
    def __init__(self, params: FluxWorkflowParams):
        """Create a workflow based on the provided parameters
        
        Args:
            params: FluxWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the internal workflow structure"""
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        })
        
        clip = self.add_node("DualCLIPLoader", {
            "clip_name1": self.params.model.clip_name1,
            "clip_name2": self.params.model.clip_name2,
            "type": "flux"
        })
        
        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": "fp8_e4m3fn"
        })

        sampler = self.add_node("KSamplerSelect", {
            "sampler_name": self.params.sampler
        })

        # Chain LoRAs in sequence
        current_model = unet
        current_clip = clip
        clip_output_index = 0  # Start with output 0
        
        # If we have LoRAs, the final CLIP output will be on index 1
        if self.params.model.loras:
            clip_output_index = 1
            for lora_spec in self.params.model.loras:
                lora = self.add_node("LoraLoader", {
                    "lora_name": lora_spec.name,
                    "strength_model": lora_spec.weight,
                    "strength_clip": lora_spec.weight,
                    "model": [current_model, 0],
                    "clip": [current_clip, 0]
                })
                current_model = lora
                current_clip = lora

        prompt_encode = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        })

        model_sampling = self.add_node("ModelSamplingFlux", {
            "max_shift": 1.16,
            "base_shift": 0.5,
            "width": self.params.size[0],
            "height": self.params.size[1],
            "model": [current_model, 0]
        })

        scheduler = self.add_node("BasicScheduler", {
            "scheduler": self.params.scheduler,
            "steps": self.params.steps,
            "denoise": 1,
            "model": [model_sampling, 0]
        })

        guidance = self.add_node("FluxGuidance", {
            "guidance": self.params.guidance,
            "conditioning": [prompt_encode, 0]
        })

        noise = self.add_node("RandomNoise", {
            "noise_seed": self.params.seed
        })

        latent = self.add_node("EmptySD3LatentImage", {
            "width": self.params.size[0],
            "height": self.params.size[1],
            "batch_size": self.params.batch_size
        })

        basic_guider = self.add_node("BasicGuider", {
            "model": [model_sampling, 0],
            "conditioning": [guidance, 0]
        })

        sampler_advanced = self.add_node("SamplerCustomAdvanced", {
            "noise": [noise, 0],
            "guider": [basic_guider, 0],
            "sampler": [sampler, 0],
            "sigmas": [scheduler, 0],
            "latent_image": [latent, 0]
        })

        decode = self.add_node("VAEDecode", {
            "samples": [sampler_advanced, 0],
            "vae": [vae, 0]
        })

        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")
