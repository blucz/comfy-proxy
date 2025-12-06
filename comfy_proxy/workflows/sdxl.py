"""SDXL text-to-image workflow for ComfyUI."""
from dataclasses import dataclass, field
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora


@dataclass
class SDXLModel:
    """Configuration for an SDXL model checkpoint."""
    checkpoint_name: str = "sd_xl_base_1.0.safetensors"
    loras: List[Lora] = field(default_factory=list)

    @classmethod
    def default(cls) -> 'SDXLModel':
        """Returns the default SDXL model configuration."""
        return cls()


@dataclass
class SDXLWorkflowParams:
    """Parameters for configuring an SDXL text-to-image workflow."""
    prompt: str
    model: SDXLModel = field(default_factory=SDXLModel.default)
    negative_prompt: str = "text, watermark"
    size: Size = Sizes.SQUARE_1K  # SDXL trained on 1024x1024
    cfg: float = 8.0
    steps: int = 20
    scheduler: str = "normal"
    sampler: str = "euler"
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)


class SDXLWorkflow(ComfyWorkflow):
    """A workflow for SDXL models with LoRA support."""

    def __init__(self, params: SDXLWorkflowParams):
        """Create a workflow based on the provided parameters.

        Args:
            params: SDXLWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure."""
        # Load checkpoint (MODEL, CLIP, VAE in one node)
        checkpoint = self.add_node("CheckpointLoaderSimple", {
            "ckpt_name": self.params.model.checkpoint_name
        }, title="Load Checkpoint")

        # Chain LoRAs if present
        current_model = checkpoint
        current_clip = checkpoint
        model_output_index = 0
        clip_output_index = 1

        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoader", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "strength_clip": lora_spec.weight,
                "model": [current_model, model_output_index],
                "clip": [current_clip, clip_output_index]
            }, title="Load LoRA")
            current_model = lora
            current_clip = lora
            model_output_index = 0  # LoraLoader outputs model at 0
            clip_output_index = 1   # LoraLoader outputs clip at 1

        # Encode prompts
        positive_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Positive)")

        negative_prompt = self.add_node("CLIPTextEncode", {
            "text": self.params.negative_prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Negative)")

        # Create empty latent
        latent = self.add_node("EmptyLatentImage", {
            "width": self.params.size[0],
            "height": self.params.size[1],
            "batch_size": self.params.batch_size
        }, title="Empty Latent Image")

        # Sample using KSamplerAdvanced
        sampler = self.add_node("KSamplerAdvanced", {
            "add_noise": "enable",
            "noise_seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": self.params.cfg,
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "start_at_step": 0,
            "end_at_step": self.params.steps,
            "return_with_leftover_noise": "disable",
            "model": [current_model, model_output_index],
            "positive": [positive_prompt, 0],
            "negative": [negative_prompt, 0],
            "latent_image": [latent, 0]
        }, title="KSampler (Advanced)")

        # Decode with VAE
        decode = self.add_node("VAEDecode", {
            "samples": [sampler, 0],
            "vae": [checkpoint, 2]  # VAE is output 2 from CheckpointLoaderSimple
        }, title="VAE Decode")

        # Save via websocket
        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")
