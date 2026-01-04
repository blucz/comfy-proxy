from dataclasses import dataclass
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora


@dataclass
class FluxFillModel:
    """Configuration for Flux.1 Fill Dev model (inpaint/outpaint)"""
    clip_name1: str = "clip_l.safetensors"
    clip_name2: str = "t5xxl_fp16.safetensors"
    vae_name: str = "ae.safetensors"
    unet_name: str = "flux1-fill-dev.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'FluxFillModel':
        """Returns the default Flux Fill model configuration"""
        return cls()


@dataclass
class FluxFillInpaintWorkflowParams:
    """Parameters for Flux Fill inpainting workflow.

    Takes an image and mask (white = area to inpaint) and fills the masked region.
    """
    prompt: str
    image: str  # Path to input image
    mask: str   # Path to mask image (white = inpaint area)
    model: FluxFillModel = None
    guidance: float = 30.0  # Flux Fill uses higher guidance
    steps: int = 20
    scheduler: str = "normal"
    sampler: str = "euler"
    denoise: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        if self.model is None:
            self.model = FluxFillModel.default()


@dataclass
class FluxFillOutpaintWorkflowParams:
    """Parameters for Flux Fill outpainting workflow.

    Expands an image by padding sides and filling the new areas.
    """
    prompt: str
    image: str  # Path to input image
    model: FluxFillModel = None
    # Padding in pixels for each side
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    feathering: int = 24  # Feathering for the mask edge
    guidance: float = 30.0  # Flux Fill uses higher guidance
    steps: int = 20
    scheduler: str = "normal"
    sampler: str = "euler"
    denoise: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        if self.model is None:
            self.model = FluxFillModel.default()


class FluxFillInpaintWorkflow(ComfyWorkflow):
    """Workflow for Flux.1 Fill Dev inpainting.

    Uses the flux1-fill-dev model which is specifically trained for inpainting.
    The mask should have white areas where content should be generated.

    Based on official ComfyUI workflow:
    - UNETLoader -> DifferentialDiffusion -> KSampler
    - InpaintModelConditioning with noise_mask=false
    - Standard KSampler (not SamplerCustomAdvanced)
    """

    def __init__(self, params: FluxFillInpaintWorkflowParams):
        super().__init__()
        self.params = params
        self.input_image_node_id: str = None
        self.mask_image_node_id: str = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load UNET (flux1-fill-dev)
        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        # Load CLIP (DualCLIP for Flux) - note the order: clip_l first, then t5xxl
        clip = self.add_node("DualCLIPLoader", {
            "clip_name1": self.params.model.clip_name1,
            "clip_name2": self.params.model.clip_name2,
            "type": "flux"
        }, title="DualCLIPLoader")

        # Load VAE
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        }, title="Load VAE")

        # Load input image (with mask embedded)
        load_image = self.add_node("LoadImage", {
            "image": self.params.image
        }, title="Load Image")
        self.input_image_node_id = load_image

        # Load mask image
        load_mask = self.add_node("LoadImage", {
            "image": self.params.mask
        }, title="Load Mask")
        self.mask_image_node_id = load_mask

        # Chain LoRAs if present
        current_model = unet
        current_clip = clip
        clip_output_index = 0

        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoader", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "strength_clip": lora_spec.weight,
                "model": [current_model, 0],
                "clip": [current_clip, clip_output_index]
            }, title="LoraLoader")
            clip_output_index = 1
            current_model = lora
            current_clip = lora

        # DifferentialDiffusion - wraps the model for fill operations
        diff_diffusion = self.add_node("DifferentialDiffusion", {
            "model": [current_model, 0]
        }, title="DifferentialDiffusion")

        # Encode positive prompt
        prompt_encode = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Positive Prompt)")

        # Encode negative prompt (empty for Flux)
        negative_encode = self.add_node("CLIPTextEncode", {
            "text": "",
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Apply FluxGuidance
        guidance = self.add_node("FluxGuidance", {
            "guidance": self.params.guidance,
            "conditioning": [prompt_encode, 0]
        }, title="FluxGuidance")

        # InpaintModelConditioning - creates proper conditioning for fill model
        # noise_mask should be False per official workflow
        inpaint_cond = self.add_node("InpaintModelConditioning", {
            "positive": [guidance, 0],
            "negative": [negative_encode, 0],
            "vae": [vae, 0],
            "pixels": [load_image, 0],
            "mask": [load_mask, 1],  # MASK output from LoadImage
            "noise_mask": False
        }, title="InpaintModelConditioning")

        # Standard KSampler
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": 1,  # Flux uses cfg=1 with FluxGuidance
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [diff_diffusion, 0],
            "positive": [inpaint_cond, 0],
            "negative": [inpaint_cond, 1],
            "latent_image": [inpaint_cond, 2]
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


class FluxFillOutpaintWorkflow(ComfyWorkflow):
    """Workflow for Flux.1 Fill Dev outpainting.

    Pads the image and fills the new areas using the flux1-fill-dev model.

    Based on official ComfyUI workflow:
    - ImagePadForOutpaint generates padded image and mask
    - UNETLoader -> DifferentialDiffusion -> KSampler
    - InpaintModelConditioning with noise_mask=false
    """

    def __init__(self, params: FluxFillOutpaintWorkflowParams):
        super().__init__()
        self.params = params
        self.input_image_node_id: str = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load UNET (flux1-fill-dev)
        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

        # Load CLIP (DualCLIP for Flux) - note the order: clip_l first, then t5xxl
        clip = self.add_node("DualCLIPLoader", {
            "clip_name1": self.params.model.clip_name1,
            "clip_name2": self.params.model.clip_name2,
            "type": "flux"
        }, title="DualCLIPLoader")

        # Load VAE
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        }, title="Load VAE")

        # Load input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.image
        }, title="Load Image")
        self.input_image_node_id = load_image

        # Chain LoRAs if present
        current_model = unet
        current_clip = clip
        clip_output_index = 0

        for lora_spec in self.params.model.loras:
            lora = self.add_node("LoraLoader", {
                "lora_name": lora_spec.name,
                "strength_model": lora_spec.weight,
                "strength_clip": lora_spec.weight,
                "model": [current_model, 0],
                "clip": [current_clip, clip_output_index]
            }, title="LoraLoader")
            clip_output_index = 1
            current_model = lora
            current_clip = lora

        # DifferentialDiffusion - wraps the model for fill operations
        diff_diffusion = self.add_node("DifferentialDiffusion", {
            "model": [current_model, 0]
        }, title="DifferentialDiffusion")

        # Encode positive prompt
        prompt_encode = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Positive Prompt)")

        # Encode negative prompt (empty for Flux)
        negative_encode = self.add_node("CLIPTextEncode", {
            "text": "",
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Negative Prompt)")

        # Apply FluxGuidance
        guidance = self.add_node("FluxGuidance", {
            "guidance": self.params.guidance,
            "conditioning": [prompt_encode, 0]
        }, title="FluxGuidance")

        # Pad image for outpainting - generates both padded image and mask
        pad_image = self.add_node("ImagePadForOutpaint", {
            "image": [load_image, 0],
            "left": self.params.left,
            "top": self.params.top,
            "right": self.params.right,
            "bottom": self.params.bottom,
            "feathering": self.params.feathering
        }, title="Pad Image for Outpaint")

        # InpaintModelConditioning - creates proper conditioning for fill model
        # noise_mask should be False per official workflow
        inpaint_cond = self.add_node("InpaintModelConditioning", {
            "positive": [guidance, 0],
            "negative": [negative_encode, 0],
            "vae": [vae, 0],
            "pixels": [pad_image, 0],  # Padded image
            "mask": [pad_image, 1],  # MASK from padding
            "noise_mask": False
        }, title="InpaintModelConditioning")

        # Standard KSampler
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": 1,  # Flux uses cfg=1 with FluxGuidance
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": self.params.denoise,
            "model": [diff_diffusion, 0],
            "positive": [inpaint_cond, 0],
            "negative": [inpaint_cond, 1],
            "latent_image": [inpaint_cond, 2]
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
