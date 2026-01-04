from dataclasses import dataclass, field
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora


@dataclass
class FluxKontextModel:
    """Configuration for Flux.1 Kontext Dev model (context-aware editing)"""
    clip_name1: str = "clip_l.safetensors"
    clip_name2: str = "t5xxl_fp16.safetensors"
    vae_name: str = "ae.safetensors"
    unet_name: str = "flux1-dev-kontext_fp8_scaled.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'FluxKontextModel':
        """Returns the default Flux Kontext model configuration"""
        return cls()


@dataclass
class FluxKontextWorkflowParams:
    """Parameters for Flux Kontext context-aware editing workflow.

    Takes reference images and a prompt to generate edited/styled output.
    Supports 1-10 reference images that are combined and processed.
    """
    prompt: str
    reference_images: List[str] = field(default_factory=list)  # Paths to reference images (1-10)
    model: FluxKontextModel = None
    size: Size = Sizes.SQUARE_1K  # Output size
    guidance: float = 2.5  # Kontext uses lower guidance than Fill
    steps: int = 20
    scheduler: str = "simple"
    sampler: str = "euler"
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        if self.model is None:
            self.model = FluxKontextModel.default()
        if len(self.reference_images) > 10:
            raise ValueError("Maximum 10 reference images supported")
        if len(self.reference_images) == 0:
            raise ValueError("At least 1 reference image is required for Kontext workflow")


class FluxKontextWorkflow(ComfyWorkflow):
    """Workflow for Flux.1 Kontext Dev context-aware image editing.

    Based on official ComfyUI workflow:
    - Uses FluxKontextImageScale to scale reference images
    - Uses ImageStitch to combine multiple reference images
    - Uses ReferenceLatent to inject latent conditioning
    - Uses ConditioningZeroOut for negative conditioning
    - Standard KSampler with cfg=1 and FluxGuidance
    """

    def __init__(self, params: FluxKontextWorkflowParams):
        super().__init__()
        self.params = params
        self.reference_image_node_ids: List[str] = []
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load UNET (flux1-dev-kontext)
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

        # Load and combine reference images
        # For single image: LoadImage -> FluxKontextImageScale
        # For multiple images: LoadImage(s) -> ImageStitch -> FluxKontextImageScale

        if len(self.params.reference_images) == 1:
            # Single reference image
            load_image = self.add_node("LoadImage", {
                "image": self.params.reference_images[0]
            }, title="Load Reference Image")
            self.reference_image_node_ids.append(load_image)

            # Scale for Kontext
            scaled_image = self.add_node("FluxKontextImageScale", {
                "image": [load_image, 0]
            }, title="FluxKontextImageScale")
        else:
            # Multiple reference images - stitch them together
            # Load first image
            prev_image = self.add_node("LoadImage", {
                "image": self.params.reference_images[0]
            }, title="Load Reference Image 1")
            self.reference_image_node_ids.append(prev_image)

            # Stitch additional images
            for i, ref_image in enumerate(self.params.reference_images[1:], start=2):
                load_image = self.add_node("LoadImage", {
                    "image": ref_image
                }, title=f"Load Reference Image {i}")
                self.reference_image_node_ids.append(load_image)

                stitch = self.add_node("ImageStitch", {
                    "image1": [prev_image, 0],
                    "image2": [load_image, 0],
                    "direction": "right",
                    "match_sizes": True,
                    "gap": 0,
                    "gap_color": "white"
                }, title=f"ImageStitch {i-1}")
                prev_image = stitch

            # Scale the stitched result
            scaled_image = self.add_node("FluxKontextImageScale", {
                "image": [prev_image, 0]
            }, title="FluxKontextImageScale")

        # Encode scaled image to latent
        vae_encode = self.add_node("VAEEncode", {
            "pixels": [scaled_image, 0],
            "vae": [vae, 0]
        }, title="VAE Encode")

        # Encode prompt
        prompt_encode = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Positive Prompt)")

        # ReferenceLatent - combines conditioning with reference latent
        ref_latent = self.add_node("ReferenceLatent", {
            "conditioning": [prompt_encode, 0],
            "latent": [vae_encode, 0]
        }, title="ReferenceLatent")

        # Apply FluxGuidance
        guidance = self.add_node("FluxGuidance", {
            "guidance": self.params.guidance,
            "conditioning": [ref_latent, 0]
        }, title="FluxGuidance")

        # ConditioningZeroOut for negative conditioning
        zero_cond = self.add_node("ConditioningZeroOut", {
            "conditioning": [prompt_encode, 0]
        }, title="ConditioningZeroOut")

        # Standard KSampler - uses the reference latent as input
        sampler = self.add_node("KSampler", {
            "seed": self.params.seed,
            "steps": self.params.steps,
            "cfg": 1,  # Flux uses cfg=1 with FluxGuidance
            "sampler_name": self.params.sampler,
            "scheduler": self.params.scheduler,
            "denoise": 1,
            "model": [current_model, 0],
            "positive": [guidance, 0],
            "negative": [zero_cond, 0],
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

    def get_image_uploads(self, image_paths: List[str]) -> dict:
        """Helper to create image_uploads dict for the generate() call.

        Args:
            image_paths: List of actual file paths corresponding to reference_images

        Returns:
            Dict mapping reference image placeholders to actual file paths
        """
        if len(image_paths) != len(self.params.reference_images):
            raise ValueError(
                f"Expected {len(self.params.reference_images)} image paths, "
                f"got {len(image_paths)}"
            )
        return {
            ref_img: path
            for ref_img, path in zip(self.params.reference_images, image_paths)
        }
