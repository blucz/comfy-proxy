from dataclasses import dataclass, field
from typing import List, Optional
import random
from ..workflow import ComfyWorkflow, Sizes, Size, Lora


@dataclass
class Flux2Model:
    """Configuration for a Flux2 model and its components"""
    clip_name: str = "mistral_3_small_flux2_fp8.safetensors"
    vae_name: str = "flux2-vae.safetensors"
    unet_name: str = "flux2_dev_fp8mixed.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"
    clip_type: str = "flux2"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'Flux2Model':
        """Returns the default Flux2 model configuration"""
        return cls()


@dataclass
class Flux2WorkflowParams:
    """Parameters for configuring a Flux2 text-to-image workflow"""
    prompt: str
    model: Flux2Model
    size: Size = Sizes.SQUARE_1K
    guidance: float = 3.5
    steps: int = 20
    sampler: str = "euler"
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)


class Flux2Workflow(ComfyWorkflow):
    """A workflow for the Flux2 model (text-to-image) with LoRA support."""

    def __init__(self, params: Flux2WorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: Flux2WorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load VAE
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        }, title="Load VAE")

        # Load CLIP (single CLIP for Flux2, not DualCLIP)
        clip = self.add_node("CLIPLoader", {
            "clip_name": self.params.model.clip_name,
            "type": self.params.model.clip_type,
            "device": self.params.model.clip_device
        }, title="Load CLIP")

        # Load UNET
        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

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

        # Encode prompt
        prompt_encode = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Positive Prompt)")

        # Apply FluxGuidance
        guidance = self.add_node("FluxGuidance", {
            "guidance": self.params.guidance,
            "conditioning": [prompt_encode, 0]
        }, title="FluxGuidance")

        # Sampler select
        sampler = self.add_node("KSamplerSelect", {
            "sampler_name": self.params.sampler
        }, title="KSamplerSelect")

        # Random noise
        noise = self.add_node("RandomNoise", {
            "noise_seed": self.params.seed
        }, title="RandomNoise")

        # Empty latent (Flux2-specific)
        latent = self.add_node("EmptyFlux2LatentImage", {
            "width": self.params.size[0],
            "height": self.params.size[1],
            "batch_size": self.params.batch_size
        }, title="Empty Flux 2 Latent")

        # Flux2 scheduler
        scheduler = self.add_node("Flux2Scheduler", {
            "steps": self.params.steps,
            "width": self.params.size[0],
            "height": self.params.size[1]
        }, title="Flux2Scheduler")

        # Basic guider (model goes directly in, no ModelSamplingFlux needed)
        basic_guider = self.add_node("BasicGuider", {
            "model": [current_model, 0],
            "conditioning": [guidance, 0]
        }, title="BasicGuider")

        # Advanced sampler
        sampler_advanced = self.add_node("SamplerCustomAdvanced", {
            "noise": [noise, 0],
            "guider": [basic_guider, 0],
            "sampler": [sampler, 0],
            "sigmas": [scheduler, 0],
            "latent_image": [latent, 0]
        }, title="SamplerCustomAdvanced")

        # Decode
        decode = self.add_node("VAEDecode", {
            "samples": [sampler_advanced, 0],
            "vae": [vae, 0]
        }, title="VAE Decode")

        # Save via websocket
        self.add_node("SaveImageWebsocket", {
            "images": [decode, 0]
        }, node_id="save_image_websocket_node")


@dataclass
class Flux2EditModel:
    """Configuration for a Flux2 edit model and its components"""
    clip_name: str = "mistral_3_small_flux2_fp8.safetensors"
    vae_name: str = "flux2-vae.safetensors"
    unet_name: str = "flux2_dev_fp8mixed.safetensors"
    loras: List[Lora] = None
    weight_dtype: str = "default"
    clip_type: str = "flux2"
    clip_device: str = "default"

    def __post_init__(self):
        if self.loras is None:
            self.loras = []

    @classmethod
    def default(cls) -> 'Flux2EditModel':
        """Returns the default Flux2 edit model configuration"""
        return cls()


@dataclass
class Flux2EditWorkflowParams:
    """Parameters for configuring a Flux2 edit workflow with reference images.

    Supports up to 10 reference images that are chained via ReferenceLatent nodes.
    """
    prompt: str
    model: Flux2EditModel
    reference_images: List[str] = field(default_factory=list)  # Paths to reference images (1-10)
    size: Size = Sizes.SQUARE_1K
    guidance: float = 3.5
    steps: int = 20
    sampler: str = "euler"
    megapixels: float = 1.0  # Target megapixels for scaling reference images
    upscale_method: str = "area"
    seed: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        if len(self.reference_images) > 10:
            raise ValueError("Maximum 10 reference images supported")
        if len(self.reference_images) == 0:
            raise ValueError("At least 1 reference image is required for edit workflow")


class Flux2EditWorkflow(ComfyWorkflow):
    """A workflow for the Flux2 model (image editing) with reference images and LoRA support.

    Reference images are chained via ReferenceLatent nodes which modify the conditioning.
    Supports 1-10 reference images.
    """

    def __init__(self, params: Flux2EditWorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: Flux2EditWorkflowParams object containing all generation parameters
        """
        super().__init__()
        self.params = params
        self.reference_image_node_ids: List[str] = []  # Track LoadImage node IDs for upload
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load VAE
        vae = self.add_node("VAELoader", {
            "vae_name": self.params.model.vae_name
        }, title="Load VAE")

        # Load CLIP
        clip = self.add_node("CLIPLoader", {
            "clip_name": self.params.model.clip_name,
            "type": self.params.model.clip_type,
            "device": self.params.model.clip_device
        }, title="Load CLIP")

        # Load UNET
        unet = self.add_node("UNETLoader", {
            "unet_name": self.params.model.unet_name,
            "weight_dtype": self.params.model.weight_dtype
        }, title="Load Diffusion Model")

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

        # Encode prompt
        prompt_encode = self.add_node("CLIPTextEncode", {
            "text": self.params.prompt,
            "clip": [current_clip, clip_output_index]
        }, title="CLIP Text Encode (Positive Prompt)")

        # Apply FluxGuidance
        guidance = self.add_node("FluxGuidance", {
            "guidance": self.params.guidance,
            "conditioning": [prompt_encode, 0]
        }, title="FluxGuidance")

        # Build reference image chains and ReferenceLatent nodes
        # Each reference image: LoadImage -> ImageScaleToTotalPixels -> VAEEncode -> ReferenceLatent
        # ReferenceLatent nodes chain: guidance -> ref1 -> ref2 -> ... -> refN -> BasicGuider
        current_conditioning = guidance

        for i, ref_image in enumerate(self.params.reference_images):
            # Load reference image
            load_image = self.add_node("LoadImage", {
                "image": ref_image
            }, title=f"Load Reference Image {i+1}")
            self.reference_image_node_ids.append(load_image)

            # Scale reference image
            scale_image = self.add_node("ImageScaleToTotalPixels", {
                "upscale_method": self.params.upscale_method,
                "megapixels": self.params.megapixels,
                "image": [load_image, 0]
            }, title=f"Scale Reference Image {i+1}")

            # Encode to latent
            vae_encode = self.add_node("VAEEncode", {
                "pixels": [scale_image, 0],
                "vae": [vae, 0]
            }, title=f"VAE Encode Reference {i+1}")

            # Chain ReferenceLatent - takes conditioning from previous and latent from this image
            ref_latent = self.add_node("ReferenceLatent", {
                "conditioning": [current_conditioning, 0],
                "latent": [vae_encode, 0]
            }, title=f"ReferenceLatent {i+1}")

            current_conditioning = ref_latent

        # Sampler select
        sampler = self.add_node("KSamplerSelect", {
            "sampler_name": self.params.sampler
        }, title="KSamplerSelect")

        # Random noise
        noise = self.add_node("RandomNoise", {
            "noise_seed": self.params.seed
        }, title="RandomNoise")

        # Empty latent (Flux2-specific)
        latent = self.add_node("EmptyFlux2LatentImage", {
            "width": self.params.size[0],
            "height": self.params.size[1],
            "batch_size": self.params.batch_size
        }, title="Empty Flux 2 Latent")

        # Flux2 scheduler
        scheduler = self.add_node("Flux2Scheduler", {
            "steps": self.params.steps,
            "width": self.params.size[0],
            "height": self.params.size[1]
        }, title="Flux2Scheduler")

        # Basic guider - conditioning comes from the last ReferenceLatent in the chain
        basic_guider = self.add_node("BasicGuider", {
            "model": [current_model, 0],
            "conditioning": [current_conditioning, 0]
        }, title="BasicGuider")

        # Advanced sampler
        sampler_advanced = self.add_node("SamplerCustomAdvanced", {
            "noise": [noise, 0],
            "guider": [basic_guider, 0],
            "sampler": [sampler, 0],
            "sigmas": [scheduler, 0],
            "latent_image": [latent, 0]
        }, title="SamplerCustomAdvanced")

        # Decode
        decode = self.add_node("VAEDecode", {
            "samples": [sampler_advanced, 0],
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
