"""SAM3 Segmentation Workflow for ComfyUI.

This workflow uses SAM3 (Segment Anything Model 3) to segment images based on
text prompts. It returns masks, bounding boxes, and confidence scores for
detected objects.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from ..workflow import ComfyWorkflow


@dataclass
class SAM3Model:
    """Configuration for SAM3 model"""
    model_path: str = "models/sam3/sam3.pt"
    config_path: str = ""  # Optional config path

    @classmethod
    def default(cls) -> 'SAM3Model':
        """Returns the default SAM3 model configuration"""
        return cls()


@dataclass
class SAM3WorkflowParams:
    """Parameters for configuring a SAM3 segmentation workflow"""
    input_image: str  # Path/filename of image to segment
    prompt: str  # Text prompt describing what to segment (e.g., "animal", "person", "car")
    model: SAM3Model = None
    confidence_threshold: float = 0.2  # Minimum confidence score for detections
    max_detections: int = -1  # Maximum number of detections (-1 = unlimited)
    multimask_output: bool = False  # Whether to output multiple masks per detection
    output_prefix: str = "sam3_"  # Prefix for output filenames

    def __post_init__(self):
        if self.model is None:
            self.model = SAM3Model.default()


@dataclass
class SAM3Detection:
    """A single detection result from SAM3."""
    bbox: Dict[str, float]  # {x, y, width, height} in pixels
    score: float  # Confidence score
    mask_filename: str  # Filename of the saved mask image


@dataclass
class SAM3Result:
    """Complete result from SAM3 segmentation."""
    detections: List[SAM3Detection] = field(default_factory=list)
    visualization: bytes | None = None  # Preview image with overlaid masks
    original_width: int = 0
    original_height: int = 0


class SAM3Workflow(ComfyWorkflow):
    """A workflow for segmenting images using SAM3.

    This workflow takes an image and a text prompt, and returns segmentation
    masks, bounding boxes, and confidence scores for detected objects.

    SAM3Grounding node outputs:
        - Output 0: masks (MASK type - stacked mask tensor)
        - Output 1: visualization (IMAGE type - preview with masks overlaid)
        - Output 2: boxes (STRING type - JSON array of bboxes)
        - Output 3: scores (STRING type - JSON array of confidence scores)
    """

    def __init__(self, params: SAM3WorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: SAM3WorkflowParams object containing all segmentation parameters
        """
        super().__init__()
        self.params = params
        self.input_image_node_id = None
        self.sam3_grounding_node_id = None
        self.mask_save_node_id = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the internal workflow structure"""
        # Load SAM3 model
        sam3_model = self.add_node("LoadSAM3Model", {
            "model_path": self.params.model.model_path,
            "config_path": self.params.model.config_path
        }, title="Load SAM3 Model")

        # Load input image
        load_image = self.add_node("LoadImage", {
            "image": self.params.input_image
        }, title="Load Image")
        self.input_image_node_id = load_image

        # Run SAM3 grounding/segmentation
        # SAM3Grounding outputs: [0]=masks, [1]=visualization, [2]=boxes_json, [3]=scores_json
        # Note: The node uses "text_prompt" as the input name for the prompt
        sam3_grounding = self.add_node("SAM3Grounding", {
            "sam3_model": [sam3_model, 0],
            "image": [load_image, 0],
            "confidence_threshold": self.params.confidence_threshold,
            "text_prompt": self.params.prompt,
            "max_detections": self.params.max_detections,
        }, title="SAM3 Grounding")
        self.sam3_grounding_node_id = sam3_grounding

        # Convert masks to images for saving
        # MaskToImage converts MASK tensor to IMAGE format
        mask_to_image = self.add_node("MaskToImage", {
            "mask": [sam3_grounding, 0]  # Output 0 is masks
        }, title="Mask to Image")

        # Save masks to files (will save each mask in the batch as separate file)
        mask_save = self.add_node("SaveImage", {
            "images": [mask_to_image, 0],
            "filename_prefix": self.params.output_prefix + "mask"
        }, title="Save Masks")
        self.mask_save_node_id = mask_save

        # Save visualization via websocket for streaming output
        self.add_node("SaveImageWebsocket", {
            "images": [sam3_grounding, 1]  # Output 1 is visualization
        }, node_id="save_image_websocket_node")
