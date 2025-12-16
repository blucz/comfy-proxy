"""SAM3 Segmentation Workflow for ComfyUI.

This workflow uses SAM3 (Segment Anything Model 3) to segment images based on
text prompts. It returns masks, bounding boxes, and confidence scores for
detected objects.
"""

from dataclasses import dataclass
from typing import Optional
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

    def __post_init__(self):
        if self.model is None:
            self.model = SAM3Model.default()


class SAM3Workflow(ComfyWorkflow):
    """A workflow for segmenting images using SAM3.

    This workflow takes an image and a text prompt, and returns segmentation
    masks, bounding boxes, and confidence scores for detected objects.

    Outputs:
        - masks: Segmentation masks for each detected object
        - visualization: Image with overlaid masks/boxes for preview
        - boxes: JSON string of bounding box coordinates
        - scores: JSON string of confidence scores
    """

    def __init__(self, params: SAM3WorkflowParams):
        """Create a workflow based on the provided parameters

        Args:
            params: SAM3WorkflowParams object containing all segmentation parameters
        """
        super().__init__()
        self.params = params
        self.input_image_node_id = None
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
        # SAM3Grounding widget order from workflow: [threshold, prompt, max_detections, multimask_output]
        sam3_grounding = self.add_node("SAM3Grounding", {
            "sam3_model": [sam3_model, 0],
            "image": [load_image, 0],
            "confidence_threshold": self.params.confidence_threshold,
            "prompt": self.params.prompt,
            "max_detections": self.params.max_detections,
            "multimask_output": self.params.multimask_output
        }, title="SAM3 Grounding")

        # Save visualization via websocket for streaming output
        self.add_node("SaveImageWebsocket", {
            "images": [sam3_grounding, 1]  # Output 1 is visualization
        }, node_id="save_image_websocket_node")

        # Store the grounding node ID for accessing other outputs
        self.sam3_grounding_node_id = sam3_grounding
