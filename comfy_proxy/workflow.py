from typing import Dict, Optional,Any
from dataclasses import dataclass

Size = tuple[int, int]  # (width, height)

@dataclass
class Lora:
    """Specification for a LoRA model and its weight"""
    name: str
    weight: float = 1.0


class Sizes:
    """Common image sizes for workflows"""
    # Square formats
    SQUARE_1K = (1024, 1024)  # 1:1
    SQUARE_2K = (2048, 2048)  # 1:1

    # Landscape formats
    LANDSCAPE_8_5 = (1216, 768)   # 8:5
    LANDSCAPE_4_3 = (1152, 896)   # 4:3
    LANDSCAPE_3_2 = (1216, 832)   # 3:2
    LANDSCAPE_7_5 = (1176, 840)   # 7:5
    LANDSCAPE_16_9 = (1344, 768)  # 16:9
    LANDSCAPE_21_9 = (1536, 640)  # 21:9
    LANDSCAPE_19_9 = (1472, 704)  # 19:9

    # Portrait formats
    PORTRAIT_3_4 = (896, 1152)    # 3:4
    PORTRAIT_2_3 = (832, 1216)    # 2:3
    PORTRAIT_5_7 = (840, 1176)    # 5:7
    PORTRAIT_9_16 = (768, 1344)   # 9:16
    PORTRAIT_9_21 = (640, 1536)   # 9:21
    PORTRAIT_5_8 = (768, 1216)    # 5:8
    PORTRAIT_9_19 = (704, 1472)   # 9:19

    # Video resolutions (16:9)
    VIDEO_480P_LANDSCAPE = (854, 480)   # 480p landscape
    VIDEO_720P_LANDSCAPE = (1280, 720)  # 720p landscape
    VIDEO_1080P_LANDSCAPE = (1920, 1080)  # 1080p landscape

@dataclass
class ComfyNode:
    class_type: str
    inputs: Dict[str, Any]
    meta: Optional[Dict[str, str]] = None

class ComfyWorkflow:
    """Base class for defining ComfyUI workflows.
    
    Provides methods to build node graphs that define image generation pipelines.
    Each node represents an operation (loading models, encoding prompts, etc).
    """
    
    def __init__(self):
        self.nodes = {}
        self._counter = 0

    def _next_id(self) -> str:
        """Generate next sequential node ID.
        
        Returns:
            String representation of next available node ID
        """
        self._counter += 1
        return str(self._counter)

    def add_node(self, class_type: str, inputs: Dict[str, Any], title: Optional[str] = None, node_id: Optional[str] = None) -> str:
        """Add a node to the workflow graph.
        
        Args:
            class_type: Type of ComfyUI node to create
            inputs: Dictionary of input values for the node
            title: Optional display title for the node
            node_id: Optional specific ID to use, otherwise auto-generated
            
        Returns:
            ID of the created node
        """
        node_id = node_id or self._next_id()
        meta = {"title": title or class_type} if title else None
        self.nodes[node_id] = ComfyNode(class_type, inputs, meta)
        return node_id

    def _update_image_reference(self, field_name: str, uploaded_filename: str, node_id: Optional[str] = None) -> None:
        """Update LoadImage nodes to reference an uploaded image.

        Args:
            field_name: The input field name to update (e.g. "image")
            uploaded_filename: The filename returned from upload
            node_id: Optional specific node ID to update (if None, updates all matching nodes)
        """
        if node_id:
            # Update specific node
            if node_id in self.nodes and self.nodes[node_id].class_type == "LoadImage":
                self.nodes[node_id].inputs[field_name] = uploaded_filename
        else:
            # Update all LoadImage nodes with matching field
            for nid, node in self.nodes.items():
                if node.class_type == "LoadImage" and field_name in node.inputs:
                    node.inputs[field_name] = uploaded_filename

    def to_dict(self) -> Dict:
        """Convert workflow to ComfyUI-compatible dictionary format.

        Returns:
            Dictionary representation of the complete workflow graph
        """
        result = {}
        for node_id, node in self.nodes.items():
            node_dict = {
                "class_type": node.class_type,
                "inputs": node.inputs
            }
            if node.meta:
                node_dict["_meta"] = node.meta
            result[node_id] = node_dict
        return result

