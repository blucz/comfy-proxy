import uuid
import json
import logging
import aiohttp
import asyncio
import traceback
from collections import deque
from typing import List, AsyncGenerator
# Force eager import of websockets submodules to avoid PyInstaller lazy-import issues
# Import connect directly from the client module to bypass lazy imports
import websockets
import websockets.client
import websockets.exceptions
import websockets.frames
import websockets.protocol
from websockets.client import connect as websockets_connect
from .workflow import Sizes, ComfyWorkflow
logging.getLogger('websockets').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _debug_response(resp, method_name: str):
    """Log detailed response info for debugging bundled app issues."""
    try:
        headers_dict = dict(resp.headers)
        logger.info(f"[{method_name}] Response status: {resp.status}")
        logger.info(f"[{method_name}] Response headers: {headers_dict}")
        logger.info(f"[{method_name}] Content-Type: {resp.headers.get('Content-Type', 'none')}")
        logger.info(f"[{method_name}] Content-Encoding: {resp.headers.get('Content-Encoding', 'none')}")
        logger.info(f"[{method_name}] Content-Length: {resp.headers.get('Content-Length', 'none')}")
    except Exception as e:
        logger.error(f"[{method_name}] Error logging response debug info: {e}")
from PIL import Image
from PIL.ExifTags import TAGS
import io
from PIL import PngImagePlugin
from datetime import datetime
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import random
import tempfile
import os


# Minimum inpaint region size - models degrade below this
INPAINT_MIN_REGION_SIZE = 192


def get_mask_bounding_box(mask_path: str) -> Optional[Tuple[int, int, int, int]]:
    """Get bounding box of the inpaint region in a mask.

    The mask is expected to have alpha channel where alpha < 128 means "inpaint here".

    Args:
        mask_path: Path to mask image (RGBA where alpha=0 means inpaint)

    Returns:
        Tuple of (left, top, right, bottom) or None if no inpaint region found
    """
    mask = Image.open(mask_path)
    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    # Get alpha channel
    _, _, _, alpha = mask.split()

    # Find pixels where alpha < 128 (inpaint region)
    import numpy as np
    alpha_arr = np.array(alpha)
    inpaint_pixels = alpha_arr < 128

    if not np.any(inpaint_pixels):
        return None

    # Get bounding box
    rows = np.any(inpaint_pixels, axis=1)
    cols = np.any(inpaint_pixels, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    return (int(left), int(top), int(right) + 1, int(bottom) + 1)


def preprocess_inpaint_images(
    image_path: str,
    mask_path: str,
    min_region_size: int = INPAINT_MIN_REGION_SIZE
) -> Tuple[str, str, float, Tuple[int, int]]:
    """Preprocess image and mask for inpainting, scaling up if the inpaint region is too small.

    Inpaint models degrade significantly when the generation area is smaller than ~512x512.
    This function scales up both image and mask if the masked region is below the minimum size.

    Args:
        image_path: Path to input image
        mask_path: Path to mask (RGBA where alpha=0 means inpaint)
        min_region_size: Minimum dimension for the inpaint region (default 512)

    Returns:
        Tuple of (processed_image_path, processed_mask_path, scale_factor, original_size)
        - If no scaling needed, returns original paths with scale_factor=1.0
        - If scaled, returns paths to temp files that should be cleaned up by caller
    """
    # Get original size
    original = Image.open(image_path)
    original_size = original.size

    # Get the inpaint region bounding box
    bbox = get_mask_bounding_box(mask_path)

    if bbox is None:
        # No inpaint region found, return as-is
        logger.info("[preprocess_inpaint] No inpaint region found in mask")
        return (image_path, mask_path, 1.0, original_size)

    left, top, right, bottom = bbox
    region_width = right - left
    region_height = bottom - top

    logger.info(f"[preprocess_inpaint] Inpaint region: {region_width}x{region_height} at ({left},{top})-({right},{bottom})")

    # Check if scaling is needed
    min_dimension = min(region_width, region_height)
    if min_dimension >= min_region_size:
        logger.info(f"[preprocess_inpaint] Region size {min_dimension}px >= {min_region_size}px, no scaling needed")
        return (image_path, mask_path, 1.0, original_size)

    # Calculate scale factor to bring the smaller dimension up to min_region_size
    scale_factor = min_region_size / min_dimension

    # Cap scale factor to avoid excessive scaling (max 4x)
    scale_factor = min(scale_factor, 4.0)

    logger.info(f"[preprocess_inpaint] Scaling up by {scale_factor:.2f}x (region {min_dimension}px < {min_region_size}px)")

    # Scale up the image
    new_width = int(original_size[0] * scale_factor)
    new_height = int(original_size[1] * scale_factor)

    original = original.convert("RGBA")
    scaled_image = original.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Scale up the mask
    mask = Image.open(mask_path).convert("RGBA")
    scaled_mask = mask.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save to temp files
    temp_dir = tempfile.gettempdir()

    scaled_image_path = os.path.join(temp_dir, f"inpaint_scaled_image_{os.getpid()}.png")
    scaled_mask_path = os.path.join(temp_dir, f"inpaint_scaled_mask_{os.getpid()}.png")

    scaled_image.save(scaled_image_path, format="PNG")
    scaled_mask.save(scaled_mask_path, format="PNG")

    logger.info(f"[preprocess_inpaint] Saved scaled images: {new_width}x{new_height}")

    return (scaled_image_path, scaled_mask_path, scale_factor, original_size)


def postprocess_inpaint_result(
    result_bytes: bytes,
    scale_factor: float,
    original_size: Tuple[int, int]
) -> bytes:
    """Postprocess inpaint result, scaling back down if the input was scaled up.

    Args:
        result_bytes: PNG bytes from inpaint workflow
        scale_factor: Scale factor used in preprocessing (1.0 = no scaling was done)
        original_size: Original (width, height) of the input image

    Returns:
        PNG bytes, scaled back to original size if needed
    """
    if scale_factor <= 1.0:
        return result_bytes

    logger.info(f"[postprocess_inpaint] Scaling result back down by {1/scale_factor:.2f}x to {original_size}")

    result = Image.open(io.BytesIO(result_bytes))

    # Scale back down to original size
    result_scaled = result.resize(original_size, Image.Resampling.LANCZOS)

    # Save to bytes
    output = io.BytesIO()
    result_scaled.save(output, format="PNG")
    return output.getvalue()


def cleanup_inpaint_temp_files(image_path: str, mask_path: str, scale_factor: float) -> None:
    """Clean up temp files created by preprocess_inpaint_images.

    Only removes files if they are in the temp directory and scale_factor > 1.0.
    """
    if scale_factor <= 1.0:
        return

    temp_dir = tempfile.gettempdir()
    for path in [image_path, mask_path]:
        if path.startswith(temp_dir) and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"[cleanup_inpaint] Removed temp file: {path}")
            except OSError as e:
                logger.warning(f"[cleanup_inpaint] Failed to remove temp file {path}: {e}")

class SingleComfy:
    """Client for interacting with a ComfyUI instance.
    
    Handles communication with ComfyUI server including prompt queueing,
    websocket connections, and image generation.
    """
    
    def __init__(self, addr: str):
        """Initialize Comfy client.
        
        Args:
            addr: ComfyUI server address in format 'host:port'
        """
        self.addr = addr
        self.client_id = str(uuid.uuid4())
        self.websocket = None

    async def queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a workflow prompt for execution on ComfyUI server.

        Args:
            prompt: Workflow prompt dictionary to execute

        Returns:
            Response dictionary containing prompt_id

        Raises:
            RuntimeError: If ComfyUI returns error or invalid response
            aiohttp.ClientError: On network/connection errors
        """
        # MARKER: comfy-proxy v2024.12.23 - if you see this, the new code is loaded
        logger.info(f"[COMFY-PROXY-V2] queue_prompt called for {self.addr}")
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p)
        logger.info(f"Sending prompt to ComfyUI at {self.addr}")
        logger.debug(f"Prompt data: {data}")
        
        try:
            headers = {"Content-Type": "application/json"}
            # Disable auto_decompress to avoid zlib errors with some server responses
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                async with session.post(f"http://{self.addr}/prompt", data=data, headers=headers) as resp:
                    await _debug_response(resp, "queue_prompt")
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"ComfyUI returned status {resp.status}: {error_text}")

                    # Read raw bytes first for debugging
                    raw_bytes = await resp.read()
                    logger.info(f"[queue_prompt] Raw response length: {len(raw_bytes)}, first 100 bytes: {raw_bytes[:100]}")

                    try:
                        response = json.loads(raw_bytes.decode('utf-8'))
                    except Exception as parse_err:
                        logger.error(f"[queue_prompt] JSON parse failed. Raw bytes: {raw_bytes[:500]}")
                        raise

                    if 'prompt_id' not in response:
                        logger.error(f"Unexpected response from ComfyUI: {response}")
                        raise RuntimeError(f"ComfyUI response missing prompt_id: {response}")

                    logger.debug(f"Received prompt_id: {response['prompt_id']}")
                    return response

        except aiohttp.ClientError as e:
            logger.error(f"[queue_prompt] aiohttp.ClientError: {type(e).__name__}: {str(e)}")
            logger.error(f"[queue_prompt] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to connect to ComfyUI: {str(e)}") from e
        except json.JSONDecodeError as e:
            logger.error(f"[queue_prompt] JSONDecodeError: {str(e)}")
            logger.error(f"[queue_prompt] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Invalid response from ComfyUI: {str(e)}") from e
        except Exception as e:
            logger.error(f"[queue_prompt] Unexpected {type(e).__name__}: {str(e)}")
            logger.error(f"[queue_prompt] Full traceback:\n{traceback.format_exc()}")
            raise
        
    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt.

        Args:
            prompt_id: The prompt ID to get history for

        Returns:
            History dictionary containing outputs

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                async with session.get(f"http://{self.addr}/history/{prompt_id}") as resp:
                    await _debug_response(resp, "get_history")
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"ComfyUI history fetch failed with status {resp.status}: {error_text}")

                    raw_bytes = await resp.read()
                    logger.info(f"[get_history] Raw response length: {len(raw_bytes)}, first 100 bytes: {raw_bytes[:100]}")

                    try:
                        history = json.loads(raw_bytes.decode('utf-8'))
                    except Exception as parse_err:
                        logger.error(f"[get_history] JSON parse failed. Raw bytes: {raw_bytes[:500]}")
                        raise

                    return history

        except aiohttp.ClientError as e:
            logger.error(f"[get_history] aiohttp.ClientError: {type(e).__name__}: {str(e)}")
            logger.error(f"[get_history] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to fetch history from ComfyUI: {str(e)}") from e
        except Exception as e:
            logger.error(f"[get_history] Unexpected {type(e).__name__}: {str(e)}")
            logger.error(f"[get_history] Full traceback:\n{traceback.format_exc()}")
            raise

    async def get_video(self, filename: str, subfolder: str = "") -> bytes:
        """Fetch a saved video file from ComfyUI server.

        Args:
            filename: The video filename to fetch
            subfolder: Optional subfolder path (e.g. "wan_i2v")

        Returns:
            Video file data as bytes

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            params = {"filename": filename, "type": "output"}
            if subfolder:
                params["subfolder"] = subfolder

            async with aiohttp.ClientSession(auto_decompress=False) as session:
                async with session.get(f"http://{self.addr}/view", params=params) as resp:
                    await _debug_response(resp, "get_video")
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"ComfyUI video fetch failed with status {resp.status}: {error_text}")

                    video_data = await resp.read()
                    logger.info(f"[get_video] Video fetched successfully: {filename}, size: {len(video_data)} bytes")
                    return video_data

        except aiohttp.ClientError as e:
            logger.error(f"[get_video] aiohttp.ClientError: {type(e).__name__}: {str(e)}")
            logger.error(f"[get_video] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to fetch video from ComfyUI: {str(e)}") from e
        except Exception as e:
            logger.error(f"[get_video] Unexpected {type(e).__name__}: {str(e)}")
            logger.error(f"[get_video] Full traceback:\n{traceback.format_exc()}")
            raise

    async def upload_image(self, image_path: str, image_type: str = "input", overwrite: bool = True) -> str:
        """Upload an image to ComfyUI server.

        Args:
            image_path: Path to the image file to upload
            image_type: Type of image - "input", "output", or "temp" (default: "input")
            overwrite: Whether to overwrite existing file (default: True)

        Returns:
            Uploaded filename as stored on server

        Raises:
            RuntimeError: If upload fails
            FileNotFoundError: If image_path doesn't exist
        """
        import os
        from pathlib import Path

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        filename = Path(image_path).name

        try:
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                with open(image_path, 'rb') as f:
                    form_data = aiohttp.FormData()
                    form_data.add_field('image', f, filename=filename, content_type='image/png')
                    form_data.add_field('type', image_type)
                    form_data.add_field('overwrite', str(overwrite).lower())

                    async with session.post(f"http://{self.addr}/upload/image", data=form_data) as resp:
                        await _debug_response(resp, "upload_image")
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise RuntimeError(f"ComfyUI image upload failed with status {resp.status}: {error_text}")

                        raw_bytes = await resp.read()
                        logger.info(f"[upload_image] Raw response length: {len(raw_bytes)}, first 100 bytes: {raw_bytes[:100]}")

                        try:
                            response = json.loads(raw_bytes.decode('utf-8'))
                        except Exception as parse_err:
                            logger.error(f"[upload_image] JSON parse failed. Raw bytes: {raw_bytes[:500]}")
                            raise

                        logger.debug(f"Image uploaded successfully: {response}")
                        return response.get('name', filename)

        except aiohttp.ClientError as e:
            logger.error(f"[upload_image] aiohttp.ClientError: {type(e).__name__}: {str(e)}")
            logger.error(f"[upload_image] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to upload image to ComfyUI: {str(e)}") from e
        except Exception as e:
            logger.error(f"[upload_image] Unexpected {type(e).__name__}: {str(e)}")
            logger.error(f"[upload_image] Full traceback:\n{traceback.format_exc()}")
            raise

    async def upload_video(self, video_path: str, overwrite: bool = True) -> str:
        """Upload a video to ComfyUI server.

        ComfyUI uses the same /upload/image endpoint for all files (images and videos).
        Videos are stored in the input directory and can be referenced by LoadVideo nodes.

        Args:
            video_path: Path to the video file to upload
            overwrite: Whether to overwrite existing file (default: True)

        Returns:
            Uploaded filename as stored on server

        Raises:
            RuntimeError: If upload fails
            FileNotFoundError: If video_path doesn't exist
        """
        import os
        from pathlib import Path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        filename = Path(video_path).name
        # Determine content type based on extension
        ext = Path(video_path).suffix.lower()
        content_type_map = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
        }
        content_type = content_type_map.get(ext, 'video/mp4')

        try:
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                with open(video_path, 'rb') as f:
                    form_data = aiohttp.FormData()
                    # ComfyUI uses 'image' field name for all file uploads
                    form_data.add_field('image', f, filename=filename, content_type=content_type)
                    form_data.add_field('type', 'input')
                    form_data.add_field('overwrite', str(overwrite).lower())

                    # ComfyUI uses /upload/image for all files (images and videos)
                    async with session.post(f"http://{self.addr}/upload/image", data=form_data) as resp:
                        await _debug_response(resp, "upload_video")
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise RuntimeError(f"ComfyUI video upload failed with status {resp.status}: {error_text}")

                        raw_bytes = await resp.read()
                        logger.info(f"[upload_video] Raw response length: {len(raw_bytes)}, first 100 bytes: {raw_bytes[:100]}")

                        try:
                            response = json.loads(raw_bytes.decode('utf-8'))
                        except Exception as parse_err:
                            logger.error(f"[upload_video] JSON parse failed. Raw bytes: {raw_bytes[:500]}")
                            raise

                        logger.debug(f"Video uploaded successfully: {response}")
                        return response.get('name', filename)

        except aiohttp.ClientError as e:
            logger.error(f"[upload_video] aiohttp.ClientError: {type(e).__name__}: {str(e)}")
            logger.error(f"[upload_video] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to upload video to ComfyUI: {str(e)}") from e
        except Exception as e:
            logger.error(f"[upload_video] Unexpected {type(e).__name__}: {str(e)}")
            logger.error(f"[upload_video] Full traceback:\n{traceback.format_exc()}")
            raise

    async def connect(self) -> None:
        """Establish websocket connection to ComfyUI server"""
        if not self.websocket or self.websocket.closed:
            logger.info(f"[connect] Establishing websocket connection to {self.addr}")
            # Use directly imported connect function to bypass lazy imports in bundled app
            self.websocket = await websockets_connect(
                f"ws://{self.addr}/ws?clientId={self.client_id}",
                max_size=None,  # No limit on message size
                max_queue=None,  # No limit on queue size
                compression=None  # Disable per-message deflate to avoid zlib issues in bundled apps
            )
            logger.info(f"[connect] Websocket connected to {self.addr}")

    async def disconnect(self) -> None:
        """Close websocket connection if open"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.websocket = None

    async def interrupt(self) -> bool:
        """Interrupt the currently executing prompt on this ComfyUI instance.

        Note: This does NOT disconnect the websocket. Use interrupt_and_clear() for
        a complete stop that also disconnects.

        Returns:
            True if interrupt was successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                async with session.post(f"http://{self.addr}/interrupt") as resp:
                    if resp.status == 200:
                        logger.info(f"Interrupted execution on {self.addr}")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.warning(f"Interrupt failed on {self.addr} with status {resp.status}: {error_text}")
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Network error interrupting ComfyUI at {self.addr}: {str(e)}")
            return False

    async def clear_queue(self) -> bool:
        """Clear all pending prompts from this ComfyUI instance's queue.

        Returns:
            True if queue was cleared successfully, False otherwise
        """
        try:
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                async with session.post(
                    f"http://{self.addr}/queue",
                    json={"clear": True}
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Cleared queue on {self.addr}")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.warning(f"Clear queue failed on {self.addr} with status {resp.status}: {error_text}")
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Network error clearing queue on ComfyUI at {self.addr}: {str(e)}")
            return False

    async def interrupt_and_clear(self) -> bool:
        """Interrupt current execution AND clear pending queue.

        This provides a complete stop - both the running generation and any queued work.
        The websocket is disconnected AFTER clearing the queue to ensure any workers
        that reconnected during the clear operation are also unblocked.

        Returns:
            True if both operations succeeded, False if either failed
        """
        interrupt_ok = await self.interrupt()
        clear_ok = await self.clear_queue()
        # Disconnect AFTER clearing to unblock any get_images() calls
        # This handles the race condition where a worker might submit a new prompt
        # between interrupt and clear_queue, then wait forever for that cleared prompt
        await self.disconnect()
        return interrupt_ok and clear_ok

    async def get_images(self, prompt_id: str) -> Dict[str, List[bytes]]:
        """Receive generated images or videos over websocket connection.

        Args:
            prompt_id: ID of prompt to receive images/videos for

        Returns:
            Dict mapping node IDs to lists of image/video data bytes
        """
        output_images = {}
        current_node = ""
        logger.info(f"[get_images] Waiting for images for prompt {prompt_id}")

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data['type'] == 'executing':
                        exec_data = data['data']
                        if exec_data.get('prompt_id') == prompt_id:
                            if exec_data['node'] is None:
                                logger.info(f"[get_images] Execution complete for prompt {prompt_id}")
                                break  # Execution is done
                            else:
                                current_node = exec_data['node']
                                logger.debug(f"[get_images] Executing node: {current_node}")
                else:
                    # Handle both image and video websocket nodes
                    if current_node in ['save_image_websocket_node', 'save_video_websocket_node']:
                        logger.info(f"[get_images] Received binary data from {current_node}, size: {len(message)} bytes")
                        images_output = output_images.get(current_node, [])
                        images_output.append(message[8:])
                        output_images[current_node] = images_output

            logger.info(f"[get_images] Returning {sum(len(v) for v in output_images.values())} images/videos")
            return output_images
        except Exception as e:
            logger.error(f"[get_images] Error receiving images: {type(e).__name__}: {str(e)}")
            logger.error(f"[get_images] Full traceback:\n{traceback.format_exc()}")
            raise

    async def generate(self, workflow: ComfyWorkflow, image_uploads: Optional[Dict[str, str]] = None) -> AsyncGenerator[Tuple[bytes, Dict[str, Any]], None]:
        """Generate images or videos from a workflow.

        Args:
            workflow: ComfyWorkflow instance defining the generation pipeline
            image_uploads: Optional dict mapping node input field names to local image paths to upload
                          e.g. {"image": "/path/to/input.jpg"} will upload the image and update
                          the workflow's LoadImage node to reference it

        Yields:
            Tuple of (data_bytes, workflow_dict) where:
                - data_bytes: Generated image/video data as bytes (PNG for images, MP4 for videos)
                - workflow_dict: The workflow dictionary used for generation (for ComfyUI drag-drop)

        Raises:
            RuntimeError: On ComfyUI errors
            websockets.WebSocketException: On websocket errors
        """
        # Handle image uploads if provided
        if image_uploads:
            for field_name, image_path in image_uploads.items():
                uploaded_name = await self.upload_image(image_path)
                # Update workflow to use uploaded image
                workflow._update_image_reference(field_name, uploaded_name)

        # generate the comfy json
        prompt_data = workflow.to_dict()

        # Check if this is a video workflow (has SaveVideo node)
        has_save_video = any(node.get('class_type') == 'SaveVideo' for node in prompt_data.values())

        # Queue the prompt first
        response = await self.queue_prompt(prompt_data)
        prompt_id = response['prompt_id']

        try:
            await self.connect()
            images = await self.get_images(prompt_id)

            # If this is a video workflow, fetch the video file from history
            if has_save_video:
                logger.info(f"Video workflow detected, fetching history for prompt {prompt_id}")
                history = await self.get_history(prompt_id)
                logger.debug(f"History response: {history}")

                if prompt_id in history:
                    outputs = history[prompt_id].get('outputs', {})
                    logger.debug(f"Outputs: {outputs}")

                    found_video = False
                    for node_id, output in outputs.items():
                        # Check for 'videos' key or 'images' with 'animated' flag
                        videos_list = output.get('videos', [])
                        if not videos_list and 'images' in output and output.get('animated'):
                            # SaveVideo node returns videos in 'images' key with 'animated' flag
                            videos_list = output['images']

                        if videos_list:
                            found_video = True
                            for video_info in videos_list:
                                filename = video_info['filename']
                                subfolder = video_info.get('subfolder', '')
                                logger.info(f"Fetching video: {filename} from subfolder: {subfolder}")
                                video_data = await self.get_video(filename, subfolder)
                                yield (video_data, prompt_data)

                    if not found_video:
                        logger.error(f"No videos found in outputs: {outputs}")
                        raise RuntimeError(f"No videos found in ComfyUI outputs for prompt {prompt_id}")
                else:
                    logger.error(f"Prompt {prompt_id} not found in history: {history}")
                    raise RuntimeError(f"Prompt {prompt_id} not found in history")
            else:
                # Regular image workflow
                for node_id in images:
                    for image_data in images[node_id]:
                        yield (image_data, prompt_data)
        except Exception as e:
            logger.error(f"[generate] Error during generation: {type(e).__name__}: {str(e)}")
            logger.error(f"[generate] Full traceback:\n{traceback.format_exc()}")
            await self.disconnect()  # Force reconnect on error
            raise


class Comfy:
    """Manages multiple Comfy instances for parallel image generation.
    
    Distributes generation workload across multiple ComfyUI instances,
    handling queuing and parallel execution.
    """
    """Manages multiple Comfy instances with parallel work distribution"""
    
    def __init__(self, addresses):
        """Initialize with Comfy instance addresses
        
        Args:
            addresses: Can be:
                - List of addresses (e.g. ["127.0.0.1:7821", "127.0.0.1:7822"])
                - Single address string (e.g. "127.0.0.1:7821")
                - Comma-separated addresses (e.g. "127.0.0.1:7821,127.0.0.1:7822") 
                - Address with port range (e.g. "127.0.0.1:7821-7824")
                Each address can optionally include a port range.
        """
        from .address import parse_addresses
        self.addresses = parse_addresses(addresses)
        self.instances = [SingleComfy(addr) for addr in self.addresses]
        self.queue = asyncio.Queue()
        self.instance_locks = [asyncio.Lock() for _ in self.instances]
        self.workers = []

    async def _worker(self, instance_id: int) -> None:
        """Worker process that handles generation requests for a Comfy instance"""
        try:
            while True:
                workflow, future = await self.queue.get()
                try:
                    async with self.instance_locks[instance_id]:
                        async for result in self.instances[instance_id].generate(workflow):
                            if not future.cancelled():
                                future.set_result(result)  # result is (bytes, workflow_dict) tuple
                            break  # Only yield first result for now
                except Exception as e:
                    if not future.cancelled():
                        future.set_exception(e)
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            return

    async def start(self) -> None:
        """Start worker tasks for all instances"""
        for i in range(len(self.instances)):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

    async def stop(self) -> None:
        """Stop all worker tasks and cleanup connections"""
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish cancelling
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()

        # Close all websocket connections
        for instance in self.instances:
            await instance.disconnect()

    async def interrupt_all(self) -> int:
        """Interrupt all currently executing prompts across all ComfyUI instances.

        Returns:
            Number of instances where interrupt succeeded
        """
        results = await asyncio.gather(
            *[instance.interrupt() for instance in self.instances],
            return_exceptions=True
        )
        return sum(1 for r in results if r is True)

    async def clear_all_queues(self) -> int:
        """Clear all pending prompts from all ComfyUI instances' queues.

        Returns:
            Number of instances where queue was cleared successfully
        """
        results = await asyncio.gather(
            *[instance.clear_queue() for instance in self.instances],
            return_exceptions=True
        )
        return sum(1 for r in results if r is True)

    async def interrupt_and_clear_all(self) -> int:
        """Interrupt all running prompts AND clear all pending queues.

        This provides a complete stop across all instances - both running
        generations and any queued work are cancelled.

        Returns:
            Number of instances where both operations succeeded
        """
        results = await asyncio.gather(
            *[instance.interrupt_and_clear() for instance in self.instances],
            return_exceptions=True
        )
        return sum(1 for r in results if r is True)

    async def generate(self, workflow: ComfyWorkflow, image_uploads: Optional[Dict[str, str]] = None) -> AsyncGenerator[Tuple[bytes, Dict[str, Any]], None]:
        """Generate images using available Comfy instances in parallel

        Args:
            workflow: The workflow to execute
            image_uploads: Optional dict mapping node input field names to local image paths to upload

        Yields:
            Tuple of (data_bytes, workflow_dict) where:
                - data_bytes: Generated image data as bytes (PNG format)
                - workflow_dict: The workflow dictionary used for generation (for ComfyUI drag-drop)
        """
        if not self.workers:
            await self.start()

        # If there are image uploads, handle them before queueing
        if image_uploads:
            # Use the first instance to upload images
            for field_name, image_path in image_uploads.items():
                uploaded_name = await self.instances[0].upload_image(image_path)
                workflow._update_image_reference(field_name, uploaded_name)

        future = asyncio.Future()
        await self.queue.put((workflow, future))
        result = await future
        yield result  # result is (bytes, workflow_dict) tuple


