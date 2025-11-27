import uuid
import json
import logging
import aiohttp
import asyncio
from collections import deque
from typing import List, AsyncGenerator
import websockets
from .workflow import Sizes, ComfyWorkflow
logging.getLogger('websockets').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
from PIL import Image
from PIL.ExifTags import TAGS
import io
from PIL import PngImagePlugin
from datetime import datetime
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import random

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
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p)
        logger.info(f"Sending prompt to ComfyUI at {self.addr}")
        logger.debug(f"Prompt data: {data}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{self.addr}/prompt", data=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"ComfyUI returned status {resp.status}: {error_text}")
                    
                    response = await resp.json()
                    if 'prompt_id' not in response:
                        logger.error(f"Unexpected response from ComfyUI: {response}")
                        raise RuntimeError(f"ComfyUI response missing prompt_id: {response}")
                    
                    logger.debug(f"Received prompt_id: {response['prompt_id']}")
                    return response
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error connecting to ComfyUI at {self.addr}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to connect to ComfyUI: {str(e)}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from ComfyUI: {str(e)}", exc_info=True)
            raise RuntimeError(f"Invalid response from ComfyUI: {str(e)}") from e
        
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
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.addr}/history/{prompt_id}") as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"ComfyUI history fetch failed with status {resp.status}: {error_text}")

                    history = await resp.json()
                    return history

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching history from ComfyUI at {self.addr}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to fetch history from ComfyUI: {str(e)}") from e

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

            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.addr}/view", params=params) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"ComfyUI video fetch failed with status {resp.status}: {error_text}")

                    video_data = await resp.read()
                    logger.debug(f"Video fetched successfully: {filename}")
                    return video_data

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching video from ComfyUI at {self.addr}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to fetch video from ComfyUI: {str(e)}") from e

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
            async with aiohttp.ClientSession() as session:
                with open(image_path, 'rb') as f:
                    form_data = aiohttp.FormData()
                    form_data.add_field('image', f, filename=filename, content_type='image/png')
                    form_data.add_field('type', image_type)
                    form_data.add_field('overwrite', str(overwrite).lower())

                    async with session.post(f"http://{self.addr}/upload/image", data=form_data) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise RuntimeError(f"ComfyUI image upload failed with status {resp.status}: {error_text}")

                        response = await resp.json()
                        logger.debug(f"Image uploaded successfully: {response}")
                        return response.get('name', filename)

        except aiohttp.ClientError as e:
            logger.error(f"Network error uploading image to ComfyUI at {self.addr}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to upload image to ComfyUI: {str(e)}") from e

    async def connect(self) -> None:
        """Establish websocket connection to ComfyUI server"""
        if not self.websocket or self.websocket.closed:
            self.websocket = await websockets.connect(
                f"ws://{self.addr}/ws?clientId={self.client_id}",
                max_size=None,  # No limit on message size
                max_queue=None  # No limit on queue size
            )

    async def disconnect(self) -> None:
        """Close websocket connection if open"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.websocket = None

    async def interrupt(self) -> bool:
        """Interrupt the currently executing prompt on this ComfyUI instance.

        This also disconnects the websocket to unblock any pending get_images() calls.

        Returns:
            True if interrupt was successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{self.addr}/interrupt") as resp:
                    if resp.status == 200:
                        logger.info(f"Interrupted execution on {self.addr}")
                        # Disconnect websocket to unblock any waiting calls
                        await self.disconnect()
                        return True
                    else:
                        error_text = await resp.text()
                        logger.warning(f"Interrupt failed on {self.addr} with status {resp.status}: {error_text}")
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Network error interrupting ComfyUI at {self.addr}: {str(e)}")
            return False

    async def get_images(self, prompt_id: str) -> Dict[str, List[bytes]]:
        """Receive generated images or videos over websocket connection.

        Args:
            prompt_id: ID of prompt to receive images/videos for

        Returns:
            Dict mapping node IDs to lists of image/video data bytes
        """
        output_images = {}
        current_node = ""

        async for message in self.websocket:
            if isinstance(message, str):
                data = json.loads(message)
                if data['type'] == 'executing':
                    exec_data = data['data']
                    if exec_data.get('prompt_id') == prompt_id:
                        if exec_data['node'] is None:
                            break  # Execution is done
                        else:
                            current_node = exec_data['node']
            else:
                # Handle both image and video websocket nodes
                if current_node in ['save_image_websocket_node', 'save_video_websocket_node']:
                    images_output = output_images.get(current_node, [])
                    images_output.append(message[8:])
                    output_images[current_node] = images_output

        return output_images

    async def generate(self, workflow: ComfyWorkflow, image_uploads: Optional[Dict[str, str]] = None) -> AsyncGenerator[bytes, None]:
        """Generate images or videos from a workflow.

        Args:
            workflow: ComfyWorkflow instance defining the generation pipeline
            image_uploads: Optional dict mapping node input field names to local image paths to upload
                          e.g. {"image": "/path/to/input.jpg"} will upload the image and update
                          the workflow's LoadImage node to reference it

        Yields:
            Generated image/video data as bytes (PNG for images, MP4 for videos)

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
                                yield video_data

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
                        yield image_data
        except Exception as e:
            await self.disconnect()  # Force reconnect on error
            raise e


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
                        async for image in self.instances[instance_id].generate(workflow):
                            if not future.cancelled():
                                future.set_result(image)
                            break  # Only yield first image for now
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

    async def generate(self, workflow: ComfyWorkflow, image_uploads: Optional[Dict[str, str]] = None) -> AsyncGenerator[bytes, None]:
        """Generate images using available Comfy instances in parallel

        Args:
            workflow: The workflow to execute
            image_uploads: Optional dict mapping node input field names to local image paths to upload

        Yields:
            Generated image data as byte arrays containing .png format
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
        yield result


