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
        
    async def get_images(self, websocket: websockets.WebSocketClientProtocol, prompt_id: str) -> Dict[str, List[bytes]]:
        """Receive generated images over websocket connection.
        
        Args:
            websocket: Connected websocket client
            prompt_id: ID of prompt to receive images for
            
        Returns:
            Dict mapping node IDs to lists of image data bytes
        """
        try:
            output_images = {}
            current_node = ""
            
            async for message in websocket:
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
                    if current_node == 'save_image_websocket_node':
                        images_output = output_images.get(current_node, [])
                        images_output.append(message[8:])
                        output_images[current_node] = images_output

            return output_images
        finally:
            await websocket.close()

    async def generate(self, workflow: ComfyWorkflow) -> AsyncGenerator[bytes, None]:
        """Generate images from a workflow.
        
        Args:
            workflow: ComfyWorkflow instance defining the generation pipeline
            
        Yields:
            Generated image data as PNG bytes
            
        Raises:
            RuntimeError: On ComfyUI errors
            websockets.WebSocketException: On websocket errors
        """
        """Generate images from a prompt and return PNG bytes"""
        # generate the comfy json
        prompt_data = workflow.to_dict()
        
        # Queue the prompt first
        response = await self.queue_prompt(prompt_data)
        prompt_id = response['prompt_id']
        
        websocket = None
        try:
            # Connect to websocket and get images
            websocket = await websockets.connect(
                f"ws://{self.addr}/ws?clientId={self.client_id}",
                max_size=None,  # No limit on message size
                max_queue=None  # No limit on queue size
            )
            images = await self.get_images(websocket, prompt_id)
            for node_id in images:
                for image_data in images[node_id]:
                    yield image_data
        finally:
            if websocket and not websocket.closed:
                await websocket.close()


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
        """Stop all worker tasks"""
        for worker in self.workers:
            worker.cancel()
        self.workers.clear()

    async def generate(self, workflow: ComfyWorkflow) -> AsyncGenerator[bytes, None]:
        """Generate images using available Comfy instances in parallel
        
        Args:
            workflow: The workflow to exeGcute
            
        Yields:
            Generated image data as byte arrays containing .png format
        """
        if not self.workers:
            await self.start()

        future = asyncio.Future()
        await self.queue.put((workflow, future))
        result = await future
        yield result


