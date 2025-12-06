import pytest
import asyncio
import random
import os
from comfy_proxy.comfy import Comfy
from comfy_proxy.workflows.wan_i2v import (
    WanI2VModel,
    WanI2VLightningModel,
    WanI2VWorkflowParams,
    WanI2VLightningWorkflowParams,
    WanI2VWorkflow,
    WanI2VLightningWorkflow
)
from comfy_proxy.workflow import Sizes

"""
# Commented out - too slow for regular testing
@pytest.mark.asyncio
async def test_wan_i2v_single_image() -> None:
    # Test Wan I2V with only a start image (using WanImageToVideo)
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")

    try:
        # Create a model configuration
        model = WanI2VModel.default()

        # Set up workflow parameters (only start_image provided)
        params = WanI2VWorkflowParams(
            prompt="She tilts her head, maintaining seductive eye contact with the viewer",
            start_image="start_frame.jpg",  # Placeholder, will be replaced by uploaded filename
            model=model,
            size=Sizes.VIDEO_480P_LANDSCAPE,
            frame_count=81,
            fps=16,
            seed=random.randint(0, 2**64)
        )

        # Create the workflow
        workflow = WanI2VWorkflow(params)

        # Get the path to start_frame.jpg in tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        start_image_path = os.path.join(test_dir, "start_frame.jpg")

        # Generate and save the video with image upload
        async for video_data, workflow_dict in comfy.generate(workflow, image_uploads={"image": start_image_path}):
            with open("wan_i2v_output.mp4", "wb") as f:
                f.write(video_data)
            break  # Only save the first video
    finally:
        # Properly cleanup connections and workers
        await comfy.stop()


@pytest.mark.asyncio
async def test_wan_i2v_first_last_frames() -> None:
    # Test Wan I2V with both start and end images (using WanFirstLastFrameToVideo)
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")

    try:
        # Create a model configuration
        model = WanI2VModel.default()

        # Set up workflow parameters (both start_image and end_image provided)
        params = WanI2VWorkflowParams(
            prompt="She tilts her head, maintaining seductive eye contact with the viewer",
            start_image="start_frame.jpg",  # Placeholder, will be replaced by uploaded filename
            end_image="end_frame.jpg",  # Placeholder, will be replaced by uploaded filename
            model=model,
            size=Sizes.VIDEO_480P_LANDSCAPE,
            frame_count=81,
            fps=16,
            seed=random.randint(0, 2**64)
        )

        # Create the workflow
        workflow = WanI2VWorkflow(params)

        # Get the paths to images in tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        start_image_path = os.path.join(test_dir, "start_frame.jpg")
        end_image_path = os.path.join(test_dir, "end_frame.jpg")

        # Upload both images by uploading them manually and updating specific nodes
        from comfy_proxy.comfy import SingleComfy
        single_comfy = SingleComfy("127.0.0.1:8188")

        # Upload start frame and update its node
        start_filename = await single_comfy.upload_image(start_image_path)
        workflow._update_image_reference("image", start_filename, workflow.start_image_node_id)

        # Upload end frame and update its node
        end_filename = await single_comfy.upload_image(end_image_path)
        workflow._update_image_reference("image", end_filename, workflow.end_image_node_id)

        # Generate video (no image_uploads param needed since we already uploaded)
        async for video_data, workflow_dict in comfy.generate(workflow):
            with open("wan_i2v_first_last_output.mp4", "wb") as f:
                f.write(video_data)
            break  # Only save the first video
    finally:
        # Properly cleanup connections and workers
        await comfy.stop()
"""


@pytest.mark.asyncio
async def test_wan_i2v_lightning_single_image() -> None:
    """Test Wan I2V Lightning with only a start image (4 steps, cfg=1.0)"""
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")

    try:
        # Create a Lightning model configuration
        model = WanI2VLightningModel.default()

        # Set up workflow parameters (steps=4, cfg=1.0 by default)
        params = WanI2VLightningWorkflowParams(
            prompt="She tilts her head, maintaining seductive eye contact with the viewer",
            start_image="start_frame.jpg",  # Placeholder, will be replaced by uploaded filename
            model=model,
            size=Sizes.VIDEO_480P_LANDSCAPE,
            frame_count=81,
            fps=16,
            seed=random.randint(0, 2**64)
        )

        # Create the workflow
        workflow = WanI2VLightningWorkflow(params)

        # Get the path to start_frame.jpg in tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        start_image_path = os.path.join(test_dir, "start_frame.jpg")

        # Generate and save the video with image upload
        async for video_data, workflow_dict in comfy.generate(workflow, image_uploads={"image": start_image_path}):
            with open("wan_i2v_lightning_output.mp4", "wb") as f:
                f.write(video_data)
            break  # Only save the first video
    finally:
        # Properly cleanup connections and workers
        await comfy.stop()


@pytest.mark.asyncio
async def test_wan_i2v_lightning_first_last() -> None:
    """Test Wan I2V Lightning with first and last frames (using same frame for both)"""
    # Initialize the Comfy client
    comfy = Comfy("127.0.0.1:8188")

    try:
        # Create a Lightning model configuration
        model = WanI2VLightningModel.default()

        # Set up workflow parameters with both start and end frames
        params = WanI2VLightningWorkflowParams(
            prompt="She tilts her head, maintaining seductive eye contact with the viewer",
            start_image="start_frame.jpg",  # Placeholder, will be replaced by uploaded filename
            end_image="start_frame.jpg",  # Using same frame for both start and end
            model=model,
            size=Sizes.VIDEO_480P_LANDSCAPE,
            frame_count=81,
            fps=16,
            seed=random.randint(0, 2**64)
        )

        # Create the workflow
        workflow = WanI2VLightningWorkflow(params)

        # Get the path to start_frame.jpg in tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        start_image_path = os.path.join(test_dir, "start_frame.jpg")

        # Upload both images (same image for both start and end)
        from comfy_proxy.comfy import SingleComfy
        single_comfy = SingleComfy("127.0.0.1:8188")

        # Upload start frame and update its node
        start_filename = await single_comfy.upload_image(start_image_path)
        workflow._update_image_reference("image", start_filename, workflow.start_image_node_id)

        # Use same uploaded file for end frame
        workflow._update_image_reference("image", start_filename, workflow.end_image_node_id)

        # Generate video
        async for video_data, workflow_dict in comfy.generate(workflow):
            with open("wan_i2v_lightning_first_last_output.mp4", "wb") as f:
                f.write(video_data)
            break  # Only save the first video
    finally:
        # Properly cleanup connections and workers
        await comfy.stop()


