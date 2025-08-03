#!/usr/bin/env python3
"""
wan_api.py
==============

This module provides a RESTful API for the Wan 2.2 video generation models.
It mirrors the functionality of the Gradio interface contained in
``wan_web.py`` but exposes endpoints suitable for integration with other
applications (e.g. bots, chat services, or custom front‑ends).  All
generation parameters are surfaced through JSON request bodies and
responses include both status messages and, when appropriate, the path to
the generated video file on disk.

Example usage:

    python wan_api.py --model-dir ./models --host 0.0.0.0 --port 8000

Then interact with the API using curl or any HTTP client.  See the
``/docs`` endpoint for interactive Swagger documentation or refer to
the help tab in ``wan_web.py`` for detailed examples.

Memory Management:
    The API sets the environment variable ``PYTORCH_CUDA_ALLOC_CONF`` to
    ``expandable_segments:True`` by default to mitigate memory
    fragmentation in the CUDA allocator.  Additionally, the
    ``WanWebInterface`` object provides automatic fallback to CPU when
    encountering CUDA out‑of‑memory errors.
"""

import argparse
import base64
import io
import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

import torch

# Ensure local imports resolve correctly when running as a script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set allocator options to reduce memory fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from wan_web import WanWebInterface


class LoadModelRequest(BaseModel):
    model_path: str = Field(..., description="Absolute or relative path to the model directory")
    model_type: str = Field(..., description="One of 't2v', 'i2v' or 'ti2v'")
    t5_cpu: bool = Field(False, description="Place T5 encoder on CPU")
    init_on_cpu: bool = Field(True, description="Initialise model on CPU before moving to GPU")
    convert_model_dtype: bool = Field(False, description="Convert weights to bfloat16")
    t5_fsdp: bool = Field(False, description="Enable FSDP for T5")
    dit_fsdp: bool = Field(False, description="Enable FSDP for DiT")
    ulysses_size: int = Field(1, ge=1, description="Number of parallel workers/shards")
    force_cpu: bool = Field(False, description="Force model to run entirely on CPU")
    keep_alive: bool = Field(False, description="Unload model after each generation to free memory")
    load_8bit: bool = Field(False, description="Attempt to load models in 8‑bit precision (experimental)")


class UnloadModelResponse(BaseModel):
    status: str


class T2VRequest(BaseModel):
    prompt: str
    size: str = Field(..., description="Resolution string, e.g. '1280*720'")
    frame_num: int = Field(..., ge=5, description="Number of frames (4n+1)")
    steps: int = Field(50, ge=10, description="Number of diffusion steps")
    guide_scale: float = Field(5.0, ge=1.0, description="Guidance scale")
    seed: int = Field(-1, description="Random seed; -1 for random each run")
    negative_prompt: str = Field("", description="Negative prompt to avoid undesired artefacts")
    sample_solver: str = Field("unipc", description="Solver algorithm ('unipc' or 'dpm++')")
    sample_shift: float = Field(5.0, ge=0.0, description="Sampling shift parameter")
    offload_model: bool = Field(True, description="Offload model weights between inference passes")


class I2VRequest(T2VRequest):
    image: str = Field(..., description="Base64‑encoded PNG or JPEG image string")


class TI2VRequest(T2VRequest):
    image: Optional[str] = Field(None, description="Optional base64‑encoded image")
    use_prompt_extend: bool = Field(False, description="Whether to extend the prompt")
    prompt_extend_method: str = Field("local_qwen", description="Prompt extension method")
    prompt_extend_model: Optional[str] = Field(None, description="Path to custom prompt extension model")
    prompt_extend_target_lang: str = Field("en", description="Target language for prompt extension ('en' or 'zh')")


class GenerationResponse(BaseModel):
    status: str
    video_path: Optional[str] = None


def decode_image(b64_str: str) -> Image.Image:
    """Decode a base64 image into a PIL Image.

    Args:
        b64_str (str): Base64 encoded image string.

    Returns:
        PIL.Image.Image: The decoded image.

    Raises:
        HTTPException: If the image cannot be decoded.
    """
    try:
        image_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def create_app(model_dir: str, output_dir: str = "output", keep_alive: bool = False, load_8bit: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_dir (str): Directory containing Wan model checkpoints.

    Returns:
        FastAPI: Configured FastAPI instance.
    """
    app = FastAPI(title="Wan 2.2 Video Generation API", version="0.1.0")
    # Instantiate the WanWebInterface with output directory, keep‑alive and 8‑bit settings
    interface = WanWebInterface(
        model_dir=model_dir,
        output_dir=output_dir,
        keep_alive=keep_alive,
        load_8bit=load_8bit,
    )

    @app.get("/models", response_model=List[str])
    def list_models() -> List[str]:
        """Return the list of available model directories."""
        return interface.available_models

    @app.post("/load_model")
    def load_model(req: LoadModelRequest) -> JSONResponse:
        """Load a model into memory.

        You must call this before generating any videos.  If a model is
        already loaded it will be unloaded and replaced.
        """
        # Respect force_cpu flag
        if req.force_cpu:
            interface.device = "cpu"
        # Update keep‑alive and 8‑bit flags on the interface
        interface.keep_alive = req.keep_alive
        interface.load_8bit = req.load_8bit
        status = interface.load_model(
            model_path=req.model_path,
            model_type=req.model_type,
            t5_cpu=req.t5_cpu,
            init_on_cpu=req.init_on_cpu,
            convert_model_dtype=req.convert_model_dtype,
            t5_fsdp=req.t5_fsdp,
            dit_fsdp=req.dit_fsdp,
            ulysses_size=req.ulysses_size,
        )
        return JSONResponse(content={"status": status})

    @app.post("/unload_model", response_model=UnloadModelResponse)
    def unload_model() -> UnloadModelResponse:
        """Unload the currently loaded model and free GPU/CPU memory."""
        status = interface._unload_model()
        return UnloadModelResponse(status=status)

    @app.post("/t2v", response_model=GenerationResponse)
    def t2v_generation(req: T2VRequest) -> GenerationResponse:
        """Generate a video from text.

        Note: A model of type 't2v' must already be loaded via /load_model.
        """
        video_path, status = interface.generate_t2v(
            req.prompt,
            req.size,
            req.frame_num,
            req.steps,
            req.guide_scale,
            req.seed,
            req.negative_prompt,
            req.sample_solver,
            req.sample_shift,
            req.offload_model,
            False,  # convert_model_dtype not exposed via API
        )
        return GenerationResponse(status=status, video_path=video_path)

    @app.post("/i2v", response_model=GenerationResponse)
    def i2v_generation(req: I2VRequest) -> GenerationResponse:
        """Generate a video from an image and optional text prompt.

        Requires that a model of type 'i2v' is loaded.
        """
        # Decode image from base64
        pil_img = decode_image(req.image)
        video_path, status = interface.generate_i2v(
            req.prompt,
            pil_img,
            req.size,
            req.frame_num,
            req.steps,
            req.guide_scale,
            req.seed,
            req.negative_prompt,
            req.sample_solver,
            req.sample_shift,
            req.offload_model,
        )
        return GenerationResponse(status=status, video_path=video_path)

    @app.post("/ti2v", response_model=GenerationResponse)
    def ti2v_generation(req: TI2VRequest) -> GenerationResponse:
        """Generate a video from text and optionally an image.

        Requires that a model of type 'ti2v' is loaded.
        """
        pil_img: Optional[Image.Image] = None
        if req.image is not None:
            pil_img = decode_image(req.image)
        video_path, status = interface.generate_ti2v(
            req.prompt,
            pil_img,
            req.size,
            req.frame_num,
            req.steps,
            req.guide_scale,
            req.seed,
            req.negative_prompt,
            req.sample_solver,
            req.sample_shift,
            req.offload_model,
            req.use_prompt_extend,
            req.prompt_extend_method,
            req.prompt_extend_model,
            req.prompt_extend_target_lang,
        )
        return GenerationResponse(status=status, video_path=video_path)

    @app.get("/memory", response_model=Dict[str, Any])
    def memory_info() -> Dict[str, Any]:
        """Return current GPU memory usage statistics."""
        mem = interface.get_gpu_memory_info()
        return mem or {"note": "Running on CPU"}

    return app


def main() -> None:
    """Parse CLI arguments and run the API server."""
    parser = argparse.ArgumentParser(description="Wan 2.2 REST API server")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory containing Wan model folders",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host/IP to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory in which to save generated videos",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Unload the model after each generation to free memory",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Attempt to load models in 8‑bit precision (experimental)",
    )
    args = parser.parse_args()
    app = create_app(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        keep_alive=args.keep_alive,
        load_8bit=args.load_8bit,
    )
    # Only import uvicorn here to avoid overhead when imported as a module
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()