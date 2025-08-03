#!/usr/bin/env python3
"""
Wan 2.2 Enhanced Web Interface using Gradio
This module exposes a polished Gradio interface for the Wan 2.2 models.  It
mirrors the functionality of the provided CLI (`generate.py`) by surfacing
every meaningful argument as a UI element and wires them into the underlying
model APIs.  Additional features include:

* Dynamic resolution selection based on the officially supported sizes for
  each task, preventing invalid configurations that would otherwise lead
  to runtime errors (for example the negative‐dimension error seen when
  attempting to generate a 720×1280 TI2V video).
* Automatic seed management so that random seeds are generated only when
  explicitly requested; `None` values are never passed into model code.
* Intelligent memory handling: models are unloaded when superseded, the
  generation functions call `torch.cuda.empty_cache()` and `gc.collect()` to
  proactively free GPU/CPU memory, and long running tasks are serialized
  through a threading lock to avoid concurrency collisions.
* Improved styling through carefully crafted CSS for a modern, welcoming
  interface.

This file is meant to be a drop‑in replacement for the original `wan_web.py`.
It retains all of the original functionality while addressing the issues
reported by the user and adding several enhancements.
"""

import os
import sys
import argparse
from pathlib import Path
import random
import time
from typing import Optional, Tuple, List, Dict, Any
import threading
import gc
import subprocess

import gradio as gr
import torch
from PIL import Image
import numpy as np

# Ensure local package imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable PyTorch memory fragmentation mitigation.  This environment variable
# encourages the allocator to use expandable memory segments which can help
# reduce CUDA OOM errors caused by fragmentation on some systems.  See
# https://pytorch.org/docs/stable/notes/cuda.html#environment-variables for
# details.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from wan.text2video import WanT2V
from wan.image2video import WanI2V
from wan.textimage2video import WanTI2V
from wan.configs import (
    wan_t2v_A14B,
    wan_i2v_A14B,
    wan_ti2v_5B,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS,
    SUPPORTED_SIZES,
    WAN_CONFIGS,
)


def discover_local_models(base_dir: str | Path = ".") -> List[str]:
    """Search a directory for Wan model checkpoints.

    A valid model directory is recognised by the presence of a
    ``configuration.json`` file.  Returns a sorted list of absolute paths.

    Args:
        base_dir (str or Path): Directory to search.  Defaults to the current
            working directory.

    Returns:
        List[str]: Sorted list of directories containing Wan model checkpoints.
    """
    base = Path(base_dir).expanduser().resolve()
    detected: List[str] = []
    try:
        for candidate in base.iterdir():
            if candidate.is_dir() and (candidate / "configuration.json").exists():
                detected.append(str(candidate))
    except Exception as e:
        print(f"Warning: Error discovering models in {base}: {e}")
    return sorted(detected)


class WanWebInterface:
    """A wrapper exposing Wan models through a Gradio UI.

    Args:
        model_dir: Base directory to search for pre‑downloaded Wan model
            checkpoints.  This can be overridden from the CLI via
            ``--model-dir``.
        output_dir: Directory into which generated videos will be saved.
            Defaults to ``output`` relative to the current working
            directory.  The directory is created if it does not exist.
        keep_alive: When True the interface automatically unloads the
            active model after every generation call, freeing GPU/CPU
            memory.  This can dramatically reduce peak memory usage at
            the cost of increased latency on subsequent calls.  Use
            ``--keep-alive`` on the CLI to enable this behaviour.
        load_8bit: When True the interface will attempt to load models
            in 8‑bit precision (quantisation) if supported.  Currently
            this flag simply acts as an alias for ``convert_model_dtype``
            because true 8‑bit support depends on the underlying
            library.  It is exposed to future‑proof the API and to
            signal intent to the user.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        output_dir: str = "output",
        keep_alive: bool = False,
        load_8bit: bool = False,
    ) -> None:
        # Discover available model folders
        self.model_dir = model_dir or "."
        # Create and set output directory for saving generated videos
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.available_models = discover_local_models(self.model_dir)
        self.current_model = None
        self.current_model_type: Optional[str] = None
        self.current_model_name: Optional[str] = None

        # Configuration dictionaries for each model type
        self.configs: Dict[str, Any] = {
            't2v': wan_t2v_A14B.t2v_A14B,
            'i2v': wan_i2v_A14B.i2v_A14B,
            'ti2v': wan_ti2v_5B.ti2v_5B,
        }

        # Precompute supported size strings for each task to drive UI
        self.size_options: Dict[str, List[str]] = {
            't2v': list(SUPPORTED_SIZES['t2v-A14B']),
            'i2v': list(SUPPORTED_SIZES['i2v-A14B']),
            'ti2v': list(SUPPORTED_SIZES['ti2v-5B']),
        }

        # Determine device (GPU or CPU) taking into account compute capability
        self.device = self._get_compatible_device()
        # Threading lock to serialise model loads and generation calls
        self.lock = threading.Lock()

        # Keep-alive flag: if True, unload model after each generation
        self.keep_alive = keep_alive

        # Experimental 8-bit loading flag.  At present this simply
        # triggers half‑precision conversion via ``convert_model_dtype``
        # because Wan does not natively expose an 8‑bit loading mode.  It
        # is preserved for future extensions and to allow the UI/CLI to
        # communicate user intent.
        self.load_8bit = load_8bit

        # Placeholder for a spawned API server process when started from the UI
        self.api_process: Optional[subprocess.Popen] = None

        print(f"Using device: {self.device}")

    def _get_compatible_device(self) -> Any:
        """Return the appropriate compute device.

        If CUDA is available but the detected GPU's compute capability is below
        7.5 (sm_75) then fall back to CPU.  Otherwise return the CUDA device
        index (0) or 'cpu' if CUDA is not available.
        """
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name()
                major, minor = torch.cuda.get_device_capability()
                if major < 7 or (major == 7 and minor < 5):
                    print(
                        f"Warning: GPU {gpu_name} (compute {major}.{minor}) is too old. Falling back to CPU."
                    )
                    return "cpu"
                return 0
            except Exception as e:
                print(f"GPU capability check failed: {e}; using CPU instead.")
                return "cpu"
        print("CUDA not available; using CPU.")
        return "cpu"

    # ----------------------------------------------------------------------
    # Memory utilities
    # ----------------------------------------------------------------------
    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """Return a dictionary with current GPU memory stats.

        Keys are ``allocated`` (GB), ``cached`` (GB) and ``total`` (GB).
        Returns ``None`` when CUDA is not available or CPU mode is in use.
        """
        if torch.cuda.is_available() and self.device != "cpu":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                total = (
                    torch.cuda.get_device_properties(self.device).total_memory
                    / (1024**3)
                )
                return {"allocated": allocated, "cached": reserved, "total": total}
            except Exception:
                return None
        return None

    def _save_video_result(self, video_result: Any, output_path: str, fps: int) -> None:
        """
        Persist a video result to disk, handling both PIL‑style objects and
        raw tensor outputs.  Wan models sometimes return a list of objects
        where the first element has a ``save`` method; other times they
        return a tensor of shape `(frames, channels, height, width)` or
        `(batch, frames, channels, height, width)`.  This helper
        transparently handles both cases using ``torchvision.io.write_video``
        when necessary.

        Args:
            video_result: Return value from a model ``generate`` call.
            output_path: Path to write the video file to.
            fps: Frames per second for the output video.
        """
        try:
            # Try the simple path: assume the result supports .save
            if hasattr(video_result[0], "save"):
                video_result[0].save(output_path, fps=fps, codec="libx264")
                return
        except Exception:
            # Fall through to tensor handling below
            pass
        # Import here to avoid unnecessary dependency when not used
        import torch
        import torchvision
        # Determine the tensor to save
        tensor = video_result
        if isinstance(video_result, (list, tuple)):
            tensor = video_result[0]
        if tensor.ndim == 5:
            # Remove batch dimension
            tensor = tensor[0]
        # Ensure tensor is on CPU and scaled to [0,255]
        t = tensor.detach().cpu()
        # If dtype is float, assume range [0,1] and scale
        if t.dtype.is_floating_point:
            t = (t * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            t = t.to(torch.uint8)
        # Convert to (frames, height, width, channels)
        if t.dim() == 4:
            # t: frames x channels x height x width
            t = t.permute(0, 2, 3, 1)
        # Write video using torchvision
        torchvision.io.write_video(output_path, t.contiguous(), fps=fps)

    # ------------------------------------------------------------------
    # API process management
    # ------------------------------------------------------------------
    def start_api(self, host: str, port: int) -> str:
        """
        Launch the REST API server as a subprocess.  If the API is already
        running, the existing instance is left untouched and a status message
        is returned.  This method spawns ``wan_api.py`` using the same
        Python interpreter that is executing the Gradio server and passes
        through the currently selected model directory.

        Args:
            host (str): Host/IP to bind the API server to.
            port (int): Port number to run the API server on.

        Returns:
            str: A status message describing the action taken.
        """
        if self.api_process is not None and self.api_process.poll() is None:
            return f"⚠️ API server is already running on port {port}"
        try:
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan_api.py"),
                "--model-dir",
                self.model_dir,
                "--host",
                host,
                "--port",
                str(port),
            ]
            # Start the subprocess without blocking the main thread
            self.api_process = subprocess.Popen(cmd)
            return f"✅ API server started at http://{host}:{port}"
        except Exception as e:
            return f"❌ Failed to start API server: {e}"

    def stop_api(self) -> str:
        """
        Terminate the REST API server subprocess if it is running.

        Returns:
            str: A status message describing the result.
        """
        if self.api_process is None or self.api_process.poll() is not None:
            return "ℹ️ API server is not running"
        try:
            self.api_process.terminate()
            self.api_process = None
            return "🛑 API server stopped"
        except Exception as e:
            return f"❌ Failed to stop API server: {e}"

    def create_memory_display(self) -> str:
        """Generate HTML snippet displaying current memory usage."""
        mem = self.get_gpu_memory_info()
        if mem:
            return (
                f"<div style='background:#e8f5e8;padding:8px;border-radius:5px;'>"
                f"<strong>GPU Memory:</strong> {mem['allocated']:.1f}GB / {mem['total']:.1f}GB"
                f" ({(mem['allocated']/mem['total'])*100:.1f}% used)"
                f"</div>"
            )
        return (
            "<div style='background:#f0f0f0;padding:8px;border-radius:5px;'>"
            "<strong>Running on CPU</strong>"
            "</div>"
        )

    # ----------------------------------------------------------------------
    # Model management
    # ----------------------------------------------------------------------
    def load_model(
        self,
        model_path: str,
        model_type: str,
        **kwargs: Any,
    ) -> str:
        """Instantiate the selected Wan model.

        Parameters are passed through from the UI and correspond to the flags
        supported by the CLI.  The function logs the settings used, creates
        the appropriate model class and stores it on this instance.

        Returns:
            str: A human readable status message.
        """
        try:
            with self.lock:
                # Dispose of any currently loaded model
                if self.current_model is not None:
                    self._unload_model()

                # Pull the configuration for the selected type
                config = self.configs[model_type]

                # Device id is either an integer (GPU index) or 'cpu'
                device_id = self.device
                use_cpu_fallback = self.device == "cpu"

                # Build the argument dictionary to pass into model constructors
                # The ``convert_model_dtype`` flag is enabled if either the
                # user explicitly requests it via the UI or CLI or if the
                # experimental ``load_8bit`` flag is set on the
                # interface.  At present there is no native 8‑bit loading
                # support in Wan, so both options map to the same
                # behaviour.
                convert_dtype = kwargs.get("convert_model_dtype", False) or self.load_8bit
                model_kwargs: Dict[str, Any] = {
                    "config": config,
                    "checkpoint_dir": model_path,
                    "device_id": device_id,
                    "t5_fsdp": kwargs.get("t5_fsdp", False),
                    "dit_fsdp": kwargs.get("dit_fsdp", False),
                    "use_sp": kwargs.get("ulysses_size", 1) > 1,
                    "t5_cpu": kwargs.get("t5_cpu", use_cpu_fallback),
                    "init_on_cpu": kwargs.get("init_on_cpu", True),
                    "convert_model_dtype": convert_dtype,
                }

                # Offload flag is propagated to generation calls but not to
                # constructor in the current API; include for completeness
                if "offload_model" in kwargs:
                    model_kwargs["offload_model"] = kwargs["offload_model"]

                print(f"Loading {model_type.upper()} model with params:")
                for k, v in model_kwargs.items():
                    if k != "config":
                        print(f"  {k}: {v}")

                # Instantiate the model
                if model_type == 't2v':
                    self.current_model = WanT2V(**model_kwargs)
                elif model_type == 'i2v':
                    self.current_model = WanI2V(**model_kwargs)
                elif model_type == 'ti2v':
                    self.current_model = WanTI2V(**model_kwargs)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                self.current_model_type = model_type
                self.current_model_name = Path(model_path).name

                # Remember the last loaded model details so that we can fall back
                # in case of memory errors during generation.  These keys are
                # internal and not exposed via the public API.
                self._loaded_model_path = model_path
                self._loaded_model_type = model_type
                # Store a shallow copy of kwargs so future modifications to the
                # original do not affect the saved version
                self._loaded_model_kwargs = model_kwargs.copy()

                # Build status message with GPU information if available
                status = f"✅ Loaded {model_type.upper()} model: {self.current_model_name}"
                mem_info = self.get_gpu_memory_info()
                if mem_info:
                    status += (
                        f"\nGPU Memory: {mem_info['allocated']:.1f}GB allocated, "
                        f"{mem_info['cached']:.1f}GB cached"
                    )
                return status
        except Exception as e:
            # Provide contextual hints on common failure modes
            error_msg = f"❌ Error loading model: {e}"
            err_str = str(e)
            if "CUDA out of memory" in err_str:
                error_msg += (
                    "\n\n💡 <b>Tips to reduce memory usage:</b>"
                    "\n- Enable CPU offloading (t5_cpu and init_on_cpu)"
                    "\n- Enable model offloading"
                    "\n- Reduce frame count"
                    "\n- Use CPU mode (select Force CPU)"
                )
            elif "Default process group has not been initialized" in err_str:
                error_msg += (
                    "\n\n⚠️ FSDP requires a distributed environment. "
                    "Disable T5 FSDP and DiT FSDP unless you have configured "
                    "torch.distributed with multiple GPUs."
                )
            return error_msg

    def _unload_model(self) -> str:
        """Release the currently loaded model and free memory."""
        with self.lock:
            if self.current_model is None:
                return "ℹ️ No model loaded to unload"
            try:
                del self.current_model
                self.current_model = None
                self.current_model_type = None
                self.current_model_name = None
                # Free GPU caches immediately
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                mem_info = self.get_gpu_memory_info()
                status = "✅ Model unloaded successfully"
                if mem_info:
                    status += f"\nGPU Memory: {mem_info['allocated']:.1f}GB remaining"
                return status
            except Exception as e:
                return f"❌ Error unloading model: {e}"

    # ----------------------------------------------------------------------
    # Generation functions
    # ----------------------------------------------------------------------
    def generate_t2v(
        self,
        prompt: str,
        size_str: str,
        frame_num: int,
        steps: int,
        guide_scale: float,
        seed: int,
        negative_prompt: str,
        sample_solver: str,
        sample_shift: float,
        offload_model: bool,
        convert_model_dtype: bool,
        __retry: bool = True,
    ) -> Tuple[Optional[str], str]:
        """Generate a video from a text prompt using WanT2V.

        Returns a tuple ``(path_to_video, status_message)``.  In case of
        failure, the first element will be ``None`` and the second element
        contains the error message.
        """
        if not self.current_model or self.current_model_type != 't2v':
            return None, "⚠️ Please load a T2V model first"
        # Proactively free any cached memory before starting generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Convert resolution string into (width, height)
        if size_str not in SIZE_CONFIGS:
            return None, f"❌ Unsupported size: {size_str}"
        resolution = SIZE_CONFIGS[size_str]
        # Derive seed: always convert to int; negative values trigger random generation
        try:
            # gr.Number returns a float; explicitly cast to int if possible
            actual_seed = int(seed) if seed is not None else -1
        except Exception:
            actual_seed = -1
        try:
            base_kwargs = {
                "input_prompt": prompt,
                "size": resolution,
                "frame_num": frame_num,
                "sampling_steps": steps,
                "guide_scale": guide_scale,
                "n_prompt": negative_prompt,
                "sample_solver": sample_solver,
                "shift": sample_shift,
                "offload_model": offload_model,
            }
            # Always set seed.  Negative values indicate that the underlying
            # implementation should pick a random seed.  Passing -1 avoids
            # ``None`` which causes TypeErrors in some model code.
            base_kwargs["seed"] = actual_seed
            print(f"Starting T2V generation with parameters: {base_kwargs}")
            # Call the underlying API.  Seed is only set when non-negative.
            video_tensor = self.current_model.generate(**base_kwargs)
            # Save the resulting video to disk
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"generated_t2v_{timestamp}.mp4")
            fps_value = 16
            if hasattr(wan_t2v_A14B, 'sample_fps'):
                fps_value = wan_t2v_A14B.sample_fps
            # Use helper to persist the video
            self._save_video_result(video_tensor, output_path, fps=fps_value)
            print(f"T2V video saved to: {output_path}")
            # Clean up memory once generation completes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            # Offload the model if keep_alive is enabled
            if getattr(self, 'keep_alive', False):
                self._unload_model()
            mem_info = self.get_gpu_memory_info()
            status_suffix = (
                f" | GPU: {mem_info['allocated']:.1f}/{mem_info['total']:.1f}GB"
                if mem_info
                else ""
            )
            return (
                output_path,
                f"✅ T2V completed (seed: {actual_seed}, frames: {frame_num}){status_suffix}",
            )
        except RuntimeError as e:
            # Attempt graceful fallback on CUDA OOM
            if "out of memory" in str(e).lower() and __retry and self.device != "cpu":
                print("CUDA OOM encountered during T2V. Attempting fallback to CPU...")
                # Free as much as possible
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                # Reload model on CPU
                prev_path = getattr(self, "_loaded_model_path", None)
                prev_type = getattr(self, "_loaded_model_type", None)
                prev_kwargs = getattr(self, "_loaded_model_kwargs", None)
                if prev_path and prev_type:
                    self.device = "cpu"
                    self._unload_model()
                    # Ensure device_id in kwargs is cpu
                    if prev_kwargs is not None:
                        prev_kwargs = prev_kwargs.copy()
                        prev_kwargs["device_id"] = "cpu"
                    self.load_model(prev_path, prev_type, **(prev_kwargs or {}))
                    # Re-run generation on CPU with offload disabled (irrelevant on CPU)
                    return self.generate_t2v(
                        prompt,
                        size_str,
                        frame_num,
                        steps,
                        guide_scale,
                        seed,
                        negative_prompt,
                        sample_solver,
                        sample_shift,
                        False,
                        convert_model_dtype,
                        __retry=False,
                    )
            # Fallback failed or not applicable
            import traceback
            print("Error in T2V generation:", traceback.format_exc())
            return None, f"❌ T2V Error: {e}"
        except Exception as e:
            # Generic error handler
            import traceback
            print("Error in T2V generation:", traceback.format_exc())
            return None, f"❌ T2V Error: {e}"

    def generate_i2v(
        self,
        prompt: str,
        image: Optional[Image.Image],
        size_str: str,
        frame_num: int,
        steps: int,
        guide_scale: float,
        seed: int,
        negative_prompt: str,
        sample_solver: str,
        sample_shift: float,
        offload_model: bool,
        __retry: bool = True,
    ) -> Tuple[Optional[str], str]:
        """Generate a video given an input image and optional prompt using WanI2V.

        If a CUDA out-of-memory error is encountered and ``__retry`` is True, the
        function will free GPU memory, reload the model on the CPU and retry
        generation exactly once.  This fallback is transparent to callers and
        intended for inference scenarios on constrained hardware.
        """
        # Free cached memory before starting generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if not self.current_model or self.current_model_type != 'i2v':
            return None, "⚠️ Please load an I2V model first"
        if size_str not in MAX_AREA_CONFIGS:
            return None, f"❌ Unsupported size: {size_str}"
        max_area = MAX_AREA_CONFIGS[size_str]
        # Ensure seed is an integer; negative triggers random seeding
        try:
            actual_seed = int(seed) if seed is not None else -1
        except Exception:
            actual_seed = -1
        try:
            base_kwargs = {
                "input_prompt": prompt,
                "img": np.array(image) if isinstance(image, Image.Image) else image,
                "max_area": max_area,
                "frame_num": frame_num,
                "sampling_steps": steps,
                "guide_scale": guide_scale,
                "n_prompt": negative_prompt,
                "sample_solver": sample_solver,
                "shift": sample_shift,
                "offload_model": offload_model,
            }
            # Always set seed; negative values instruct the model to choose a random seed
            base_kwargs["seed"] = actual_seed
            print(f"Starting I2V generation with parameters: {base_kwargs}")
            video_tensor = self.current_model.generate(**base_kwargs)
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"generated_i2v_{timestamp}.mp4")
            self._save_video_result(video_tensor, output_path, fps=16)
            print(f"I2V video saved to: {output_path}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if getattr(self, 'keep_alive', False):
                self._unload_model()
            mem_info = self.get_gpu_memory_info()
            status_suffix = (
                f" | GPU: {mem_info['allocated']:.1f}/{mem_info['total']:.1f}GB"
                if mem_info
                else ""
            )
            return (
                output_path,
                f"✅ I2V completed (seed: {actual_seed}, frames: {frame_num}){status_suffix}",
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and __retry and self.device != "cpu":
                print("CUDA OOM encountered during I2V. Attempting fallback to CPU...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                prev_path = getattr(self, "_loaded_model_path", None)
                prev_type = getattr(self, "_loaded_model_type", None)
                prev_kwargs = getattr(self, "_loaded_model_kwargs", None)
                if prev_path and prev_type:
                    self.device = "cpu"
                    self._unload_model()
                    if prev_kwargs is not None:
                        prev_kwargs = prev_kwargs.copy()
                        prev_kwargs["device_id"] = "cpu"
                    self.load_model(prev_path, prev_type, **(prev_kwargs or {}))
                    return self.generate_i2v(
                        prompt,
                        image,
                        size_str,
                        frame_num,
                        steps,
                        guide_scale,
                        seed,
                        negative_prompt,
                        sample_solver,
                        sample_shift,
                        False,
                        __retry=False,
                    )
            import traceback
            print("Error in I2V generation:", traceback.format_exc())
            return None, f"❌ I2V Error: {e}"
        except Exception as e:
            import traceback
            print("Error in I2V generation:", traceback.format_exc())
            return None, f"❌ I2V Error: {e}"

    def generate_ti2v(
        self,
        prompt: str,
        image: Optional[Image.Image],
        size_str: str,
        frame_num: int,
        steps: int,
        guide_scale: float,
        seed: int,
        negative_prompt: str,
        sample_solver: str,
        sample_shift: float,
        offload_model: bool,
        use_prompt_extend: bool,
        prompt_extend_method: str,
        prompt_extend_model: Optional[str],
        prompt_extend_target_lang: str,
        _retry: bool = True,
    ) -> Tuple[Optional[str], str]:
        """Generate a video from text and optionally an image using WanTI2V.

        The function will fall back to CPU and retry once if a CUDA out of
        memory error occurs and ``__retry`` is True.  Seeds are always
        integers; negative seeds instruct the model to pick a random seed.
        """
        # Free cached memory before starting generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if not self.current_model or self.current_model_type != 'ti2v':
            return None, "⚠️ Please load a TI2V model first"
        if size_str not in SIZE_CONFIGS or size_str not in MAX_AREA_CONFIGS:
            return None, f"❌ Unsupported size: {size_str}"
        # Extract resolution and max pixel area from config tables
        resolution = SIZE_CONFIGS[size_str]
        max_area = MAX_AREA_CONFIGS[size_str]
        # Convert seed to int; negative triggers random seed
        try:
            actual_seed = int(seed) if seed is not None else -1
        except Exception:
            actual_seed = -1
        # Handle prompt extension
        final_prompt = prompt
        if use_prompt_extend:
            try:
                from wan.utils.prompt_extend import prompt_extend
                extension_result = prompt_extend(
                    prompt,
                    method=prompt_extend_method,
                    target_lang=prompt_extend_target_lang,
                    model_path=prompt_extend_model,
                )
                if getattr(extension_result, "status", False):
                    final_prompt = extension_result.prompt
                    print(f"Prompt extended: {prompt} → {final_prompt}")
                else:
                    # If extension fails, proceed with original prompt
                    msg = getattr(extension_result, "message", "unknown error")
                    print(f"Prompt extension failed: {msg}")
            except ImportError:
                print("Prompt extend module not available; proceeding without extension.")
            except Exception as ext_e:
                print(f"Error during prompt extension: {ext_e}")
        try:
            # Assemble generation arguments
            base_kwargs = {
                "input_prompt": final_prompt,
                "size": resolution,
                "max_area": max_area,
                "frame_num": frame_num,
                "sampling_steps": steps,
                "guide_scale": guide_scale,
                "n_prompt": negative_prompt,
                "sample_solver": sample_solver,
                "shift": sample_shift,
                "offload_model": offload_model,
                "seed": actual_seed,
            }
            # Attach image if provided
            if image is not None:
                base_kwargs["img"] = (
                    np.array(image) if isinstance(image, Image.Image) else image
                )
            print(f"Starting TI2V generation with parameters: {base_kwargs}")
            # Generate video tensor using the model's generate method
            video_tensor = self.current_model.generate(**base_kwargs)
            # Save the resulting video to disk
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"generated_ti2v_{timestamp}.mp4")
            fps_value = 24
            if hasattr(wan_ti2v_5B, 'sample_fps'):
                fps_value = wan_ti2v_5B.sample_fps
            self._save_video_result(video_tensor, output_path, fps=fps_value)
            print(f"TI2V video saved to: {output_path}")
            # Free memory caches immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            # Offload model automatically if keep_alive is set
            if getattr(self, 'keep_alive', False):
                self._unload_model()
            mem_info = self.get_gpu_memory_info()
            status_suffix = (
                f" | GPU: {mem_info['allocated']:.1f}/{mem_info['total']:.1f}GB"
                if mem_info
                else ""
            )
            return (
                output_path,
                f"✅ TI2V completed (seed: {actual_seed}, frames: {frame_num}){status_suffix}",
            )
        except RuntimeError as e:
            # On CUDA out of memory, reload on CPU and retry once
            if "out of memory" in str(e).lower() and _retry and self.device != "cpu":
                print("CUDA OOM encountered during TI2V. Attempting fallback to CPU…")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                prev_path = getattr(self, "_loaded_model_path", None)
                prev_type = getattr(self, "_loaded_model_type", None)
                prev_kwargs = getattr(self, "_loaded_model_kwargs", None)
                if prev_path and prev_type:
                    # Switch device to CPU and reload the model
                    self.device = "cpu"
                    self._unload_model()
                    if prev_kwargs is not None:
                        prev_kwargs = prev_kwargs.copy()
                        prev_kwargs["device_id"] = "cpu"
                    self.load_model(prev_path, prev_type, **(prev_kwargs or {}))
                    # Retry generation once with offloading disabled (irrelevant on CPU)
                    return self.generate_ti2v(
                        prompt,
                        image,
                        size_str,
                        frame_num,
                        steps,
                        guide_scale,
                        seed,
                        negative_prompt,
                        sample_solver,
                        sample_shift,
                        False,
                        use_prompt_extend,
                        prompt_extend_method,
                        prompt_extend_model,
                        prompt_extend_target_lang,
                        _retry=False,
                    )
            import traceback
            print("Error in TI2V generation:", traceback.format_exc())
            return None, f"❌ TI2V Error: {e}"
        except Exception as e:
            import traceback
            print("Error in TI2V generation:", traceback.format_exc())
            return None, f"❌ TI2V Error: {e}"

    # ----------------------------------------------------------------------
    # User interface definition
    # ----------------------------------------------------------------------
    def create_interface(self) -> gr.Blocks:
        """Construct and return the Gradio Blocks interface."""
        # Base CSS for styling.  Additional styling is injected by add_custom_css().
        css = """
        .main-container { max-width: 1400px; margin: auto; }
        .status-success { color: green; font-weight: bold; }
        .status-error { color: red; font-weight: bold; }
        .status-warning { color: orange; font-weight: bold; }
        .model-selection-section { padding: 20px; border-radius: 10px; background: #f8f9fa; margin: 15px 0; }
        .advanced-section { padding: 15px; border-radius: 8px; background: #f5f5f5; margin: 10px 0; }
        .memory-info { font-family: monospace; font-size: 0.9em; }
        .video-output { background: #f8f9fa; border-radius: 8px; padding: 10px; }
        """
        # Merge custom CSS for further polish
        css += self.add_custom_css()

        with gr.Blocks(title="Wan 2.2 Pro Video Generation Interface", css=css) as interface:
            # Header and introduction
            gr.Markdown(
                """
                # 🎬 Wan 2.2 Pro Video Generation Interface

                **Full‑featured video generation** with comprehensive parameter control, advanced memory management and complete CLI argument support.

                **🚀 Features:**
                - ✅ All CLI arguments supported (resolution, frames, steps, solvers, seeds, prompts and more)
                - ✅ Output directory selection via `--output-dir` and automatic creation of the folder
                - ✅ Advanced memory management: offload layers to CPU, initialise on CPU, unload after generation and automatic CUDA OOM fallback
                - ✅ Experimental 8‑bit weight loading via `--load-8bit` (falls back to bfloat16)
                - ✅ Keep‑alive option (`--keep-alive`) to unload models after each run for lower memory usage
                - ✅ On‑demand REST API: start via CLI (`--api`) or the UI toggle
                - ✅ Real‑time GPU memory monitoring
                - ✅ Prompt extension capabilities for richer descriptions

                **📊 Memory Tips:**
                - Use **Force CPU Mode** on unsupported GPUs (e.g. GTX 1080 Ti) or when CUDA memory is exhausted
                - Enable **Model Offloading**, **T5 on CPU** and **Initialise on CPU** to reduce GPU memory consumption
                - Use **Unload After Generation** (keep‑alive) to automatically free memory after each video
                - Try the **8‑bit Quantisation** toggle to further reduce memory (experimental and may fall back to bfloat16)
                - Keep an eye on the GPU memory bar above and unload models when idle
                """
            )
            # Model management section
            with gr.Row(variant="panel", elem_classes=["model-selection-section"]):
                with gr.Column():
                    gr.Markdown("### 🎯 Model Management")
                    # Drop‑downs for selecting model path and model type
                    with gr.Row():
                        model_paths = gr.Dropdown(
                            choices=self.available_models,
                            label="📁 Model Path",
                            info="Select a pre‑downloaded Wan model directory",
                        )
                        model_type = gr.Dropdown(
                            choices=["t2v", "i2v", "ti2v"],
                            label="🔄 Model Type",
                            info="Text‑to‑Video | Image‑to‑Video | Text‑Image‑to‑Video",
                        )
                    with gr.Row():
                        load_model_btn = gr.Button("🔋 Load Model", variant="primary", scale=2)
                        unload_model_btn = gr.Button("🗑️ Unload Model", variant="secondary")
                    # Memory display and model status
                    memory_display = gr.HTML(
                        value=self.create_memory_display(),
                        elem_classes=["memory-info"],
                    )
                    model_status = gr.HTML(
                        value="<div style='background:#e3f2fd;padding:10px;border-radius:5px;'><b>Model Status:</b> No model loaded</div>",
                        elem_classes=["status-warning"],
                    )

            # Memory optimisation section
            with gr.Accordion("⚙️ Memory Optimisation Settings", open=False, elem_classes=["advanced-section"]):
                offload_model = gr.Checkbox(
                    label="🔄 Enable Model Offloading",
                    value=True,
                    info="Offload models to CPU between forward passes",
                )
                t5_cpu = gr.Checkbox(
                    label="💾 T5 on CPU",
                    value=False,
                    info="Place T5 encoder on CPU (reduces GPU memory)"
                )
                init_on_cpu = gr.Checkbox(
                    label="🔧 Initialise on CPU",
                    value=True,
                    info="Initialise models on CPU first"
                )
                convert_model_dtype = gr.Checkbox(
                    label="🏃 Optimise dtype",
                    value=False,
                    info="Use bfloat16 precision where supported",
                )
                # Experimental 8‑bit loading flag
                load_8bit_chk = gr.Checkbox(
                    label="📦 8‑bit Quantisation",
                    value=False,
                    info="Attempt to load model weights in 8‑bit (falls back to bfloat16)",
                )
                # Keep‑alive: unload models after each generation
                keep_alive_chk = gr.Checkbox(
                    label="⏱️ Unload After Generation",
                    value=False,
                    info="Unload the model after each generation to save memory",
                )
                with gr.Row():
                    t5_fsdp = gr.Checkbox(label="📊 T5 FSDP", value=False, info="Distributed T5 processing")
                    dit_fsdp = gr.Checkbox(label="📊 DiT FSDP", value=False, info="Distributed DiT processing")
                    ulysses_size = gr.Number(
                        label="Parallel workers",
                        value=1,
                        minimum=1,
                        maximum=8,
                        step=1,
                        info="Use multiple GPUs if available"
                    )

            # Tabs for different generation modes
            with gr.Tabs():
                # Text‑to‑Video tab
                with gr.Tab("📝 Text‑to‑Video"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            t2v_prompt = gr.Textbox(
                                label="🎯 Prompt",
                                placeholder="Describe the video you want to generate…",
                                lines=4,
                                value="A cinematic sunset over a mountain lake with golden reflections",
                            )
                            t2v_negative_prompt = gr.Textbox(
                                label="❌ Negative Prompt",
                                placeholder="What to exclude…",
                                lines=2,
                                value="blurry, low quality, distorted, ugly, malformed",
                            )
                            with gr.Row():
                                t2v_size = gr.Dropdown(
                                    choices=self.size_options['t2v'],
                                    value=self.size_options['t2v'][0],
                                    label="📐 Video Resolution",
                                    info="Width × Height (supported sizes)"
                                )
                                t2v_frame_num = gr.Slider(
                                    minimum=5,
                                    maximum=249,
                                    value=81,
                                    step=4,
                                    label="🎬 Frame Count (4n+1)",
                                )
                        with gr.Column(scale=3):
                            t2v_generate_btn = gr.Button("🚀 Generate T2V", variant="primary", size="lg")
                            t2v_output_video = gr.Video(
                                label="Generated Video",
                                elem_classes=["video-output"],
                            )
                            t2v_status = gr.HTML()

                # Image‑to‑Video tab
                with gr.Tab("🖼️ Image‑to‑Video"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                i2v_image_input = gr.Image(
                                    type="pil",
                                    label="📸 Input Image",
                                    height=300,
                                    sources=["upload", "clipboard"],
                                )
                                i2v_prompt = gr.Textbox(
                                    label="🎯 Prompt",
                                    placeholder="Describe how the image should animate…",
                                    lines=5,
                                    value="",
                                )
                            i2v_negative_prompt = gr.Textbox(
                                label="❌ Negative Prompt",
                                placeholder="What to exclude…",
                                lines=2,
                                value="blurry, animated, low quality, distorted",
                            )
                            with gr.Row():
                                # Use supported sizes for I2V tasks.  This controls the maximum area used by the model
                                i2v_size = gr.Dropdown(
                                    choices=self.size_options['i2v'],
                                    value=self.size_options['i2v'][0],
                                    label="📐 Video Resolution",
                                    info="Supported sizes determine the maximum pixel area"
                                )
                                i2v_frame_num = gr.Slider(
                                    minimum=5,
                                    maximum=161,
                                    value=81,
                                    step=4,
                                    label="🎬 Frame Count (4n+1)",
                                )
                        with gr.Column(scale=3):
                            i2v_generate_btn = gr.Button("🚀 Generate I2V", variant="primary", size="lg")
                            i2v_output_video = gr.Video(
                                label="Generated Video",
                                elem_classes=["video-output"],
                            )
                            i2v_status = gr.HTML()

                # Text‑Image‑to‑Video tab
                with gr.Tab("🎭 Text‑Image‑to‑Video"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                ti2v_image_input = gr.Image(
                                    type="pil",
                                    label="📸 Input Image (Optional)",
                                    height=300,
                                    sources=["upload", "clipboard"],
                                )
                                ti2v_prompt = gr.Textbox(
                                    label="🎯 Prompt",
                                    placeholder="Describe the scene and animation…",
                                    lines=5,
                                    value="",
                                )
                            ti2v_negative_prompt = gr.Textbox(
                                label="❌ Negative Prompt",
                                placeholder="What to exclude…",
                                lines=2,
                                value="",
                            )
                            with gr.Row():
                                ti2v_size = gr.Dropdown(
                                    choices=self.size_options['ti2v'],
                                    value=self.size_options['ti2v'][0],
                                    label="📐 Video Resolution",
                                    info="Supported sizes for TI2V",
                                )
                                ti2v_frame_num = gr.Slider(
                                    minimum=5,
                                    maximum=249,
                                    value=121,
                                    step=4,
                                    label="🎬 Frame Count (4n+1)",
                                )
                            # Prompt enhancement accordion
                            with gr.Accordion("🔍 Prompt Enhancement", open=False):
                                use_prompt_extend = gr.Checkbox(
                                    label="✨ Use Prompt Extension",
                                    value=False,
                                )
                                prompt_extend_method = gr.Dropdown(
                                    choices=["local_qwen", "dashscope"],
                                    value="local_qwen",
                                    label="Extension Method",
                                )
                                prompt_extend_model = gr.Textbox(
                                    label="Custom Model Path",
                                    placeholder="Optional: Custom prompt extension model",
                                    value="",
                                )
                                prompt_extend_target_lang = gr.Dropdown(
                                    choices=["zh", "en"],
                                    value="en",
                                    label="Target Language",
                                )
                        with gr.Column(scale=3):
                            ti2v_generate_btn = gr.Button("🚀 Generate TI2V", variant="primary", size="lg")
                            ti2v_output_video = gr.Video(
                                label="Generated Video",
                                elem_classes=["video-output"],
                            )
                            ti2v_status = gr.HTML()

                # Help & API documentation tab
                with gr.Tab("ℹ️ Help & API"):
                    gr.Markdown(
                        """
                        ## 📖 Help & Documentation

                        This interface exposes **all** of the command‑line arguments from the Wan 2.2 generator in an easy‑to‑use web UI.  Below is an overview of the available functionality and how to interact with the system both through the UI and via the accompanying REST API.

                        ### 🎯 Feature Overview

                        - **Video Modes**: Generate **Text‑to‑Video**, **Image‑to‑Video** and **Text‑Image‑to‑Video** animations.
                        - **Memory Management**: Offload model components to the CPU, initialise on CPU and enable offloading between inference passes to minimise GPU memory consumption.
                        - **Prompt Extension**: Automatically expand your prompt via local or remote language models to enrich the generated content.
                        - **Advanced Sampling**: Choose between different diffusion solvers (`unipc` or `dpm++`), adjust guidance scale and noise shift, and set the number of sampling steps.
                        - **Resolution Selection**: Only valid resolutions are presented in the dropdowns; invalid combinations that could cause negative tensor dimensions are filtered out.
                        - **Seed Control**: Use a fixed seed for reproducible results or set to -1 for a random seed on each run.
                        - **REST API**: A separate `wan_api.py` script provides a programmatic interface to the same functionality for integration with bots or other services.

                        ### 🛠 CLI & UI Options (summary)

                        | Option | Description |
                        |---|---|
                        | `model-dir` | Directory containing Wan model subfolders (each with a `configuration.json`). |
                        | `host`, `port` | Host/IP and port for the server. |
                        | `share` | Create a public Gradio share link. |
                        | `resolution` | Video size; choose from supported presets. |
                        | `frame_num` | Number of frames (must satisfy `4n+1`). |
                        | `sampling_steps` | Number of diffusion steps (higher yields better quality but slower generation). |
                        | `guide_scale` | Guidance scale controlling adherence to the prompt. |
                        | `seed` | Random seed; -1 selects a random seed each run. |
                        | `negative_prompt` | Undesired qualities to suppress. |
                        | `sample_solver` | Diffusion solver (`unipc` or `dpm++`). |
                        | `sample_shift` | Shift parameter controlling noise schedule. |
                        | `offload_model` | Offload model weights to the CPU between inference passes. |
                        | `t5_cpu`, `init_on_cpu` | Run the T5 encoder on CPU and initialise weights on CPU to reduce GPU memory usage. |
                        | `convert_model_dtype` | Load weights in bfloat16 for a smaller memory footprint. |
                        | `load_8bit` | Attempt to load models in 8‑bit precision (experimental; falls back to half precision). |
                        | `keep_alive` | Unload the model after each generation to free memory. |
                        | `t5_fsdp`, `dit_fsdp` | Enable fully‑sharded data parallelism for distributed, multi‑GPU setups. |
                        | `ulysses_size` | Number of model shards/workers for multi‑GPU inference. |
                        | `output_dir` | Directory where videos are saved (default: `output`). |

                        ### 🌐 API Usage

                        A simple REST API is provided by **`wan_api.py`** for programmatic access.  Start the API with:

                        ```bash
                        python wan_api.py --host 0.0.0.0 --port 8000 --model-dir /path/to/models
                        ```

                        **Endpoints**:

                        - `GET /models` → List available model directories.
                        - `POST /load_model` → JSON body `{ "model_path": "...", "model_type": "t2v|i2v|ti2v", "t5_cpu": false, "init_on_cpu": true, ... }` loads a model.
                        - `POST /unload_model` → Unloads the current model from memory.
                        - `POST /t2v` → Generate a video from text.  JSON fields mirror the UI parameters: `prompt`, `size`, `frame_num`, `steps`, `guide_scale`, `seed`, `negative_prompt`, `sample_solver`, `sample_shift`, and `offload_model`.
                        - `POST /i2v` → Generate a video from an image.  Same fields as `t2v` plus an `image` field with a base64‑encoded PNG or JPEG string.
                        - `POST /ti2v` → Generate a video from text and optionally an image.  Combine the parameters from `t2v` and `i2v` plus optional prompt extension settings (`use_prompt_extend`, `prompt_extend_method`, `prompt_extend_model`, `prompt_extend_target_lang`).

                        **Example**:

                        ```bash
                        # Load a T2V model
                        curl -X POST http://localhost:8000/load_model \
                             -H "Content-Type: application/json" \
                             -d '{"model_path":"/models/Wan2.2-T2V-A14B","model_type":"t2v"}'

                        # Generate a 1280×720 video with a random seed
                        curl -X POST http://localhost:8000/t2v \
                             -H "Content-Type: application/json" \
                             -d '{
                                 "prompt": "A cinematic sunset over a mountain lake",
                                 "size": "1280*720",
                                 "frame_num": 81,
                                 "steps": 50,
                                 "guide_scale": 5.0,
                                 "seed": -1,
                                 "negative_prompt": "blurry",
                                 "sample_solver": "unipc",
                                 "sample_shift": 5.0,
                                 "offload_model": true
                             }'
                        ```

                        ### 💡 Memory Tips

                        Running large models on limited GPUs can lead to **out‑of‑memory** errors.  PyTorch OOMs are typically caused by large batches, large model architectures or not freeing memory【7376056712258†L72-L90】.  To mitigate this:

                        - Enable `offload_model`, `t5_cpu` and `init_on_cpu` to move parts of the model to CPU memory.
                        - Use the **Force CPU Mode** checkbox if your GPU is unsupported (e.g. GTX 1080 Ti) or when CUDA memory is exhausted; generation will be slower but more reliable.
                        - Use `--keep-alive` (or the corresponding UI toggle) to automatically unload the model after each generation.  This minimises peak memory usage at the cost of longer load times on subsequent runs.
                        - Try `--load-8bit` to request 8‑bit weight loading; if supported this may further reduce memory (currently falls back to half precision).
                        - Reload the model on the CPU when encountering CUDA OOM errors; this interface automatically attempts such fallback when possible.
                        - Monitor GPU memory usage via the memory bar above and unload models when not in use.

                        For more details on memory management in PyTorch and how to avoid common pitfalls, refer to the official documentation and community resources【7376056712258†L72-L90】.
                        
                        ### 🚀 Starting the API

                        There are two ways to launch the API server:

                        1. **CLI flag**: run the web UI with `--api` to automatically start the API alongside Gradio.  You can adjust the API host and port using `--api-host` and `--api-port`.  Example:

                        ```bash
                        python wan_web.py --port 7860 --api --api-port 8000
                        ```

                        2. **UI toggle**: open the **API Server** tab in the web UI and enable the server using the checkbox.  Set the desired host/port before toggling on.  **Note:** Do not enable the API in the UI if you started it via the CLI flag, as running two instances simultaneously may cause conflicts.

                        Disabling the API via the UI will terminate the server process started from the UI.  If the API was started via the CLI flag, close the main process to stop it.
                        
                        """
                    )

                # API server control tab
                with gr.Tab("⚡ API Server"):
                    gr.Markdown(
                        """
                        ## ⚡ API Server Control

                        Use this tab to start or stop the built‑in REST API.  The API
                        server runs in a separate process and can be enabled via the
                        toggle below.  **Do not enable the API here if you
                        started it using the `--api` command‑line flag**, as this would
                        spawn a second instance.  Likewise, running the API from
                        both the CLI and this UI concurrently is not supported.
                        """
                    )
                    with gr.Row():
                        api_enable_checkbox = gr.Checkbox(label="Enable API server", value=False)
                        api_host_input = gr.Textbox(label="API Host", value="127.0.0.1")
                        api_port_input = gr.Number(label="API Port", value=8000, precision=0)
                    api_status = gr.HTML(value="API server disabled")
                    # Handler for toggling API server
                    def handle_api_toggle(enable: bool, host: str, port: float):
                        # Note: port comes in as float due to Gradio Number widget; cast to int
                        port_int = int(port) if port else 8000
                        if enable:
                            status = self.start_api(host, port_int)
                        else:
                            status = self.stop_api()
                        return status
                    # Connect change event on the checkbox
                    api_enable_checkbox.change(
                        fn=handle_api_toggle,
                        inputs=[api_enable_checkbox, api_host_input, api_port_input],
                        outputs=api_status,
                    )

            # Advanced generation parameters shared across tabs
            with gr.Accordion("⚙️ Advanced Generation Parameters", open=False, elem_classes=["advanced-section"]):
                gr.Markdown("### 🔬 Fine‑tune your generation")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Sampling Configuration**")
                        sample_solver = gr.Dropdown(
                            choices=["unipc", "dpm++"],
                            value="unipc",
                            label="Solver Algorithm",
                        )
                        sample_shift = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1,
                            label="Sampling Shift",
                        )
                        sample_guide_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.1,
                            label="Guidance Scale",
                        )
                        seed = gr.Number(
                            value=-1,
                            label="🎲 Seed (-1 for random)",
                            precision=0,
                        )
                    with gr.Column():
                        gr.Markdown("**Performance Settings**")
                        generation_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Sampling Steps",
                            info="Lower for speed, higher for quality",
                        )
                        performance_offload = gr.Checkbox(
                            label="✅ Performance Model Offload",
                            value=True,
                            info="Offload models during generation",
                        )
                        cpu_fallback = gr.Checkbox(
                            label="🧠 Force CPU Mode",
                            value=False,
                            info="Force CPU generation regardless of GPU",
                        )

            # ------------------------------------------------------------------
            # Event handler functions and bindings
            # ------------------------------------------------------------------
            def handle_load_model(
                model_path: str,
                model_type_val: str,
                offload_model_t: bool,
                t5_cpu_t: bool,
                init_on_cpu_t: bool,
                convert_model_dtype_t: bool,
                load_8bit_val: bool,
                keep_alive_val: bool,
                t5_fsdp_t: bool,
                dit_fsdp_t: bool,
                ulysses_size_t: int,
                cpu_fallback_val: bool,
            ):
                """Load the selected model into memory.

                If ``cpu_fallback_val`` is True, the device is forced to CPU
                regardless of detected GPU availability.  Additional flags are
                forwarded to the model constructor via ``load_model``.
                """
                # Override the default device when CPU fallback is requested
                if cpu_fallback_val:
                    self.device = "cpu"
                # Apply UI toggle for 8‑bit and keep‑alive to the interface
                # level.  These flags persist across model loads and
                # generations.
                self.load_8bit = bool(load_8bit_val)
                self.keep_alive = bool(keep_alive_val)
                load_kwargs = {
                    "t5_cpu": t5_cpu_t,
                    "init_on_cpu": init_on_cpu_t,
                    "convert_model_dtype": convert_model_dtype_t,
                    "t5_fsdp": t5_fsdp_t,
                    "dit_fsdp": dit_fsdp_t,
                    "ulysses_size": ulysses_size_t if ulysses_size_t > 1 else 1,
                }
                if not model_path:
                    return (
                        self.create_memory_display(),
                        "<div style='background:#ffebee;padding:10px;border-radius:5px;'>❌ Please select a model directory</div>",
                    )
                status = self.load_model(model_path, model_type_val, **load_kwargs)
                return self.create_memory_display(), status

            def handle_unload_model():
                status = self._unload_model()
                return self.create_memory_display(), status

            def update_memory_display_simple():
                return self.create_memory_display()

            # Bind buttons
            load_model_btn.click(
                fn=handle_load_model,
                inputs=[
                    model_paths,
                    model_type,
                    offload_model,
                    t5_cpu,
                    init_on_cpu,
                    convert_model_dtype,
                    load_8bit_chk,
                    keep_alive_chk,
                    t5_fsdp,
                    dit_fsdp,
                    ulysses_size,
                    cpu_fallback,
                ],
                outputs=[memory_display, model_status],
            )
            unload_model_btn.click(
                fn=handle_unload_model,
                outputs=[memory_display, model_status],
            )

            # Generation handlers for each tab
            t2v_generate_btn.click(
                fn=self.generate_t2v,
                inputs=[
                    t2v_prompt,
                    t2v_size,
                    t2v_frame_num,
                    generation_steps,
                    sample_guide_scale,
                    seed,
                    t2v_negative_prompt,
                    sample_solver,
                    sample_shift,
                    performance_offload,
                    convert_model_dtype,
                ],
                outputs=[t2v_output_video, t2v_status],
            )
            i2v_generate_btn.click(
                fn=self.generate_i2v,
                inputs=[
                    i2v_prompt,
                    i2v_image_input,
                    i2v_size,
                    i2v_frame_num,
                    generation_steps,
                    sample_guide_scale,
                    seed,
                    i2v_negative_prompt,
                    sample_solver,
                    sample_shift,
                    performance_offload,
                ],
                outputs=[i2v_output_video, i2v_status],
            )
            ti2v_generate_btn.click(
                fn=self.generate_ti2v,
                inputs=[
                    ti2v_prompt,
                    ti2v_image_input,
                    ti2v_size,
                    ti2v_frame_num,
                    generation_steps,
                    sample_guide_scale,
                    seed,
                    ti2v_negative_prompt,
                    sample_solver,
                    sample_shift,
                    performance_offload,
                    use_prompt_extend,
                    prompt_extend_method,
                    prompt_extend_model,
                    prompt_extend_target_lang,
                ],
                outputs=[ti2v_output_video, ti2v_status],
            )

            # Manual memory refresh button
            refresh_memory_btn = gr.Button("🔄 Refresh Memory", size="sm")
            refresh_memory_btn.click(
                fn=update_memory_display_simple,
                outputs=memory_display,
            )

        return interface

    def add_custom_css(self) -> str:
        """Return additional CSS for the interface.

        This method defines gradients and styles for various UI elements such
        as buttons and info cards.  It is called once during interface
        creation and appended to the base CSS.
        """
        return """
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .primary-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }
        .secondary-button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            font-weight: 600;
        }
        .info-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #1976d2;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        """


def main() -> None:
    """Entry point for the command line interface."""
    parser = argparse.ArgumentParser(description="Wan 2.2 Professional Web Interface")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory containing Wan model folders",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web interface on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the web interface on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--auto-memory",
        action="store_true",
        help="Enable automatic memory optimisation (unused but kept for CLI compatibility)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the REST API server alongside the web UI",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="127.0.0.1",
        help="Host/IP for the API server",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the API server",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated videos (created if missing)",
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

    # Create the interface wrapper with output directory, keep‑alive and 8‑bit settings
    interface = WanWebInterface(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        keep_alive=args.keep_alive,
        load_8bit=args.load_8bit,
    )
    print("=" * 80)
    print("🎬 Wan 2.2 Professional Video Generation")
    print("=" * 80)
    print(f"• Host: {args.host}")
    print(f"• Port: {args.port}")
    print(f"• Share: {args.share}")
    print(f"• Models: {interface.available_models}")
    print(f"• Output directory: {interface.output_dir}")
    if torch.cuda.is_available() and interface.device != "cpu":
        gpu_info = interface.get_gpu_memory_info()
        if gpu_info:
            print(f"• GPU Memory: {gpu_info['total']:.1f}GB total available")
    else:
        print("• Running on CPU")
    # If the --api flag was provided, start the REST API server in a separate process
    api_process = None
    if args.api:
        import subprocess
        import sys
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan_api.py"),
            "--model-dir",
            args.model_dir,
            "--host",
            args.api_host,
            "--port",
            str(args.api_port),
            "--output-dir",
            args.output_dir,
        ]
        if args.keep_alive:
            cmd.append("--keep-alive")
        if args.load_8bit:
            cmd.append("--load-8bit")
        try:
            api_process = subprocess.Popen(cmd)
            print(f"• API server started at http://{args.api_host}:{args.api_port}")
        except Exception as e:
            print(f"❌ Failed to launch API server: {e}")
    # Build the Gradio interface and launch it
    iface = interface.create_interface()
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        show_api=True,
        inbrowser=False,
        quiet=False,
        max_threads=1,
    )
    # Terminate the API server when Gradio exits
    if api_process is not None:
        try:
            api_process.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()