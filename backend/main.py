# backend/main.py

import os
import io
import zipfile
import tempfile
import shutil
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict
import json # Make sure json is imported

import requests # To forward requests to Colab/ngrok
# Corrected FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Response # ADDED Response HERE
from fastapi.middleware.cors import CORSMiddleware # To allow requests from frontend
from pydantic import BaseModel, Field # For request body validation

# Import your overlay generator class
try:
    from synthetic_generator import SyntheticImageGenerator
except ImportError:
    logging.error("CRITICAL: synthetic_generator.py not found or SyntheticImageGenerator class missing.")
    # Exit or raise a more specific error if critical
    raise SystemExit("synthetic_generator.py not found")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Unified AI Image Generation API",
    description="Provides endpoints for SDXL-LoRA generation (via Colab) and local Overlay generation.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allow requests from your frontend development server (and potentially production URL later)
origins = [
    "http://localhost:3001", # Default Next.js dev port
    "http://127.0.0.1:3001",
    # Add your deployed frontend URL here later if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Configuration (Get Colab URL from environment variable for flexibility) ---
# Read the environment variable WHEN THE SCRIPT LOADS
COLAB_BACKEND_URL = os.environ.get("COLAB_NGROK_URL") # Simpler get

if not COLAB_BACKEND_URL:
    logger.warning("⚠️ COLAB_NGROK_URL environment variable not set. SDXL generation endpoint will fail.")
else:
    logger.info(f"Colab Backend URL configured: {COLAB_BACKEND_URL}")

SDXL_ENDPOINT = "/generate-augmented-images/" # Endpoint on the Colab backend


# --- Pydantic Models for Request Bodies ---
class SDXLRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, unrealistic, drawing, illustration, text, words"
    num_base_images: int = Field(default=1, ge=1)
    augmentations_per_image: int = Field(default=0, ge=0)
    num_inference_steps: int = Field(default=30, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)

class OverlayGenerationConfig(BaseModel):
    image_number: int = Field(default=10, ge=1)
    max_objects_per_image: int = Field(default=3, ge=1)
    image_width: int = Field(default=640, ge=64)
    image_height: int = Field(default=480, ge=64)
    scaling_factors: List[float] = Field(default=[0.2, 0.5], min_length=2, max_length=2) # Use min/max_length
    avoid_collisions: bool = False
    parallelize: bool = True

# --- Helper Function ---
def create_zip_archive_from_dir(folder_path: Path) -> Optional[io.BytesIO]: # Added Optional return type
    zip_buffer = io.BytesIO()
    items_added = 0
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in folder_path.rglob('*'):
                if item.is_file():
                    arcname = item.relative_to(folder_path)
                    zipf.write(item, arcname=arcname)
                    items_added += 1
        if items_added == 0:
            logger.warning(f"No files found in {folder_path} to zip.")
            return None # Return None if zip would be empty
        zip_buffer.seek(0)
        logger.info(f"Created zip archive with {items_added} files from {folder_path}")
        return zip_buffer
    except Exception as e:
        logger.error(f"Error creating zip archive from {folder_path}: {e}", exc_info=True)
        return None # Return None on error

# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def read_root():
    """Check if the API is running."""
    return {"status": "Unified Generation API is running", "colab_backend_url_status": "Set" if COLAB_BACKEND_URL else "Not Set"}

# --- SDXL Generation Endpoint ---
@app.post("/generate-sdxl", tags=["SDXL Generation"], response_class=Response)
async def generate_sdxl_proxy(request: SDXLRequest):
    """Forwards generation request to the Colab/Ngrok backend."""
    if not COLAB_BACKEND_URL: # Check again in case env var wasn't set at startup
        logger.error("SDXL request failed: COLAB_NGROK_URL environment variable is not set.")
        raise HTTPException(status_code=503, detail="Colab backend URL is not configured on the server.")

    target_url = COLAB_BACKEND_URL.rstrip('/') + SDXL_ENDPOINT
    logger.info(f"Forwarding SDXL request to: {target_url}")
    request_data = request.model_dump() # Use model_dump() for Pydantic v2+

    try:
        response = requests.post(target_url, json=request_data, timeout=1800) # 30 min timeout
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "application/zip" in content_type:
            logger.info(f"Received zip response from Colab backend.")
            # Forward necessary headers
            headers = {
                "Content-Disposition": response.headers.get("Content-Disposition", "attachment;filename=sdxl_generated_data.zip"),
                "Content-Type": "application/zip", # Ensure correct content type is sent back
            }
            return Response(
                content=response.content,
                media_type="application/zip",
                headers=headers
            )
        else:
            logger.warning(f"Received non-zip response from Colab backend: {response.status_code}")
            detail_msg = f"Colab backend error ({response.status_code})"
            try:
                # Try to parse detail if backend sent JSON error
                detail = response.json().get("detail")
                if detail:
                    detail_msg = f"Colab backend error: {detail}"
            except Exception:
                 # Otherwise use raw text
                 detail_msg = f"Colab backend error: {response.text[:500]}"
            raise HTTPException(status_code=response.status_code, detail=detail_msg)

    except requests.exceptions.Timeout:
        logger.error(f"Request to Colab backend timed out: {target_url}")
        raise HTTPException(status_code=504, detail="Request to Colab backend timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Colab backend {target_url}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Could not connect to Colab backend: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during SDXL proxy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error proxying SDXL request: {e}")

# --- Overlay Generation Endpoint ---

# This synchronous version is simpler for local testing but will block the server
# Consider BackgroundTasks for production
def run_overlay_generation_task_sync(input_dir: Path, output_dir: Path, config: OverlayGenerationConfig, aug_file_path_str: Optional[str]):
    """Synchronous version for direct execution."""
    logger.info(f"Synchronous overlay generation started in {output_dir}")
    try:
        generator = SyntheticImageGenerator(
            input_dir=str(input_dir), output_dir=str(output_dir), image_number=config.image_number,
            max_objects_per_image=config.max_objects_per_image, image_width=config.image_width, image_height=config.image_height,
            augmentation_path=aug_file_path_str, scale_foreground_by_background_size=False,
            scaling_factors=config.scaling_factors, avoid_collisions=config.avoid_collisions, parallelize=config.parallelize
        )
        generator.generate_images(progress_callback=None)
        logger.info(f"Synchronous overlay generation finished for {output_dir}")
        return True # Indicate success
    except Exception as e:
        logger.error(f"Error in synchronous overlay generation task: {e}", exc_info=True)
        # Optionally write an error file or log more details
        # Ensure output_dir exists even if generation fails partially before zipping
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "_GENERATION_ERROR.txt", "w") as f:
             f.write(f"Generation failed:\n{e}\nCheck backend logs for details.")
        return False # Indicate failure


@app.post("/generate-overlay", tags=["Overlay Generation"], response_class=Response)
async def generate_overlay_data(
    # Use BackgroundTasks if making the generation async later
    background_tasks: BackgroundTasks,
    # Use Form(...) to receive parameters alongside files
    config_json: str = Form(...),
    background_files: List[UploadFile] = File(...),
    foreground_files: List[UploadFile] = File(...),
    augmentation_file: Optional[UploadFile] = File(None)
):
    """
    Receives uploaded files and config, runs overlay generation, returns zip.
    Foreground files should have filenames indicating category, e.g., 'category1__object1.png'.
    """
    logger.info(f"Received overlay generation request. Config: {config_json[:100]}...") # Log start
    # ... (logging file counts) ...

    # --- Parse Config ---
    try:
        config_dict = json.loads(config_json)
        config = OverlayGenerationConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error parsing config_json: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration JSON: {e}")

    # --- Create Temporary Directory ---
    # Using a context manager ensures cleanup even if errors occur later
    with tempfile.TemporaryDirectory(prefix="overlay_gen_") as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        foregrounds_base = input_dir / "foregrounds"
        backgrounds_base = input_dir / "backgrounds"
        logger.info(f"Using temporary directory: {temp_dir}")

        try:
            # Create structure
            foregrounds_base.mkdir(parents=True, exist_ok=True)
            backgrounds_base.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(exist_ok=True)

            # --- Save Uploaded Files ---
            logger.info("Saving uploaded files...")
            # ... (Same file saving logic as before using shutil.copyfileobj and closing files) ...
            # Save backgrounds
            for file in background_files:
                # Sanitize filename (optional but recommended)
                safe_filename = Path(file.filename).name
                dest = backgrounds_base / safe_filename
                try:
                    with open(dest, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
                finally: file.file.close()
            # Save foregrounds
            for file in foreground_files:
                safe_filename = Path(file.filename).name # Basic sanitization
                parts = safe_filename.split("__", 1)
                if len(parts) == 2:
                    category, filename_part = parts
                    category_dir = foregrounds_base / category.replace("..","").replace("/","").replace("\\","") # Sanitize category
                    filename_part = Path(filename_part).name # Sanitize filename part
                    category_dir.mkdir(parents=True, exist_ok=True)
                    dest = category_dir / filename_part
                else:
                    logger.warning(f"Foreground filename '{safe_filename}' doesn't match 'category__name.png' format. Placing in 'unknown'.")
                    category_dir = foregrounds_base / "unknown"
                    category_dir.mkdir(parents=True, exist_ok=True)
                    dest = category_dir / safe_filename
                try:
                     with open(dest, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
                finally: file.file.close()
            # Save augmentation file
            aug_file_path_str = None
            if augmentation_file:
                safe_filename = Path(augmentation_file.filename).name
                dest = temp_path / safe_filename # Save in root of temp for simplicity
                try:
                     with open(dest, "wb") as buffer: shutil.copyfileobj(augmentation_file.file, buffer)
                     aug_file_path_str = str(dest)
                finally: augmentation_file.file.close()


            # --- Run Generation (Synchronous for now) ---
            logger.info(f"Starting overlay generation task...")
            generation_success = run_overlay_generation_task_sync(input_dir, output_dir, config, aug_file_path_str)

            # --- Zip Results ---
            if not generation_success:
                 logger.error("Overlay generation task failed. Zipping any partial results or error file.")
                 # Zip whatever is in output_dir, might include the error file
                 # Fall through to zipping

            logger.info("Zipping results...")
            zip_buffer = create_zip_archive_from_dir(output_dir)

            if not zip_buffer:
                 # Error should have been logged by helper function
                 raise HTTPException(status_code=500, detail="Failed to create zip archive of results.")

            logger.info("Generation and zipping complete.")

            # --- Return Zip File ---
            # Note: Temp dir is automatically cleaned up after this 'with' block exits
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment;filename=overlay_dataset.zip"}
            )

        except HTTPException:
             # Re-raise HTTP exceptions directly
             raise
        except Exception as e:
            logger.error(f"Error during file saving or processing in overlay request: {e}", exc_info=True)
            # Temp dir will be cleaned up by 'with' statement
            raise HTTPException(status_code=500, detail=f"Internal server error during file handling or setup: {e}")


# --- Uvicorn Runner (for local development) ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server for local development...")
    # Check NGROK URL again, as env vars might not persist across reloads easily
    if not COLAB_BACKEND_URL:
        print("⚠️ WARNING: COLAB_NGROK_URL environment variable not detected at server start.")
    print(f"FastAPI server starting. Ensure Colab backend is running if using SDXL.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)