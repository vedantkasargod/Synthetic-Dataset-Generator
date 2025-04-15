import torch
from diffusers import AutoPipelineForText2Image
import os
import gc
import io
import zipfile
from PIL import Image, ImageEnhance, ImageOps, ImageFilter # Import PIL modules
import random # For random choices
import numpy as np # For noise, cutout
import time # For timing

# Import torchvision transforms (ensure it's installed: pip install torchvision)
try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    TORCHVISION_AVAILABLE = True
    print("Torchvision found, augmentations enabled.")
except ImportError:
    print("Torchvision not found. Some augmentations might be limited (e.g., ColorJitter).")
    TORCHVISION_AVAILABLE = False

import asyncio
import nest_asyncio
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok, conf
import nest_asyncio

# Apply nest_asyncio to allow running uvicorn in Colab's event loop
nest_asyncio.apply()

# --- Global Variables ---
pipeline = None # Initialize to None
pipeline_loaded = False
LORA_WEIGHTS_PATH = "/content/pytorch_lora_weights.safetensors" # Make sure LoRA is uploaded/available here
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# --- Model Loading Function ---
def load_model_pipeline():
    global pipeline, pipeline_loaded # Declare use of global variables
    if pipeline_loaded:
        print("Pipeline already loaded.")
        return True

    print("Attempting to load pipeline (this may take time)...")
    start_time = time.time()
    try:
        # Try loading with fp16 first (common for Colab GPUs)
        current_pipeline = AutoPipelineForText2Image.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            # Add resume_download=True maybe? Or cache_dir?
        )
        # Move to GPU *before* loading LoRA
        current_pipeline.to("cuda")
        print(f"Base pipeline loaded to CUDA in {time.time() - start_time:.2f} seconds.")

        # Load LoRA weights if file exists
        if os.path.exists(LORA_WEIGHTS_PATH):
            print(f"Loading LoRA weights from {LORA_WEIGHTS_PATH}...")
            lora_start_time = time.time()
            current_pipeline.load_lora_weights(LORA_WEIGHTS_PATH, weight_name="pytorch_lora_weights.safetensors") # Specify weight_name if needed
            print(f"LoRA weights loaded in {time.time() - lora_start_time:.2f} seconds.")
            # Optional: Fuse LoRA for potential speedup, but increases memory
            # current_pipeline.fuse_lora()
            # print("LoRA weights fused.")
        else:
             print(f"Warning: LoRA file not found at {LORA_WEIGHTS_PATH}. Using base model only.")

        # Enable memory optimizations if available
        try:
            current_pipeline.enable_xformers_memory_efficient_attention()
            print("xFormers memory efficient attention enabled.")
        except ImportError:
            print("xFormers not installed or compatible. Running without it (might use more VRAM).")
        except Exception as e:
            print(f"Could not enable xFormers: {e}")

        # Final check if the loaded object is callable
        if current_pipeline is None or not callable(current_pipeline):
             print("ERROR: Pipeline object is not valid after loading attempt!")
             pipeline = None
             pipeline_loaded = False
             return False
        else:
             pipeline = current_pipeline # Assign to global variable
             pipeline_loaded = True
             print(f"Pipeline ready. Total loading time: {time.time() - start_time:.2f} seconds.")
             return True

    except Exception as e:
        print(f"CRITICAL ERROR during pipeline loading: {e}")
        pipeline = None # Ensure it's None on error
        pipeline_loaded = False
        return False

# --- Augmentation Function ---
def apply_random_augmentations(img: Image.Image) -> Image.Image:
    """Applies a random selection of augmentations to a PIL Image."""
    augmented_img = img.copy()
    applied_augs = [] # Keep track of applied augmentations

    try:
        # --- Geometric ---
        if random.random() < 0.5:
            augmented_img = ImageOps.mirror(augmented_img)
            applied_augs.append("Flip")

        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            augmented_img = augmented_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0)) # Fill bg black
            applied_augs.append(f"Rotate({angle:.1f})")

        if TORCHVISION_AVAILABLE and random.random() < 0.4:
            original_size = augmented_img.size
            scale_factor = random.uniform(0.8, 1.0)
            transform = T.RandomResizedCrop(size=original_size, scale=(scale_factor**2, scale_factor**2), ratio=(0.9, 1.1), antialias=True)
            augmented_img_tensor = TF.to_tensor(augmented_img)
            augmented_img_tensor = transform(augmented_img_tensor)
            augmented_img = TF.to_pil_image(augmented_img_tensor)
            applied_augs.append("ResizeCrop")

        if random.random() < 0.2:
            img_np = np.array(augmented_img)
            h, w = img_np.shape[:2]
            scale = random.uniform(0.05, 0.15)
            cut_h, cut_w = int(h * scale), int(w * scale)
            if h > cut_h and w > cut_w: # Ensure cutout size is valid
                y1 = np.random.randint(0, h - cut_h)
                x1 = np.random.randint(0, w - cut_w)
                img_np[y1 : y1 + cut_h, x1 : x1 + cut_w, :] = 0
                augmented_img = Image.fromarray(img_np)
                applied_augs.append("CutOut")

        # --- Color / Appearance ---
        if TORCHVISION_AVAILABLE and random.random() < 0.7:
            jitter_transform = T.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=0.0)
            augmented_img_tensor = TF.to_tensor(augmented_img)
            augmented_img_tensor = jitter_transform(augmented_img_tensor)
            augmented_img = TF.to_pil_image(augmented_img_tensor)
            applied_augs.append("ColorJitter")
        elif not TORCHVISION_AVAILABLE: # Fallback
            if random.random() < 0.5:
                 factor = random.uniform(0.7, 1.3)
                 enhancer = ImageEnhance.Brightness(augmented_img); augmented_img = enhancer.enhance(factor)
                 applied_augs.append("Brightness")
            if random.random() < 0.5:
                 factor = random.uniform(0.7, 1.3)
                 enhancer = ImageEnhance.Contrast(augmented_img); augmented_img = enhancer.enhance(factor)
                 applied_augs.append("Contrast")
            if random.random() < 0.5:
                 factor = random.uniform(0.7, 1.3)
                 enhancer = ImageEnhance.Color(augmented_img); augmented_img = enhancer.enhance(factor) # Saturation
                 applied_augs.append("Saturation")

        if random.random() < 0.3:
            img_np = np.array(augmented_img).astype(np.float32)
            sigma = random.uniform(5, 15)
            noise = np.random.normal(0, sigma, img_np.shape)
            noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            augmented_img = Image.fromarray(noisy_img_np)
            applied_augs.append("Noise")

        if not applied_augs: # If somehow no augmentation was applied, maybe apply one forcefully?
             enhancer = ImageEnhance.Brightness(augmented_img); augmented_img = enhancer.enhance(random.uniform(0.9, 1.1))
             applied_augs.append("ForcedBrightness")

        print(f"    Applied: {', '.join(applied_augs)}")

    except Exception as e:
        print(f"    Error during augmentation: {e}. Returning original image.")
        return img # Return original on error

    return augmented_img

# --- FastAPI App Definition ---
app = FastAPI(title="SDXL Image Generation API", description="Generates images using SDXL with optional LoRA and augmentations.")

# Define request body model using Pydantic
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, unrealistic, drawing, illustration, text, words"
    num_base_images: int = 1
    augmentations_per_image: int = 0 # Allow 0 augmentations
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

# Root endpoint for health check
@app.get("/")
async def read_root():
    return {"status": "API is running", "model_loaded": pipeline_loaded}

# Endpoint for generation
@app.post("/generate-augmented-images/", response_class=Response)
async def generate_augmented_images_api(request: GenerationRequest):
    global pipeline, pipeline_loaded # Access global pipeline

    if not pipeline_loaded: # Check if model is loaded
        print("Pipeline not loaded. Attempting to load now...")
        if not load_model_pipeline():
            raise HTTPException(status_code=503, detail="Model pipeline could not be loaded.")

    if not pipeline_loaded or pipeline is None or not callable(pipeline): # Double check
         raise HTTPException(status_code=503, detail="Model pipeline is not available or not callable.")

    final_images_list = [] # List to store ALL final images (original + augmented)
    print(f"Received request: {request.num_base_images} base images, {request.augmentations_per_image} augmentations each.")
    request_start_time = time.time()

    # --- Generation Loop ---
    for i in range(request.num_base_images):
        base_image_start_time = time.time()
        print(f"--- Generating Base Image {i+1}/{request.num_base_images} ---")
        base_image = None
        try:
            if not callable(pipeline): # Check right before call
                raise ValueError(f"Pipeline object became invalid before generating image {i+1}")

            # Generate the base image
            output = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                num_images_per_prompt=1
            )
            base_image = output.images[0]
            print(f"   Base image {i+1} generated in {time.time() - base_image_start_time:.2f} seconds.")

            # ***** ADD THE ORIGINAL BASE IMAGE TO THE LIST *****
            if base_image:
                final_images_list.append({"image": base_image, "base_index": i, "aug_index": -1}) # Store with indices, -1 for original

            # --- Augmentation Loop (Only if requested) ---
            if base_image and request.augmentations_per_image > 0:
                print(f"--- Augmenting Base Image {i+1} ({request.augmentations_per_image} versions) ---")
                for j in range(request.augmentations_per_image):
                    aug_start_time = time.time()
                    try:
                        # Apply potentially different random augmentations each time
                        augmented_image = apply_random_augmentations(base_image)
                        # Store augmented image with indices
                        final_images_list.append({"image": augmented_image, "base_index": i, "aug_index": j})
                        # print(f"   Augmented version {j+1} created in {time.time() - aug_start_time:.2f} seconds.")
                    except Exception as e_aug:
                        print(f"Error during augmentation {j+1} for base image {i+1}: {e_aug}")
                        # Decide: skip augmentation?

            # Clean up VRAM after processing one base image and its augmentations
            del base_image
            del output
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   Memory cleaned after base image {i+1}.")

        except Exception as e_gen:
            print(f"Error during generation or processing for base image {i+1}: {e_gen}")
            # Continue to next base image if one fails?

    # --- Zipping Results ---
    if not final_images_list:
        raise HTTPException(status_code=500, detail="No images were successfully generated or augmented.")

    print(f"--- Generation & Augmentation complete. Creating zip file with {len(final_images_list)} images... ---")
    zip_creation_start_time = time.time()
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for item in final_images_list: # Iterate through list of dictionaries
                img = item["image"]
                base_idx = item["base_index"]
                aug_idx = item["aug_index"]

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # Generate filename
                if aug_idx == -1: # Original base image
                    filename = f"base_{base_idx+1}_original.png"
                else: # Augmented image
                    filename = f"base_{base_idx+1}_aug_{aug_idx+1}.png"

                zip_file.writestr(filename, img_byte_arr.getvalue())
        zip_buffer.seek(0)
        print(f"Zip file created in {time.time() - zip_creation_start_time:.2f} seconds.")
    except Exception as e_zip:
        print(f"Error creating zip file: {e_zip}")
        raise HTTPException(status_code=500, detail="Failed to create zip file.")

    total_request_time = time.time() - request_start_time
    print(f"Total request processed in {total_request_time:.2f} seconds.")

    # --- Return Response ---
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment;filename=generated_augmented_images.zip"}
    )


# --- Setup ngrok and Run Server ---
NGROK_AUTHTOKEN = "2vTgX4uDqHnL37mxFnKimer9MMC_7RfBky7GM4UewdTTBPPyJ" # Replace with your actual token

if NGROK_AUTHTOKEN == "YOUR_NGROK_AUTHTOKEN" or not NGROK_AUTHTOKEN:
  print("ERROR: Please replace 'YOUR_NGROK_AUTHTOKEN' with your actual ngrok token from https://dashboard.ngrok.com/get-started/your-authtoken")
else:
  # Load the model pipeline ONCE before starting the server
  load_model_pipeline()

  # Setup ngrok tunnel
  conf.get_default().auth_token = NGROK_AUTHTOKEN
  try:
      # Kill any existing tunnels if script is rerun
      ngrok.kill()
  except:
      pass # Ignore if no tunnels exist
  public_url = ngrok.connect(8000)
  print(f"✅ FastAPI backend active.")
  print(f"🔗 Public Ngrok URL: {public_url}")
  print("🚀 Keep this Colab notebook running!")
  print("📋 Copy the Ngrok URL above and paste it into your Streamlit script's COLAB_API_URL variable.")

  # Start the FastAPI server
  try:
      # Use reload=False for production/stability in Colab
      uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
  except KeyboardInterrupt:
      print("Shutting down server...")
      ngrok.disconnect(public_url)
  except Exception as e_server:
      print(f"Server crashed: {e_server}")
      ngrok.disconnect(public_url)