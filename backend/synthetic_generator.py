# synthetic_generator.py

import json
import logging
import warnings
from pathlib import Path
import random
# import base64 # Not used in the provided class logic
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import numpy as np
# from skimage import measure # Not used in the provided class logic
# from shapely.geometry import Polygon # Not used in the provided class logic
import albumentations as A
from joblib import Parallel, delayed
from typing import List, Dict, Optional, Tuple

# Set up logging (can be configured further in app.py if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paste your ENTIRE SyntheticImageGenerator class definition here ---
class SyntheticImageGenerator:
    def __init__(self, input_dir: str, output_dir: str, image_number: int, max_objects_per_image: int,
                 image_width: int, image_height: int, augmentation_path: Optional[str], # Made optional
                 scale_foreground_by_background_size: bool, scaling_factors: List[float],
                 avoid_collisions: bool, parallelize: bool):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_number = image_number
        self.max_objects_per_image = max_objects_per_image
        self.image_width = image_width
        self.image_height = image_height
        self.zero_padding = 8 # Padding for output filenames
        self.augmentation_path = Path(augmentation_path) if augmentation_path else None # Handle None
        self.scale_foreground_by_background_size = scale_foreground_by_background_size # Currently unused in logic
        self.scaling_factors = scaling_factors
        self.avoid_collisions = avoid_collisions # Currently unused in logic
        self.parallelize = parallelize

        self._validate_input_directory() # Checks subdirs exist and finds images
        self._validate_output_directory() # Creates output dir
        self._load_augmentation_pipeline() # Loads augmentations if path valid

    def _validate_input_directory(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f'Input directory does not exist: {self.input_dir}')

        self.foregrounds_dir = self.input_dir / 'foregrounds'
        self.backgrounds_dir = self.input_dir / 'backgrounds'

        if not self.foregrounds_dir.is_dir():
            # Allow it not to exist initially if we create it from uploads
            logging.warning(f"'foregrounds' sub-directory not found in {self.input_dir}. Assuming it will be populated.")
            self.foregrounds_dict = {}
        else:
             self._process_foregrounds() # Only process if dir exists

        if not self.backgrounds_dir.is_dir():
             # Allow it not to exist initially
             logging.warning(f"'backgrounds' sub-directory not found in {self.input_dir}. Assuming it will be populated.")
             self.background_images = []
        else:
             self._process_backgrounds() # Only process if dir exists

    def _process_foregrounds(self) -> None:
        self.foregrounds_dict: Dict[str, List[Path]] = {}
        category_count = 0
        image_count = 0
        for category in self.foregrounds_dir.iterdir():
            if category.is_dir():
                # Find PNG images, case-insensitive check just in case
                images = [p for p in category.glob('*.png') if p.suffix.lower() == '.png']
                if images:
                    logging.info(f"Found category: {category.name}, images: {[img.name for img in images]}")
                    self.foregrounds_dict[category.name] = images
                    category_count += 1
                    image_count += len(images)
        if not self.foregrounds_dict:
            # This might happen if the directory exists but is empty or has no subdirs
             logging.warning(f"No valid foreground categories/images found in {self.foregrounds_dir}")
             # raise ValueError(f"No valid foreground images found in {self.foregrounds_dir}") # Don't raise error here
        else:
            logging.info(f"Processed {image_count} foreground images across {category_count} categories.")


    def _process_backgrounds(self) -> None:
        # Find common image formats, case-insensitive
        self.background_images: List[Path] = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
             self.background_images.extend(list(self.backgrounds_dir.glob(ext)))
             # Check uppercase extensions too if needed on case-sensitive systems
             # self.background_images.extend(list(self.backgrounds_dir.glob(ext.upper())))

        if not self.background_images:
             logging.warning(f"No valid background images (.png, .jpg, .jpeg) found in {self.backgrounds_dir}")
             # raise ValueError(f"No valid background images found in {self.backgrounds_dir}") # Don't raise error
        else:
            logging.info(f"Found {len(self.background_images)} background images.")

    def _validate_output_directory(self) -> None:
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # No need to check if empty, we'll overwrite or add

    def _load_augmentation_pipeline(self) -> None:
        self.transforms = None # Default to no transforms
        if self.augmentation_path and self.augmentation_path.is_file():
            if self.augmentation_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    self.transforms = A.load(str(self.augmentation_path), data_format='yaml')
                    logging.info(f"Loaded augmentations from: {self.augmentation_path}")
                except Exception as e:
                    logging.error(f"Error loading augmentation file {self.augmentation_path}: {e}")
                    warnings.warn(f'Could not load augmentation file {self.augmentation_path}. No augmentations applied.')
            elif self.augmentation_path.suffix.lower() == '.json':
                 try:
                    self.transforms = A.load(str(self.augmentation_path), data_format='json')
                    logging.info(f"Loaded augmentations from: {self.augmentation_path}")
                 except Exception as e:
                    logging.error(f"Error loading augmentation file {self.augmentation_path}: {e}")
                    warnings.warn(f'Could not load augmentation file {self.augmentation_path}. No augmentations applied.')
            else:
                 warnings.warn(f'Unsupported augmentation file format: {self.augmentation_path.suffix}. No augmentations applied.')
        elif self.augmentation_path:
             warnings.warn(f'Augmentation file not found at {self.augmentation_path}. No augmentations applied.')
        else:
            logging.info("No augmentation path provided. No augmentations will be applied.")


    def _generate_image(self, image_index: int, progress_callback=None) -> Optional[str]:
        """Generates a single image and its annotation."""
        # --- Input Validation for Generation ---
        if not self.background_images:
            logging.error(f"Cannot generate image {image_index}: No background images available.")
            return None
        if not self.foregrounds_dict:
             logging.error(f"Cannot generate image {image_index}: No foreground images/categories available.")
             return None

        try:
            # --- Select Background ---
            background_image_path = random.choice(self.background_images)
            try:
                background = Image.open(background_image_path).convert('RGBA')
            except (UnidentifiedImageError, FileNotFoundError) as e:
                 logging.error(f"Error opening background {background_image_path}: {e}. Skipping image {image_index}.")
                 return None
            background = background.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
            composite = background.copy() # Start with the resized background

            # --- Determine Number of Objects ---
            # Ensure max_objects is at least 1 if foregrounds exist
            max_objects = max(1, self.max_objects_per_image) if self.foregrounds_dict else 0
            num_foreground_images = random.randint(1, max_objects) if max_objects > 0 else 0

            # --- Prepare Annotations ---
            annotations = {
                'version': "1.0.0", # Add a version
                'flags': {},
                'shapes': [],
                'imagePath': f"{image_index:0{self.zero_padding}}.jpg", # Relative path
                'imageData': None, # Don't embed image data
                'imageHeight': self.image_height,
                'imageWidth': self.image_width
            }
            placed_bboxes = [] # To potentially check for collisions later

            # --- Select and Place Foreground Objects ---
            for _ in range(num_foreground_images):
                if not self.foregrounds_dict: break # Safety check
                category = random.choice(list(self.foregrounds_dict.keys()))
                if not self.foregrounds_dict[category]: continue # Skip if category somehow became empty

                fg_info = {'category': category, 'image_path': random.choice(self.foregrounds_dict[category])}

                try:
                    fg_image = Image.open(fg_info['image_path']).convert('RGBA')
                except (UnidentifiedImageError, FileNotFoundError) as e:
                    logging.warning(f"Could not open foreground {fg_info['image_path']}: {e}. Skipping this object.")
                    continue

                # --- Scaling ---
                # Simple random scale for now
                if not self.scaling_factors or len(self.scaling_factors) != 2:
                    logging.warning("Invalid scaling_factors provided, using default [0.2, 0.5].")
                    self.scaling_factors = [0.2, 0.5]
                scale = random.uniform(*self.scaling_factors)
                # Ensure scale doesn't make image zero size or too large
                min_dim = 10 # Minimum pixels allowed for width/height
                new_w = max(min_dim, int(fg_image.width * scale))
                new_h = max(min_dim, int(fg_image.height * scale))
                # Prevent exceeding background size significantly (e.g., max 95%)
                new_w = min(new_w, int(self.image_width * 0.95))
                new_h = min(new_h, int(self.image_height * 0.95))
                new_size = (new_w, new_h)

                try:
                    fg_image_resized = fg_image.resize(new_size, Image.Resampling.LANCZOS)
                except ValueError: # Handle potential zero size after int casting
                    logging.warning(f"Skipping object due to invalid resize dimensions: {new_size}")
                    continue

                # --- Augmentation ---
                if self.transforms:
                    try:
                        # Convert to numpy array for Albumentations
                        fg_np = np.array(fg_image_resized)
                        transformed = self.transforms(image=fg_np)
                        # Convert back to PIL Image
                        fg_processed = Image.fromarray(transformed['image'])
                    except Exception as aug_e:
                        logging.warning(f"Augmentation failed for {fg_info['image_path']}: {aug_e}. Using unaugmented.")
                        fg_processed = fg_image_resized
                else:
                    fg_processed = fg_image_resized

                # --- Placement ---
                # Basic random placement
                max_x = self.image_width - fg_processed.width
                max_y = self.image_height - fg_processed.height
                if max_x < 0 or max_y < 0:
                     logging.warning(f"Skipping object {fg_info['image_path']} - larger than background after scaling/augmentation.")
                     continue # Skip if object larger than background

                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

                # TODO: Implement collision detection if self.avoid_collisions is True
                # This requires comparing (x, y, w, h) with items in placed_bboxes

                # --- Pasting ---
                # Use the alpha channel as the mask
                try:
                     mask = fg_processed.getchannel('A')
                     composite.paste(fg_processed, (x, y), mask)
                except Exception as paste_e:
                     logging.error(f"Error pasting {fg_info['image_path']} at ({x},{y}): {paste_e}")
                     continue # Skip this object if pasting fails

                # --- Annotation (LabelMe format using BBox) ---
                # Bounding box coordinates (top-left x, y, bottom-right x, y)
                x_min, y_min = x, y
                x_max, y_max = x + fg_processed.width, y + fg_processed.height
                # LabelMe format expects list of points [[x1,y1], [x2,y2], ...]
                points = [
                    [float(x_min), float(y_min)],
                    [float(x_max), float(y_min)],
                    [float(x_max), float(y_max)],
                    [float(x_min), float(y_max)],
                ]
                annotation = {
                    'label': fg_info['category'],
                    'points': points,
                    'group_id': None, # LabelMe field
                    'shape_type': 'rectangle', # Specify shape type
                    'flags': {}
                }
                annotations['shapes'].append(annotation)
                placed_bboxes.append((x_min, y_min, x_max, y_max)) # Store for collision check


            # --- Save Composite Image and Annotation ---
            save_filename = f'{image_index:0{self.zero_padding}}'
            composite_path = self.output_dir / f'{save_filename}.jpg'
            annotation_path = self.output_dir / f'{save_filename}.json'

            # Convert final composite to RGB before saving as JPEG
            composite.convert('RGB').save(composite_path, format='JPEG', quality=95) # Specify quality

            # Save annotation file
            with open(annotation_path, 'w') as f:
                json.dump(annotations, f, indent=2) # Add indentation

            # Call progress callback if provided
            if progress_callback:
                progress_callback()

            return save_filename # Return filename prefix on success

        except Exception as e:
            logging.error(f"Unhandled error generating image {image_index}: {e}", exc_info=True)
            return None


    def generate_images(self, progress_callback=None) -> None:
        """Generates the specified number of images."""
        logging.info(f"Starting generation of {self.image_number} images...")
        logging.info(f"Output directory: {self.output_dir}")

        if self.parallelize:
             logging.info("Using parallel processing.")
             # Note: tqdm progress bar won't work directly with joblib like this
             # Consider using joblib's built-in progress display if needed, or remove tqdm wrapper
             results = Parallel(n_jobs=-1, backend="loky")( # Use loky backend for better robustness
                 delayed(self._generate_image)(i, None) for i in range(1, self.image_number + 1) # Cannot pass callback easily here
             )
             # Check results for errors (None indicates failure)
             errors = sum(1 for r in results if r is None)
             logging.info(f"Parallel generation complete. {self.image_number - errors} successful, {errors} errors.")

        else:
            logging.info("Using sequential processing.")
            error_count = 0
            # Wrap range with tqdm for progress bar
            for i in tqdm(range(1, self.image_number + 1), desc="Generating Images", total=self.image_number):
                result = self._generate_image(i, progress_callback)
                if result is None:
                    error_count += 1
            logging.info(f"Sequential generation complete. {self.image_number - error_count} successful, {error_count} errors.")

# --- END OF SyntheticImageGenerator class ---

# NOTE: The if __name__ == '__main__': block should be REMOVED from this file
# and placed in app.py or a separate CLI script if needed.