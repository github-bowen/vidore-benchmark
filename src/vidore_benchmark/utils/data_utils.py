from loguru import logger
import os
from pathlib import Path
from typing import List, TypeVar, Tuple, Optional, Dict, Any, Union

import huggingface_hub
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import numpy as np
from tqdm import tqdm

T = TypeVar("T")



class ListDataset(TorchDataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


def deduplicate_dataset_rows(ds: HfDataset, target_column: str) -> HfDataset:
    """
    Remove duplicate rows from a dataset based on values in a target column.

    Args:
        ds (Dataset): The dataset to deduplicate.
        target_column (str): The column to use for deduplication.

    Returns:
        Dataset: The deduplicated dataset.
    """
    if target_column not in ds.column_names:
        raise ValueError(f"Column '{target_column}' not found in dataset.")

    seen_values = set()
    keep_mask = []

    for value in ds[target_column]:
        if value is None:
            keep_mask.append(False)
            continue

        if value not in seen_values:
            seen_values.add(value)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    return ds.select([i for i, keep in enumerate(keep_mask) if keep])


def get_datasets_from_collection(collection_name: str) -> List[str]:
    """
    Get dataset names from a local directory or a HuggingFace collection.

    Args:
        collection_name: Local dirpath or HuggingFace collection ID

    Returns:
        List of dataset names
    """
    if Path(collection_name).is_dir():
        logger.info(f"Loading datasets from local directory: `{collection_name}`")
        dataset_names = os.listdir(collection_name)
        dataset_names = [os.path.join(collection_name, dataset) for dataset in dataset_names]
    else:
        logger.info(f'Loading datasets from the Hf Hub collection: "{collection_name}"')
        collection = huggingface_hub.get_collection(collection_name)
        dataset_names = [dataset_item.item_id for dataset_item in collection.items]
    return dataset_names


def segment_image(
    image: Union[Image.Image, np.ndarray], 
    grid_size: Tuple[int, int] = (2, 2),
    overlap: float = 0.0
) -> List[Image.Image]:
    """
    Split an image into a grid of segments with optional overlap.
    
    Args:
        image: PIL Image or numpy array to segment
        grid_size: Tuple of (rows, cols) defining the grid
        overlap: Percentage of overlap between segments (0.0 to 0.5)
        
    Returns:
        List of PIL Image segments
    """
    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise ValueError("Image must be a PIL Image or numpy array")
    
    if overlap < 0 or overlap >= 0.5:
        raise ValueError("Overlap must be between 0 and 0.5")
    
    width, height = image.size
    rows, cols = grid_size
    
    # Calculate segment dimensions with overlap
    segment_width = width // cols
    segment_height = height // rows
    
    # Calculate overlap in pixels
    overlap_x = int(segment_width * overlap)
    overlap_y = int(segment_height * overlap)
    
    # Adjust segment size to include overlap
    segment_width_with_overlap = segment_width + 2 * overlap_x
    segment_height_with_overlap = segment_height + 2 * overlap_y
    
    segments = []
    
    for i in range(rows):
        for j in range(cols):
            # Calculate starting position with overlap
            left = max(0, j * segment_width - overlap_x)
            upper = max(0, i * segment_height - overlap_y)
            
            # Calculate ending position with overlap
            right = min(width, left + segment_width_with_overlap)
            lower = min(height, upper + segment_height_with_overlap)
            
            # Crop segment
            segment = image.crop((left, upper, right, lower))
            segments.append(segment)
    
    return segments

def process_dataset_with_segmentation(
    ds: HfDataset, 
    image_column: str = "image",
    grid_size: Tuple[int, int] = (2, 2),
    overlap: float = 0.0
) -> HfDataset:
    """
    Process a dataset by segmenting images in the specified column and expanding each row
    into multiple rows - one for each segment. All segments from the same image are
    tracked with a unique original_image_id.
    
    Args:
        ds: The dataset containing images to segment
        image_column: The column name containing images
        grid_size: Tuple of (rows, cols) defining the grid
        overlap: Percentage of overlap between segments (0.0 to 0.5)
        
    Returns:
        Processed dataset with expanded image segments
    """
    if image_column not in ds.column_names:
        raise ValueError(f"Column '{image_column}' not found in dataset.")
    
    # Create a list to hold all new rows
    all_rows = []
    
    # Keep track of dataset size for progress
    total_images = len(ds)
    total_segments = 0
    
    # Process each row in the dataset
    for idx, example in tqdm(enumerate(ds), total=total_images, desc="Segmenting images"):
        image = example[image_column]
        
        # Skip if not an image
        if not isinstance(image, (Image.Image, np.ndarray)):
            logger.warning(f"Skipping non-image data in column '{image_column}'")
            all_rows.append(example)
            continue
        
        # Segment the image
        segments = segment_image(image, grid_size=grid_size, overlap=overlap)
        
        # Generate a unique ID for the original image
        original_image_id = f"img_{idx}"
        
        # Create one row for each segment
        for seg_idx, segment in enumerate(segments):
            # Create a copy of the original row
            new_row = dict(example)
            
            # Replace the image with the segment
            new_row[image_column] = segment
            
            # Add metadata
            new_row["original_image_id"] = original_image_id
            new_row["segment_idx"] = seg_idx
            new_row["total_segments"] = len(segments)
            new_row["grid_size"] = grid_size
            
            # Add to our collection
            all_rows.append(new_row)
            total_segments += 1
    
    # Create a new dataset from all rows
    processed_ds = HfDataset.from_list(all_rows)
    
    logger.info(f"Processed {total_images} images into {total_segments} segments ({grid_size[0]}x{grid_size[1]} grid)")
    logger.debug(f"Before segmentation: dataset shape: {ds.shape}")
    logger.debug(f"After segmentation: dataset shape: {processed_ds.shape}")
    
    return processed_ds

# Define a helper function to map original segments to parent images
def get_parent_image_mapping(ds: HfDataset) -> Dict[str, List[int]]:
    """
    Create a mapping from original image IDs to the indices of their segments in the dataset.
    
    Args:
        ds: The segmented dataset with original_image_id field
        
    Returns:
        Dictionary mapping original image IDs to list of segment indices
    """
    if "original_image_id" not in ds.column_names:
        raise ValueError("Dataset does not appear to be segmented (no 'original_image_id' column found)")
    
    mapping = {}
    for idx, row in enumerate(ds):
        image_id = row["original_image_id"]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(idx)
    
    return mapping
