from typing import Dict, List, Set, Tuple
import torch
import numpy as np

def process_segment_scores(
    scores: torch.Tensor, 
    parent_mapping: Dict[str, List[int]],
    combine_method: str = "max"
) -> torch.Tensor:
    """
    Process the scores matrix for segmented images by combining scores 
    for segments from the same original image.
    
    Args:
        scores: Original score matrix of shape [num_queries, num_segments]
        parent_mapping: Mapping from original image IDs to indices of their segments
        combine_method: Method to combine segment scores ('max', 'avg', 'sum')
        
    Returns:
        A new score matrix of shape [num_queries, num_original_images]
    """
    num_queries = scores.shape[0]
    num_original_images = len(parent_mapping)
    
    # Create a new scores matrix
    new_scores = torch.zeros((num_queries, num_original_images), dtype=scores.dtype)
    
    # Process each original image
    for new_idx, (_, segment_indices) in enumerate(parent_mapping.items()):
        # Extract scores for all segments of this original image
        segment_scores = scores[:, segment_indices]
        
        # Combine scores based on specified method
        if combine_method == "max":
            combined_score = torch.max(segment_scores, dim=1)[0]
        elif combine_method == "avg":
            combined_score = torch.mean(segment_scores, dim=1)
        elif combine_method == "sum":
            combined_score = torch.sum(segment_scores, dim=1)
        else:
            raise ValueError(f"Unsupported combine method: {combine_method}")
        
        # Assign combined score to the original image
        new_scores[:, new_idx] = combined_score
    
    return new_scores

def map_segment_indices_to_parent(
    indices: np.ndarray, 
    parent_mapping: Dict[str, List[int]]
) -> Tuple[np.ndarray, Dict[int, Set[int]]]:
    """
    Map segment indices in the result matrix to their parent image indices.
    
    Args:
        indices: Array of segment indices from the retrieval results
        parent_mapping: Mapping from original image IDs to indices of their segments
        
    Returns:
        Tuple of (parent image indices array, mapping from new indices to original indices)
    """
    # Create reverse mapping: segment index -> original image ID
    segment_to_parent = {}
    parent_to_new_idx = {}
    
    for new_idx, (parent_id, segment_indices) in enumerate(parent_mapping.items()):
        parent_to_new_idx[parent_id] = new_idx
        for seg_idx in segment_indices:
            segment_to_parent[seg_idx] = parent_id
    
    # Map each result index to its parent's new index
    new_indices = np.zeros_like(indices)
    
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            seg_idx = indices[i, j]
            parent_id = segment_to_parent.get(seg_idx)
            if parent_id is not None:
                new_indices[i, j] = parent_to_new_idx[parent_id]
            else:
                new_indices[i, j] = seg_idx  # Keep original if no mapping
    
    return new_indices, parent_to_new_idx
