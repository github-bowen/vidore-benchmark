from __future__ import annotations

from typing import List, Optional, Union

import torch
from PIL import Image

from transformers import AutoProcessor, CLIPModel

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.torch_utils import get_torch_device

@register_vision_retriever("clip")
class CLIPVisionRetriever(BaseVisionRetriever):
    def __init__(
        self, 
        pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = get_torch_device(device)

        self.model = (
            CLIPModel.from_pretrained(
                self.pretrained_model_name_or_path,
                trust_remote_code=True,  
            )
            .to(self.device)
            .eval()
        )

        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)

    def forward_queries(self, queries, batch_size, **kwargs):
        inputs = self.processor(text=queries, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device

        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)

        return text_embeddings

    def forward_passages(self, passages, batch_size, **kwargs):
        inputs = self.processor(images=passages, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device

        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)

        return image_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        Ensures that input embeddings are always tensors before applying matmul.
        """
        # Convert query embeddings to tensor if they are in list format
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings).to(self.device)  # Convert list to tensor and move to device
        
        # Convert passage embeddings to tensor if they are in list format
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings).to(self.device)  # Convert list to tensor and move to device

        # Ensure both query and passage embeddings are tensors before the matrix multiplication
        if not isinstance(query_embeddings, torch.Tensor):
            raise TypeError("query_embeddings must be a torch.Tensor or a list of torch.Tensors")
        
        if not isinstance(passage_embeddings, torch.Tensor):
            raise TypeError("passage_embeddings must be a torch.Tensor or a list of torch.Tensors")

        # Compute the dot-product similarity using the einsum function
        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores


