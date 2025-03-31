from __future__ import annotations

import math
from typing import List, Optional, Union, cast

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils.import_utils import is_flash_attn_2_available

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("llava-interleave")
class LlavaInterleaveRetriever(BaseVisionRetriever):
    """
    LlavaInterleaveRetriever class to retrieve embeddings from the LLaVA Interleave model.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "llava-hf/llava-interleave-qwen-0.5b-hf",
        device: str = "auto",
        embedding_dim: int = 512,  # Dimension to extract from the model
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)

        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[transformers]"` '
                "to use LlavaInterleaveRetriever."
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = get_torch_device(device)
        self.embedding_dim = embedding_dim

        # Load model and processor
        self.model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        
        # Set required processor attributes to fix the warning
        self.processor.patch_size = 14  # For Siglip Vision model used in llava-interleave
        self.processor.vision_feature_select_strategy = "patch"
        
        # Create a small dummy image for text-only queries
        self.dummy_image = Image.new('RGB', (32, 32), (128, 128, 128))

    def __getstate__(self):
        """Custom method for pickle serialization"""
        state = self.__dict__.copy()
        # Remove non-serializable components
        state.pop('model', None)
        state.pop('processor', None)
        state.pop('dummy_image', None)
        return state

    def __setstate__(self, state):
        """Custom method for pickle deserialization"""
        self.__dict__.update(state)
        # Reinitialize the non-serializable components
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)
        
        # Set required processor attributes to fix the warning
        self.processor.patch_size = 14  # For Siglip Vision model used in llava-interleave
        self.processor.vision_feature_select_strategy = "patch"
        
        # Create a small dummy image for text-only queries
        self.dummy_image = Image.new('RGB', (32, 32), (128, 128, 128))

    def get_embedding(self, model_output, normalize=True):
        """Extract embeddings from the model output"""
        # Get the last hidden state
        hidden_states = model_output.hidden_states[-1]
        # Use the last token embedding
        embeddings = hidden_states[:, -1, :self.embedding_dim]
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        """Process text queries by including a dummy image to prevent None image features"""
        list_emb_queries: List[torch.Tensor] = []

        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            
            # Create a batch of conversations
            conversations = []
            for query in query_batch:
                # Create a conversation with the query and a tiny dummy image
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            # Include dummy image first to ensure proper processing
                            {"type": "image"},
                            {"type": "text", "text": f"Query: {query}"}
                        ],
                    }
                ])
            
            # Apply chat template to all conversations
            prompts = [self.processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
            
            # Process all queries in the batch at once
            inputs = self.processor(
                images=[self.dummy_image] * len(query_batch),
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Forward pass for the whole batch
            with torch.no_grad():
                outputs = self.model(
                    **inputs, 
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get embeddings for the batch
                embeddings = self.get_embedding(outputs)
                list_emb_queries.extend(embeddings.cpu())

        return list_emb_queries

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        """Process image passages with a standard prompt"""
        list_emb_passages: List[torch.Tensor] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[Image.Image], passage_batch)
            
            # Make sure images are in RGB format
            processed_images = [image.convert("RGB") for image in passage_batch]
            
            # Create a standard conversation template for all images
            conversations = []
            for _ in range(len(processed_images)):
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            # Image first, then text
                            {"type": "image"},
                            {"type": "text", "text": "Describe this image in detail."},
                        ],
                    }
                ])
            
            # Apply chat template to all conversations
            prompts = [self.processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
            
            # Process all images in the batch at once
            inputs = self.processor(
                images=processed_images,
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Forward pass for the whole batch
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get embeddings for the batch
                embeddings = self.get_embedding(outputs)
                list_emb_passages.extend(embeddings.cpu())

        return list_emb_passages

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores
