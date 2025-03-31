from __future__ import annotations

import math
from typing import List, Optional, Union, cast

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2Model
from transformers import AutoModel, AutoProcessor

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device

@register_vision_retriever("blip2")
class BLIP2Retriever(BaseVisionRetriever):
    """
    BLIP2Retriever class to retrieve embeddings using the BLIP-2 model.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "Salesforce/blip2-flan-t5-xl",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
        self.device = get_torch_device(device)

        self.model = Blip2Model.from_pretrained(pretrained_model_name_or_path).to(self.device).eval()
        self.processor = Blip2Processor.from_pretrained(pretrained_model_name_or_path)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []

        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            inputs_queries = self.processor.tokenizer(
                text=query_batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            if "decoder_input_ids" not in inputs_queries:
                inputs_queries["decoder_input_ids"] = inputs_queries["input_ids"]
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**inputs_queries, return_dict=True, output_hidden_states=True)
                query_embeddings = text_outputs.encoder_last_hidden_state[:,0]
                list_emb_queries.extend(list(torch.unbind(query_embeddings, dim=0)))

        return list_emb_queries

    '''
    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[Image.Image], passage_batch)
            list_doc = [document.convert("RGB") for document in passage_batch if isinstance(document, Image.Image)]

            input_image_processed = self.processor(images=list_doc, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                vision_outputs = self.model.get_image_features(**input_image_processed, return_dict=True, output_hidden_states=True)
                passage_embeddings = vision_outputs.last_hidden_state[:, 0]
                # query_features = self.model.query_transformer(inputs_embeds=passage_embeddings)
                # passage_embedding = query_features.mean(dim=1)
                list_emb_passages.extend(list(torch.unbind(passage_embeddings, dim=0)))

        return list_emb_passages'
    '''
    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[Image.Image], passage_batch)
            list_doc = [document.convert("RGB") for document in passage_batch if isinstance(document, Image.Image)]

            # Preprocess images
            input_image_processed = self.processor(images=list_doc, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                # Step 1: Extract image embeddings using the vision model
                vision_outputs = self.model.vision_model(
                    pixel_values=input_image_processed["pixel_values"],
                    return_dict=True,
                )
                image_embeds = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches, vision_hidden_size)
                
                # Step 2: Pass image embeddings through Q-Former
                image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                query_output = query_outputs.last_hidden_state  # Shape: (batch_size, num_query_tokens, qformer_hidden_size)

                # Step 3: Project Q-Former outputs to the text embedding dimension
                passage_embeddings = self.model.language_projection(query_output)  # Shape: (batch_size, num_query_tokens, text_hidden_size)
                passage_embeddings = passage_embeddings.mean(dim=1)  # Pool across query tokens to get (batch_size, text_hidden_size)

                # Collect passage embeddings
                list_emb_passages.extend(list(torch.unbind(passage_embeddings, dim=0)))

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
            print(query_embeddings.shape)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)
            print(passage_embeddings.shape)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores