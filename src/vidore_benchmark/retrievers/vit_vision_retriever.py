from __future__ import annotations
from typing import List, Optional, Union
import torch
from PIL import Image
from transformers import BertTokenizer, BertModel, ViTModel, AutoImageProcessor
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.torch_utils import get_torch_device

@register_vision_retriever("vit")
class ViTVisionRetriever(BaseVisionRetriever):
    def __init__(
        self, 
        pretrained_model_name_or_path: str = "google/vit-base-patch16-224-in21k",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = get_torch_device(device)

        # 加载 Vision Transformer 模型用于处理图像
        self.model = ViTModel.from_pretrained(self.pretrained_model_name_or_path).to(self.device).eval()

        # 加载 BERT 模型用于处理文本查询
        self.text_model = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()

        # 加载 BERT Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 加载 AutoImageProcessor 用于图像预处理
        self.image_processor = AutoImageProcessor.from_pretrained(self.pretrained_model_name_or_path)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> torch.Tensor:
        """
        处理文本查询并生成查询的嵌入
        """
        # 对查询进行标记化
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 将输入数据移到设备上

        # 使用 BERT 模型生成文本嵌入
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # 获取 BERT 输出的最后一层隐藏状态（维度为 [batch_size, seq_len, hidden_dim]）
        last_hidden_states = outputs.last_hidden_state

        # 提取 [CLS] 位置的嵌入（通常用于表示整个句子的嵌入）
        cls_embeddings = last_hidden_states[:, 0, :]  # 形状为 [batch_size, hidden_dim]
        return cls_embeddings

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> torch.Tensor:
        """
        处理图像数据并生成视觉嵌入
        """
        visual_embeddings = []

        for image in passages:
            # 如果图像是灰度图像（2D），则将其转换为RGB图像（3D）
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 使用 AutoImageProcessor 将图像转换为模型的输入格式
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # 使用 ViT 模型生成图像的视觉嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取 ViT 输出的最后一层隐藏状态（维度为 [batch_size, seq_len, hidden_dim]）
            last_hidden_states = outputs.last_hidden_state

            # 提取 [CLS] 位置的嵌入
            cls_embeddings = last_hidden_states[:, 0, :]  # 形状为 [batch_size, hidden_dim]
            visual_embeddings.append(cls_embeddings)

        # 将所有视觉嵌入合并为一个张量
        visual_embeddings = torch.stack(visual_embeddings).to(self.device)
        return visual_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        计算查询嵌入与段落嵌入之间的点积相似度
        """
        # 如果查询嵌入是一个列表，则将其转换为张量
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings).to(self.device)

        # 如果段落嵌入是一个列表，则将其转换为张量
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings).to(self.device)

        # 确保查询和段落嵌入都是 2D 张量 [batch_size, embedding_dim]
        if query_embeddings.dim() == 3:
            query_embeddings = query_embeddings.squeeze(1)

        if passage_embeddings.dim() == 3:
            passage_embeddings = passage_embeddings.squeeze(1)

        # 使用点积计算查询与段落之间的相似度
        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores
