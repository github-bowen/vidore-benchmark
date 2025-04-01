from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.utils.data_utils import deduplicate_dataset_rows, get_parent_image_mapping
from vidore_benchmark.evaluation.segment_utils import process_segment_scores
from loguru import logger

class ViDoReEvaluatorQA(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a question-answering (QA) format, i.e. where each
    row in the dataset contains an optional query and a passage (image or text).
    """

    def __init__(self, vision_retriever: BaseVisionRetriever):
        super().__init__(vision_retriever=vision_retriever)

        # Dataset column names
        self.query_column = "query"
        self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
        self.passage_filename_column = "image_filename"
        self.id_column = "id"
        self.image_hash_column = "image_hash"

    def evaluate_dataset(
        self,
        ds: Dataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_query: Optional[int] = None,
        dataloader_prebatch_passage: Optional[int] = None,
        combine_method: str = "max",
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate a dataset with a Question Answering (QA) format.

        Important notes:
        - In the current ViDoRe Benchmark, queries are deduplicated.
        - In the next iteration of the ViDoRe Benchmark, we will allow for multiple queries per passage using
          the BEIR format.
        """
        # Preprocess the dataset, get qrels, and deduplicate the queries and passages
        ds = ds.map(lambda example, idx: {self.id_column: idx}, with_indices=True)

        ds_passages = ds.remove_columns(
            [col for col in ds.column_names if col not in [self.passage_column, self.image_hash_column, self.id_column, 
                                                          "original_image_id", "segment_idx", "total_segments", "grid_size"]]
        )
        ds_queries = ds.remove_columns(
            [col for col in ds.column_names if col not in [self.query_column, self.id_column]]
        )
        ds_queries = deduplicate_dataset_rows(ds=ds_queries, target_column=self.query_column)

        queries = list(ds_queries[self.query_column])

        if len(ds_queries) == 0:
            raise ValueError("No valid queries found in the dataset. Check if the queries are all set to `None`.")

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            scores = self.vision_retriever.get_scores_bm25(
                queries=ds_queries[self.query_column],
                passages=ds_passages[self.passage_column],
            )
            relevant_docs, results = self._get_relevant_docs_results(
                ds=ds,
                queries=queries,
                scores=scores,
            )
            metrics = self.compute_retrieval_scores(qrels=relevant_docs, results=results)
            return metrics

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds=ds_passages,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        # Get the similarity scores
        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )
        
        # Check if the dataset contains segmented images
        if "original_image_id" in ds_passages.column_names:
            logger.info(f"Detected segmented images. Processing segment scores with {combine_method} method...")
            
            # Create a mapping from original image IDs to segment indices
            parent_mapping = get_parent_image_mapping(ds_passages)
            
            # Process the scores to combine segments from the same original image
            original_scores = scores
            scores = process_segment_scores(
                scores=scores, 
                parent_mapping=parent_mapping,
                combine_method=combine_method  # Use the provided combine method
            )
            
            logger.info(f"Processed scores from {original_scores.shape} to {scores.shape} after combining segments using {combine_method} method")
            
            # Create a de-segmented version of the dataset for producing results
            # We need to filter the dataset to only keep one row per original image
            original_image_ids = list(parent_mapping.keys())
            ds_filtered = ds.filter(
                lambda example: example.get("original_image_id") in original_image_ids and 
                               example.get("segment_idx", 0) == 0  # Keep only the first segment of each image
            )
            ds = ds_filtered

            # log out current dataset shape
            logger.debug(f"Original dataset shape: {ds_passages.shape}")
            logger.debug(f"Filtered dataset shape: {ds.shape}")
            
            # Create new passage dataset with original images only
            ds_passages = ds.remove_columns(
                [col for col in ds.column_names if col not in [self.passage_column, self.image_hash_column, self.id_column]]
            )

        # Get the relevant passages and results
        relevant_docs, results = self._get_relevant_docs_results(
            ds=ds,
            queries=queries,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(
            qrels=relevant_docs,
            results=results,
            ignore_identical_ids=True,
        )

        return metrics

    def _get_relevant_docs_results(
        self,
        ds: Dataset,
        queries: List[str],
        scores: torch.Tensor,
        **kwargs,
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
        """
        Get the relevant passages and the results from the scores.

        Outputs:
        - relevant_docs: Dict[str, float]
        {
            "query_0": {"doc_0": 1},
            "query_1": {"doc_1": 1},
            ...
        }
        - results: Dict[str, Dict[str, float]] with shape:
        {
            "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
            "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
            ...
        }
        """
        relevant_docs = {}
        results = {}

        queries2filename = {
            query: image_filename
            for query, image_filename in zip(ds[self.query_column], ds[self.passage_filename_column])
        }
        passages2filename = {
            docidx: image_filename for docidx, image_filename in enumerate(ds[self.passage_filename_column])
        }

        for query, score_per_query in zip(queries, scores):
            relevant_docs[query] = {queries2filename[query]: 1}

            for docidx, score in enumerate(score_per_query):
                filename = passages2filename[docidx]
                score_passage = float(score.item())

                if query in results:
                    results[query][filename] = max(results[query].get(filename, 0), score_passage)
                else:
                    results[query] = {filename: score_passage}

        return relevant_docs, results
