# Detailed Analysis of VLMs for Cross-Modal Document Retrieval

## About This Project

This project was developed for the TU Delft **DSAIT4050 Information Retrieval course** (2025). It conducts a comprehensive analysis of various Vision-Language Models (VLMs) in cross-modal document retrieval tasks using the Vidore Benchmark. Our work extends the original benchmark with additional retrievers and significant code modifications to enable more thorough performance evaluation.

## Main Contributions

- Implementation of several **custom retrievers** to improve evaluation capabilities
- Modification of the benchmark source code to support **image segmentation** as a **preprocessing** step
- Significant performance improvements in retrieval tasks through image partitioning techniques
- Comprehensive evaluation and analysis of different retrieval approaches

## Acknowledgement

This project is a fork of [illuin-tech/vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark). We thank the original authors for their valuable work.

## Detailed Modifications

### 1. Custom Retrievers

In this fork, we have added custom retrievers for specific testing purposes. You can find the implementation of these retrievers under `src/vidore_benchmark/retrievers` directory. These new retrievers leverage advanced techniques to significantly improve retrieval performance compared to baseline models, especially when combined with our image segmentation preprocessing.

### 2. Image Segmentation

This fork includes image segmentation functionality that enhances retrieval performance by dividing images into smaller segments for more granular analysis. The image segmentation parameters can be used during evaluation:

- `--image-segmentation`: Activates the image segmentation pipeline (boolean flag)
- `--grid-rows`: Number of rows for dividing the image (default: 2)
- `--grid-cols`: Number of columns for dividing the image (default: 2)
- `--overlap`: Overlap percentage between segments, ranging from 0.0 to 0.5 (default: 0.0)

Example usage:

```bash
python -m vidore_benchmark.cli.main evaluate-retriever \
    --model-class jina-clip-v1 \
    --model-name jinaai/jina-clip-v1 \
    --image-segmentation \
    --grid-rows 2 \
    --grid-cols 2 \
    --dataset-name vidore/arxivqa_test_subsampled \
    --dataset-format qa \
    --split test \
    --output-dir segmentedOutput/
    # --overlap 0 \
```

When image segmentation is enabled, the evaluation results will include a suffix in the model ID indicating the segmentation configuration (e.g., `YourModel_seg_3x3_overlap_20`).

## Run in Colab

You can run this benchmark in Google Colab using the following link: [Run in Colab](https://colab.research.google.com/drive/1YbxLDcbjrOwJC56gDXuA7fl31YkEyRsO?usp=sharing)

## Results

See folder [results](./results/).