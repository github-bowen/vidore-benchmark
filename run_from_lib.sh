vidore-benchmark evaluate-retriever \
    --model-class test_retriever \
    --model-name test_retriever \
    --dataset-name vidore/docvqa_test_subsampled \
    --dataset-format qa \
    --split test

# export LD_LIBRARY_PATH=/home/bowen/miniconda3/lib:$LD_LIBRARY_PATH 
vidore-benchmark evaluate-retriever \
    --model-class jina-clip-v1 \
    --model-name jinaai/jina-clip-v1 \
    --dataset-name vidore/arxivqa_test_subsampled \
    --dataset-format qa \
    --split test \
    --output-dir newOutput/

vidore-benchmark evaluate-retriever \
    --model-class jina-clip-v1 \
    --model-name jinaai/jina-clip-v1 \
    --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d \
    --dataset-format qa \
    --split test \
    --output-dir newOutput/
    

# Available models: ['bge-m3-colbert', 'bge-m3', 'biqwen2', 'bm25', 'cohere', 'colidefics3', 'colpali', 
#                    'colqwen2', 'dse-qwen2', 'dummy_vision_retriever', 'jina-clip-v1', 'nomic-embed-vision', 'siglip']