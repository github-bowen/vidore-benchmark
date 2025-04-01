@echo off

echo Evaluating blip:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class blip --model-name Salesforce/blip-vqa-base --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method max --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/

echo Evaluating jina-clip-v1:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class jina-clip-v1 --model-name jinaai/jina-clip-v1 --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method max --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/

echo Evaluating siglip:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class siglip --model-name google/siglip-so400m-patch14-384 --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method max --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/

echo Evaluating llava-interleave:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class llava-interleave --model-name llava-hf/llava-interleave-qwen-0.5b-hf --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method max --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/

echo Evaluating blip2:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class blip2 --model-name Salesforce/blip2-flan-t5-xl --image-segmentation --grid-rows 2 --grid-cols 2 --dataset-name vidore/arxivqa_test_subsampled --dataset-format qa --split test --output-dir segmentedOutput/


@REM echo Evaluating with different combining methods:
@REM echo Max:
@REM python -m vidore_benchmark.cli.main evaluate-retriever --model-class clip --model-name openai/clip-vit-base-patch32 --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method max --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/compare-methods/

@REM echo Avg:
@REM python -m vidore_benchmark.cli.main evaluate-retriever --model-class clip --model-name openai/clip-vit-base-patch32 --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method avg --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/compare-methods/

@REM echo Sum:
@REM python -m vidore_benchmark.cli.main evaluate-retriever --model-class clip --model-name openai/clip-vit-base-patch32 --image-segmentation --grid-rows 2 --grid-cols 2 --combine-method sum --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir segmentedOutput/compare-methods/

echo All commands executed.
pause
