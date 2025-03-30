@echo off

echo Running command 1:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class llava-onevision --model-name llava-hf/llava-onevision-qwen2-0.5b-ov-hf --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir newOutput/

echo Running command 2:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class blip --model-name Salesforce/blip-vqa-base --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir newOutput/

echo Running command 3:
python -m vidore_benchmark.cli.main evaluate-retriever --model-class blip --model-name Salesforce/blip2-opt-2.7b --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --dataset-format qa --split test --output-dir newOutput/

echo All commands executed.
pause
