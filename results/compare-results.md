## Comparison Results

Datasets: `["arxivqa", "docvqa", "tabfquad", "tatdqa"]`

Models: 

```python
{
    'blip2-flan-t5-xl',
    'blip_Salesforce_blip-vqa-base',
    'clip_openai_clip-vit-base-patch32',
    'jina-clip-v1_jinaai_jina-clip-v1',
    'llava-interleave_llava-hf_llava-interleave-qwen-0.5b-hf',
    #'scan',
    'vit_google_vit-base-patch16-224-in21k'
}
```

Performance Changes (After segmenting inputs)

<!-- Include the result tables from ./exp2-results/ -->

| Model                                                           |   ArxivQ |   DocQ |   InfoQ |   TabF |   TATQ |   Shift |   AI |   Energy |   Gov. |   Health |
|:----------------------------------------------------------------|---------:|-------:|--------:|-------:|-------:|--------:|-----:|---------:|-------:|---------:|
| blip2-flan-t5-xl_seg_2x2                                        |    0.726 |  0.554 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| blip_Salesforce_blip-vqa-base_seg_2x2                           |    0.4   |  0.362 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| clip_openai_clip-vit-base-patch32_seg_2x2                       |    6.536 |  4.387 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| jina-clip-v1_jinaai_jina-clip-v1_seg_2x2                        |   27.024 | 19.929 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| llava-interleave_llava-hf_llava-interleave-qwen-0.5b-hf_seg_2x2 |    0.2   |  1.027 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| siglip_google_siglip-so400m-patch14-384_seg_2x2                 |   35.512 | 23.767 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| vit_google_vit-base-patch16-224-in21k_seg_2x2                   |    0.752 |  0.472 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |

| Model                                                           |   ArxivQ |   DocQ |   InfoQ |   TabF |   TATQ |   Shift |   AI |   Energy |   Gov. |   Health |
|:----------------------------------------------------------------|---------:|-------:|--------:|-------:|-------:|--------:|-----:|---------:|-------:|---------:|
| blip2-flan-t5-xl_seg_2x2                                        |    0.898 |  0.65  |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| blip_Salesforce_blip-vqa-base_seg_2x2                           |    0.564 |  0.553 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| clip_openai_clip-vit-base-patch32_seg_2x2                       |    7.594 |  4.75  |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| jina-clip-v1_jinaai_jina-clip-v1_seg_2x2                        |   29.132 | 21.551 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| llava-interleave_llava-hf_llava-interleave-qwen-0.5b-hf_seg_2x2 |    0.277 |  1.294 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| siglip_google_siglip-so400m-patch14-384_seg_2x2                 |   38.36  | 26.381 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| vit_google_vit-base-patch16-224-in21k_seg_2x2                   |    1.002 |  0.472 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |

| Model                                                           |   ArxivQ |   DocQ |   InfoQ |   TabF |   TATQ |   Shift |   AI |   Energy |   Gov. |   Health |
|:----------------------------------------------------------------|---------:|-------:|--------:|-------:|-------:|--------:|-----:|---------:|-------:|---------:|
| blip2-flan-t5-xl_seg_2x2                                        |    0.633 |  0.517 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| blip_Salesforce_blip-vqa-base_seg_2x2                           |    0.333 |  0.333 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| clip_openai_clip-vit-base-patch32_seg_2x2                       |    6.1   |  3.769 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| jina-clip-v1_jinaai_jina-clip-v1_seg_2x2                        |   26.133 | 18.551 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| llava-interleave_llava-hf_llava-interleave-qwen-0.5b-hf_seg_2x2 |    0.2   |  0.924 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| siglip_google_siglip-so400m-patch14-384_seg_2x2                 |   33.7   | 22.395 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| vit_google_vit-base-patch16-224-in21k_seg_2x2                   |    0.667 |  0.407 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |

| Model                                                           |   ArxivQ |   DocQ |   InfoQ |   TabF |   TATQ |   Shift |   AI |   Energy |   Gov. |   Health |
|:----------------------------------------------------------------|---------:|-------:|--------:|-------:|-------:|--------:|-----:|---------:|-------:|---------:|
| blip2-flan-t5-xl_seg_2x2                                        |    0.733 |  0.573 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| blip_Salesforce_blip-vqa-base_seg_2x2                           |    0.423 |  0.443 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| clip_openai_clip-vit-base-patch32_seg_2x2                       |    6.68  |  3.969 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| jina-clip-v1_jinaai_jina-clip-v1_seg_2x2                        |   27.283 | 19.438 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| llava-interleave_llava-hf_llava-interleave-qwen-0.5b-hf_seg_2x2 |    0.24  |  1.068 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| siglip_google_siglip-so400m-patch14-384_seg_2x2                 |   35.26  | 23.825 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |
| vit_google_vit-base-patch16-224-in21k_seg_2x2                   |    0.807 |  0.407 |       0 |      0 |      0 |       0 |    0 |        0 |      0 |        0 |

![improvement-heatmap](../figs/exp2/improvement-heatmap.png)

![improvement-heatmap](../figs/exp2/improvement-by-datasets.png)
