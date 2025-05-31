---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1660
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: the rock workout routine
  sentences:
  - Dwayne Johnson's intense workout routine and diet plan for his role in Black Adam,
    including his famous 'Black Adam' workout split and high-protein meal plan.
  - A complete list of Taylor Swift's studio albums in release order from her 2006
    debut to 2024's The Tortured Poets Department.
  - Ichiran Ramen in Shibuya is known for its solo dining booths and tonkotsu ramen.
- source_sentence: famous landmarks in Paris
  sentences:
  - The Eiffel Tower is Paris' most iconic monument and a must-see attraction.
  - Nakiryu, one of Tokyo's Michelin-starred ramen shops, offers a signature dan dan
    noodle dish.
  - The Eiffel Tower is Paris' most iconic monument and a must-see attraction.
- source_sentence: famous landmarks in Paris
  sentences:
  - The Louvre Museum is the world's largest art museum and home to the Mona Lisa.
  - A complete list of Taylor Swift's studio albums in release order from her 2006
    debut to 2024's The Tortured Poets Department.
  - The Louvre Museum is the world's largest art museum and home to the Mona Lisa.
- source_sentence: the rock workout routine
  sentences:
  - SpaceX, Tesla, Neuralink, and The Boring Company are all companies founded or
    led by Elon Musk.
  - Dwayne Johnson's intense workout routine and diet plan for his role in Black Adam,
    including his famous 'Black Adam' workout split and high-protein meal plan.
  - The Eiffel Tower is Paris' most iconic monument and a must-see attraction.
- source_sentence: Taylor Swift albums
  sentences:
  - Dwayne Johnson's intense workout routine and diet plan for his role in Black Adam,
    including his famous 'Black Adam' workout split and high-protein meal plan.
  - Nakiryu, one of Tokyo's Michelin-starred ramen shops, offers a signature dan dan
    noodle dish.
  - A complete list of Taylor Swift's studio albums in release order from her 2006
    debut to 2024's The Tortured Poets Department.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-mpnet-base-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.5
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 1.0
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 1.0
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 1.0
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.5
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.33333333333333337
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.2
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.1
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.5
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 1.0
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 1.0
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 1.0
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.8154648767857289
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.75
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.75
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision 12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Taylor Swift albums',
    "A complete list of Taylor Swift's studio albums in release order from her 2006 debut to 2024's The Tortured Poets Department.",
    "Nakiryu, one of Tokyo's Michelin-starred ramen shops, offers a signature dan dan noodle dish.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.5        |
| cosine_accuracy@3   | 1.0        |
| cosine_accuracy@5   | 1.0        |
| cosine_accuracy@10  | 1.0        |
| cosine_precision@1  | 0.5        |
| cosine_precision@3  | 0.3333     |
| cosine_precision@5  | 0.2        |
| cosine_precision@10 | 0.1        |
| cosine_recall@1     | 0.5        |
| cosine_recall@3     | 1.0        |
| cosine_recall@5     | 1.0        |
| cosine_recall@10    | 1.0        |
| **cosine_ndcg@10**  | **0.8155** |
| cosine_mrr@10       | 0.75       |
| cosine_map@100      | 0.75       |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,660 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                      | sentence_1                                                                       | label                                                         |
  |:--------|:--------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                          | string                                                                           | float                                                         |
  | details | <ul><li>min: 5 tokens</li><li>mean: 6.23 tokens</li><li>max: 7 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 26.5 tokens</li><li>max: 37 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                             | sentence_1                                                                                                                                 | label            |
  |:---------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Taylor Swift albums</code>       | <code>A complete list of Taylor Swift's studio albums in release order from her 2006 debut to 2024's The Tortured Poets Department.</code> | <code>1.0</code> |
  | <code>Taylor Swift albums</code>       | <code>A complete list of Taylor Swift's studio albums in release order from her 2006 debut to 2024's The Tortured Poets Department.</code> | <code>1.0</code> |
  | <code>famous landmarks in Paris</code> | <code>The Eiffel Tower is Paris' most iconic monument and a must-see attraction.</code>                                                    | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | cosine_ndcg@10 |
|:------:|:----:|:--------------:|
| 0.9615 | 100  | 0.8155         |


### Framework Versions
- Python: 3.13.2
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.7.0
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->