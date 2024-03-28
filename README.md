# silio-llm

# Sequence Classification

## Training
Here is a simple sequence classification pipeline using MLFlow, which supports (1) model training; (2) model evaluation; (3) model registering on Azure Machine Learning.

Submit MLFlow experiment job to Azure ML from local path
```bash
pip install -U azureml-mlflow

cd ./azure/

python submit.py \
--workspace_name=<workspace-name> \
--subscription_id=<subscription-id> \
--resource_group=<resource-group> \
--experiment_name=<experiment-name> \
--compute_target=<compute-target> \
--num_gpus=1 \
--datastore_name=<datastore-name> \
--model_name_or_path=bert-base-uncased \
--train_blob_path=<train-blob-path> \
--train_file=train.json \
--dev_blob_path=<dev-blob-path> \
--dev_file=dev.json \
--test_blob_path=<test-blob-path> \
--test_file=test.json \
--max_seq_length=128 \
--batch_size=16 \
--gradient_accumulation_steps=1 \
--learning_rate=3e-05 \
--num_epochs=3.0 \
--weight_decay=0.01 \
--warmup_steps=1000 \
--eval_steps=500 \
--output_dir=./output/ \
--seed=42 \
--save_total_limit=10 \
--registered_name=<registered-model-name>
```

Track job progress and review results on Azure ML
<p align="center"><img src="/doc/azure.training.detail.png" width=2000></p>
<p align="center"><i>Figure 8: the details of training</i></p>

<p align="center"><img src="/doc/azure.evaluation.report.png" width=800></p>
<p align="center"><i>Figure 9: the classification report for evaluation</i></p>

<p align="center"><img src="/doc/azure.model.registry.png" width=2000></p>
<p align="center"><i>Figure 10: the model registry</i></p>
