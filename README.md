# experiment-pipeline

```python
python hyperparameter_search.py --experiment_name debug_trial --result_dir ~/trial --data_type sequence --model_architecture simple_cnn --train_csv ~/Desktop/Datasets/PNAS-splicing-assay/split_data/train_dataset.csv --test_csv ~/Desktop/Datasets/PNAS-splicing-assay/split_data/test_dataset.csv --target_column PSI --sequence_column exon --loss kl_divergence --metrics pearson_logits_r --num_trials 2 --num_cpus 1 --upstream_sequence CATCCAGGTT --downstream_sequence CAGGTCTGAC
```