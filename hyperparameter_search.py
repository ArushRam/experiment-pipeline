import argparse
from hyperopt import hp
import numpy as np
import os
import random
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import torch

from experiment import Experiment

def initialize_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperparameter search.")

    experiment_args = parser.add_argument_group("experiment", "Experiment arguments")
    experiment_args.add_argument(
        "--experiment_name", type=str, required=True, help="Name of experiment."
    )
    experiment_args.add_argument(
        "--experiment_seed",
        type=int,
        default=42,
        required=False,
        help="Experiment seed.",
    )
    experiment_args.add_argument(
        "--num_trials",
        type=int,
        required=True,
        help="Number of trials to run in experiment.",
    )
    experiment_args.add_argument(
        "--num_concurrent",
        type=int,
        required=False,
        default=None,
        help="Maximum number of trials to run concurrently.",
    )
    experiment_args.add_argument(
        "--time_budget_s",
        type=int,
        default=None,
        required=False,
        help="Time budget for experiment in seconds.",
    )
    experiment_args.add_argument(
        "--result_dir",
        type=str,
        default="~/ray_results",
        required=False,
        help="Directory of results.",
    )
    experiment_args.add_argument(
        "--restore",
        type=bool,
        default=False,
        required=False,
        help="Whether to restore this experiment from a previous interrupted run.",
    )

    data_args = parser.add_argument_group("data", "Data arguments")
    data_args.add_argument(
        "--data_type", type=str, required=True, help="Type of data to use."
    )
    data_args.add_argument(
        "--train_csv", type=str, required=True, help="Path to training CSV."
    )
    data_args.add_argument(
        "--train_fraction",
        type=float,
        required=False,
        default=0.2,
        help="Fraction of training data to use for training.",
    )
    data_args.add_argument(
        "--test_csv", type=str, required=True, help="Path to test CSV."
    )
    data_args.add_argument(
        "--sequence_column", type=str, required=True, help="Name of sequence column."
    )
    data_args.add_argument(
        "--structure_column",
        type=str,
        required=False,
        help="Name of secondary structure column.",
    )
    data_args.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of prediction target column.",
    )
    data_args.add_argument(
        "--upstream_sequence", type=str, required=False, help="Upstream sequence.",
    )
    data_args.add_argument(
        "--downstream_sequence", type=str, required=False, help="Downstream sequence.",
    )

    resource_args = parser.add_argument_group("resources", "Resource arguments")
    resource_args.add_argument(
        "--num_cpus", type=int, required=False, help="Number of CPUs to use", default=1
    )
    resource_args.add_argument(
        "--num_cpus_per_trial",
        type=float,
        required=False, default=1,
        help="Number of CPUs to use for each trial.",
    )
    resource_args.add_argument(
        "--num_gpus_per_trial",
        type=float,
        required=False, default=0,
        help="Number of GPUs to use for each trial.",
    )
    resource_args.add_argument(
        "--amp",
        type=bool,
        default=False,
        required=False,
        help="Whether to use automatic mixed precision for training",
    )
    resource_args.add_argument(
        "--object_store_memory_bytes",
        type=int,
        default=1 * 1024 * 1024 * 1024,
        required=False,
        help="Object store memory for Ray in bytes.",
    )

    model_args = parser.add_argument_group("model", "Model arguments")
    model_args.add_argument(
        "--model_architecture",
        type=str,
        required=True,
        help="Model architecture to use.",
    )
    model_args.add_argument(
        "--loss", type=str, required=True, help="Loss function to use."
    )
    model_args.add_argument(
        "--metrics",
        nargs="*",
        default=["pearson-r"],
        required=False,
        help="Metrics to use for evaluation.",
    )

    search_args = parser.add_argument_group("search", "Hyperparameter search arguments")
    search_args.add_argument(
        "--asha_max_epochs",
        type=int,
        default=40,
        required=False,
        help="Maximum number of epochs to train a trial.",
    )
    search_args.add_argument(
        "--asha_grace_period",
        type=int,
        default=10,
        required=False,
        help="Minimum number of epochs to train a trial before early stopping.",
    )
    search_args.add_argument(
        "--asha_reduction_factor",
        type=int,
        default=3,
        required=False,
        help="Factor to reduce the number of trials by.",
    )
    search_args.add_argument(
        "--asha_brackets",
        type=int,
        default=1,
        required=False,
        help="Number of brackets to use in the ASHA scheduler.",
    )

    search_args.add_argument(
        "--checkpoints_to_keep",
        type=int,
        default=None,
        required=False,
        help="Number of checkpoints to keep per trial.",
    )
    search_args.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1,
        required=False,
        help="Frequency to save checkpoints.",
    )
    search_args.add_argument(
        "--checkpoint_at_end",
        type=bool,
        default=True,
        required=False,
        help="Whether to save a checkpoint at the end of the trial.",
    )

    return parser

def set_experiment_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_dataset_parameters(data_type: str) -> dict:
    if data_type == "sequence":
        dataset_params = {
            "sequence_channels": 4,
            "structure_channels": 0,
            "output_size": 1,
            "output_activation": ["sigmoid"],
        }
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return dataset_params
    
def get_search_space(model_architecture: str) -> dict:
    if model_architecture == "simple_cnn":
        search_space = {
            "in_channels": 4,
            "num_conv_layers": 1,
            "num_filters": hp.choice("conv.num_filters", [16, 32, 64]),
            "kernel_size": hp.choice("conv.kernel_size", [5, 7, 9, 11]),
            "conv_activation": "gelu",
            "mlp_activation": "gelu",
            "batch_norm": True,
            "weight_penalty_l1": 0,
            "weight_penalty_l2": hp.loguniform("weight_penalty_l2", np.log(1e-6), np.log(1e-2)),
            "activity_penalty_l1": hp.loguniform("activity_penalty_l1", np.log(1e-6), np.log(1e-2)),
            "activity_penalty_l2": hp.loguniform("activity_penalty_l2", np.log(1e-6), np.log(1e-2)),
            "dropout": 0.2,
            "output_activation": "sigmoid",
            "batch_size_iter": 1024,  # Don't need gradient accumulation, so just set to highest possible batch size
            "batch_size": hp.loguniform("batch_size", np.log(128), np.log(1024)),
            "init_lr": hp.loguniform(
                "init_lr", np.log(1e-6), np.log(1e-2)
            ),  # Initial learning rate
            "lr_decay_patience": hp.uniformint(
                "lr_decay_patience", 0, 10
            ),  # Number of stalled epochs to wait before decaying the learning rate
            "lr_decay_factor": hp.loguniform(
                "lr_decay_factor", np.log(1e-3), np.log(0.5)
            ),  # Factor to decay the learning rate
        }
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    return search_space

def do_hyperparameter_search(
    experiment_args: dict, search_space: dict
) -> tune.ResultGrid:

    metric_to_optimize = "valid-pearson-logits-r"
    metric_mode = "max"

    # https://docs.ray.io/en/latest/tune/api/schedulers.html#asha-tune-schedulers-ashascheduler
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr="training_iteration",
        max_t=experiment_args["asha_max_epochs"],
        grace_period=experiment_args["asha_grace_period"],
        reduction_factor=experiment_args["asha_reduction_factor"],
        brackets=experiment_args["asha_brackets"],
    )

    hyperopt_search = HyperOptSearch(
        space=experiment_args | search_space,
        metric=metric_to_optimize,
        mode=metric_mode,
        random_state_seed=experiment_args["experiment_seed"],
    )

    tune_config = tune.TuneConfig(
        metric=metric_to_optimize,
        mode=metric_mode,
        search_alg=hyperopt_search,
        scheduler=scheduler,
        num_samples=experiment_args["num_trials"],
        max_concurrent_trials=experiment_args["num_concurrent"],
        time_budget_s=experiment_args["time_budget_s"],
        reuse_actors=False,
        trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
    )

    run_config = tune.RunConfig(
        name=experiment_args["experiment_name"],
        storage_path=experiment_args["result_dir"],
        # failure_config=ray.train.FailureConfig(),
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=experiment_args["checkpoints_to_keep"],
            checkpoint_score_attribute=metric_to_optimize,
            checkpoint_score_order=metric_mode,
            checkpoint_frequency=experiment_args["checkpoint_freq"],
            checkpoint_at_end=experiment_args["checkpoint_at_end"],
        ),
    )

    experiment_with_resources = tune.with_resources(
        trainable=Experiment,
        resources={
            "cpu": experiment_args["num_cpus_per_trial"],
            "gpu": experiment_args["num_gpus_per_trial"] if torch.cuda.is_available() else 0,
        },
    )

    restore_path = os.path.join(
        experiment_args["result_dir"], experiment_args["experiment_name"]
    )
    if experiment_args["restore"] and tune.Tuner.can_restore(restore_path):
        tuner = tune.Tuner.restore(
            path=restore_path,
            trainable=experiment_with_resources,
            resume_unfinished=True,
            resume_errored=True,
        )
    else:
        tuner = tune.Tuner(
            trainable=experiment_with_resources,
            tune_config=tune_config,
            run_config=run_config,
        )

    results = tuner.fit()

    best_result = results.get_best_result(metric_to_optimize, metric_mode, "all")
    print(f"Best trial config: {best_result.config}")

    return results


def main() -> None:
    parser = initialize_argparse()
    args = parser.parse_args()

    ray.init(
        num_cpus=args.num_cpus,
        object_store_memory=args.object_store_memory_bytes,
        include_dashboard=False,
    )

    set_experiment_seed(args.experiment_seed)
    data_parameters = get_dataset_parameters(args.data_type)
    search_space = get_search_space(args.model_architecture)
    search_results = do_hyperparameter_search(
        vars(args) | data_parameters, search_space
    )

    search_results.get_dataframe().to_csv(
        f"{args.result_dir}/{args.experiment_name}-results.csv", index=False
    )

if __name__ == "__main__":
    main()