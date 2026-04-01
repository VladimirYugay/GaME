import argparse

import wandb

from src.entities.datasets import get_datasets
from src.entities.game import GaME
from src.utils import io_utils, utils


def run(game, training_datasets, test_datasets, output_path, config):
    """Execute the full training, refinement, and evaluation pipeline.

    Trains on all training datasets sequentially, evaluates the last training
    dataset and all test datasets before and after global refinement, then
    saves a checkpoint.

    Args:
        game: Initialised ``GaME`` instance.
        training_datasets: List of datasets used for training. The last entry
            is also used as the train-split evaluation sequence.
        test_datasets: List of held-out datasets evaluated but not trained on.
        output_path: Root ``Path`` under which per-run outputs are written.
        config: Full config dict; ``config["game"]["refinement_iters"]`` and
            ``config["data"]`` are read here.
    """

    for dataset in training_datasets:
        game.train(dataset, output_path / dataset.run_id)

    # pre ref eval
    game.evaluate(training_datasets[-1], output_path, split="train", refinement_state="pre_ref")
    for dataset in test_datasets:
        game.evaluate(dataset, output_path, split="test", refinement_state="pre_ref")

    # refinement
    game.optimize_model(
        iterations=config["game"]["refinement_iters"], refinement=True)

    # post ref eval
    game.evaluate(training_datasets[-1], output_path, split="train", refinement_state="post_ref")
    for dataset in test_datasets:
        game.evaluate(dataset, output_path, split="test", refinement_state="post_ref")

    # save checkpoint after refinement
    game.save(output_path, config["data"])


def main():
    """Parse arguments, build datasets and model, then run the pipeline."""
    utils.setup_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    config = io_utils.read_yaml_file(args.config_path)
    if args.output_path is not None:
        config["output_path"] = args.output_path
    if args.data_path is not None:
        config["data"]["dataset_path"] = args.data_path
    training_datasets, test_datasets = get_datasets(config["data"])

    # init wandb
    wandb_online = bool(config["wandb"])
    if wandb_online:
        group = config.get("wandb_group")
        wandb.init(project=config["project_name"],
                   config=config,
                   group=group)

    game = GaME(config["game"], wandb_online=wandb_online)

    output_path = io_utils.setup_output_path(config["output_path"])

    run(game, training_datasets, test_datasets, output_path, config)


if __name__ == "__main__":
    main()
