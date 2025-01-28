"""Copy this file to the root directory of the repository to run."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import wandb

from src import NeuralTester, ExperimentConfig
from src.optimizer import PymooLearner
from src.defaults.learner_configs import PYMOO_AGE_MOEA_DEFAULT_PARAMS
from src.defaults.objective_configs import (
    TARGETED_BOUNDARY_TESTING,
    UNTARGETED_BOUNDARY_TESTING,
    TARGETED_ADVERSARIAL_TESTING,
    UNTARGETED_ADVERSARIAL_TESTING,
    ADVERSARIAL_BOUNDARY_TESTING,
    DIVERSITY_SAMPLING,
)
from src.manipulator import StyleGANManipulator
from models import load_stylegan


"""Some dicts to easily associate elements to arguments."""
OBJECTIVES = {
    "ubt": UNTARGETED_BOUNDARY_TESTING,
    "tbt": TARGETED_BOUNDARY_TESTING,
    "abt": ADVERSARIAL_BOUNDARY_TESTING,
    "uat": UNTARGETED_ADVERSARIAL_TESTING,
    "tat": TARGETED_ADVERSARIAL_TESTING,
    "ds": DIVERSITY_SAMPLING,
}

MODEL_COMBINATIONS = {
    "MNIST": {
        "wrn": "../models/predictors/wrn_mnist.pkl",
        "vit": NotImplementedError("Model not implemented."),
        "sg2": "../models/generators/sg2_mnist.pkl",
        "sg3": "../models/generators/sg3_mnist.pkl",
        "sgXL": NotImplementedError("Model not implemented."),
    },
    "CIFAR10": {
        "wrn": "../models/predictors/wrn_cifar10.pkl",
        "vit": NotImplementedError("Model not implemented."),
        "sg2": "../models/generators/sg2_cifar10.pkl",
        "sg3": NotImplementedError("Model not implemented."),
        "sgXL": "../models/generators/sgXL_cifar10.pkl",
    },
    "FashionMNIST": {
        "wrn": "../models/predictors/wrn_fashion_mnist.pkl",
        "vit": NotImplementedError("Model not implemented."),
        "sg2": "../models/generators/sg2_fmnist.pkl",
        "sg3": NotImplementedError("Model not implemented."),
        "sgXL": NotImplementedError("Model not implemented."),
    },
    "SVHN": {
        "wrn": "../models/predictors/wrn_svhn.pkl",
        "vit": NotImplementedError("Model not implemented."),
        "sg2": "../models/generators/sg2_svhn.pkl",
        "sg3": NotImplementedError("Model not implemented."),
        "sgXL": NotImplementedError("Model not implemented."),
    },
    "Imagenet": {
        "wrn": NotImplementedError("Model not implemented."),
        "vit": NotImplementedError("Model not implemented."),
        "sg2": NotImplementedError("Model not implemented."),
        "sg3": NotImplementedError("Model not implemented."),
        "sgXL": "../models/generators/sgXL_imagenet.pkl",
    },
}


def main(*,
         objective: str,
         dataset: str,
         predictor: str,
         generator: str,
         mix_dims: tuple[int, int],
         generations: int = 150,
         interpolate: bool = True,
         frontier_pairs: bool = False,
         ) -> None:
    """
    Run the experiments done in the paper.

    :param objective: The objective for optimization.
    :param dataset: The dataset for to test with.
    :param predictor: The predictor to test.
    :param generator: The generator to generate test cases.
    :param mix_dims: The dimensions to mix in the generator.
    :param generations: The number of generations to run the optimization.
    :param interpolate: Whether to interpolate the style layers.
    :param frontier_pairs: Whether to use the frontier pairs.
    """

    # Define the configurations for our experiments.
    metrics = OBJECTIVES[objective]
    p = MODEL_COMBINATIONS[dataset][predictor]
    g = MODEL_COMBINATIONS[dataset][generator]

    device = torch.device("cuda")

    """Initialize components of the framework."""
    sut = torch.load(p)
    sut = sut.to(device)
    sut.eval()

    generator_model = load_stylegan(g)
    generator_model = generator_model.to(device)
    manipulator = StyleGANManipulator(generator_model, device, mix_dims, interpolate=interpolate)

    optimizer_params = PYMOO_AGE_MOEA_DEFAULT_PARAMS
    optimizer = PymooLearner(
        **optimizer_params,
        n_var=mix_dims[1] - mix_dims[0],
        num_objectives=len(metrics),
    )

    """Make a config (used for logging and more)."""
    conf = ExperimentConfig(
        samples_per_class=10,
        generations=generations,
        classes=10,
        save_to=f"results_lmt_{dataset}_{predictor}_{generator}_{objective}" + ("sm" if not interpolate else ""),
    )

    """Initialize the framework with all components."""
    tester = NeuralTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=metrics,
        frontier_pairs=frontier_pairs,
        silent_wandb=True,
        config=conf,
    )

    # We start the testing procedure.
    tester.test(validity_domain=validity_domain)


def _int_tuple(s: str) -> tuple[int, ...]:
    """
    Get a tuple of ints from argument string.

    :param s: The string.
    :return: A tuple of ints.
    """
    return tuple(map(int, s.split(",")))


if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser("Start a latent manipulation testing experiment.")
    parser.add_argument(
        "-ds",
        "--dataset",
        help="Dataset used for testing.",
        type=str,
        choices=["MNIST", "CIFAR10", "FashionMNIST", "SVHN", "Imagenet"],
    )
    parser.add_argument(
        "-g",
        "--generator",
        help="Generator used for testing.",
        type=str,
        choices=["sg2", "sg3", "sgXL"],
    )
    parser.add_argument(
        "-p", "--predictor", help="Predictor used for testing.", type=str, choices=["wrn", "vit"]
    )
    parser.add_argument(
        "-o",
        "--objective",
        help="Objective of the testing",
        type=str,
        choices=["ubt", "tbt", "uat", "tat", "ds"],
    )
    parser.add_argument(
        "--generations", help="Number of generations to run for.", type=int, default=150
    )
    parser.add_argument("--mix-dims", help="Mixing dimensions to use.", type=_int_tuple)
    args = parser.parse_args()

    main(
        objective=args.objective,
        generator=args.generator,
        predictor=args.predictor,
        dataset=args.dataset,
        generations=args.generations,
        mix_dims=args.mix_dims,
    )
