import argparse
import os
from datetime import datetime

import torch
from torchvision import models


def fit(model: torch.nn.modules.Module) -> None:
    # Write some training codes...
    pass


def save(model: torch.nn.modules.Module, path: str) -> None:
    """ Save a model.

    Args:
        model (torch.nn.modules.Module): A model to be saved
        path: A directory in which a model is saved
    """

    suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(path, f'model-{suffix}.pt')
    # If you use `model.state_dict()`, SageMaker compilation will fail.
    torch.save(model, path)


def parse_args() -> argparse.Namespace:
    """ Get arguments

    Mainly used to get the following arguments. <br/>
    See https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script

    Returns:
         argparse.Namespace: Arguments
    """

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    vgg16 = models.vgg16(pretrained=True)
    fit(vgg16)
    save(vgg16, args.sm_model_dir)
