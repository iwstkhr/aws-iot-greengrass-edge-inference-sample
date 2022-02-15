import argparse
import os
from PIL import Image

import numpy as np
import torch
from torchvision import transforms


def load_image_to_tensor(path: str) -> torch.Tensor:
    """ Load an image to PyTorch Tensor.

    Args:
        path (str): An image path

    Returns:
        torch.Tensor: Loaded tensor
    """

    # See https://pytorch.org/vision/stable/models.html#classification
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
    # and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(path)
    tensor_3d = preprocess(img)
    return torch.unsqueeze(tensor_3d, 0)


def save(tensor: torch.Tensor, path: str) -> None:
    """ Save tensor as a numpy array.

    Args:
        tensor (torch.Tensor): Tensor to be saved
        path (str): A path in which tensor is saved
    """

    np.save(path, tensor.numpy())


def parse_args() -> argparse.Namespace:
    """ Get arguments.

    Returns:
        argparse.Namespace: Arguments passed to this script
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    image = args.image

    tensor = load_image_to_tensor(image)
    save(tensor, os.path.basename(image) + '.npy')
