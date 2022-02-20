import argparse
import glob
import json
import os
import time

import numpy as np
from dlr import DLRModel


def load_model() -> DLRModel:
    """ Load a DLR model compiled from a PyTorch VGG16 pretrained model.

    Returns:
        DLRModel: A DLR Model
    """

    return DLRModel('/greengrass/v2/work/vgg16-component')


def load_labels() -> dict:
    """ Load class labels from the ImageNet json.

    Returns:
        dict: Dictionary which contains class labels
    """

    path = os.path.dirname(os.path.abspath(__file__))
    # See https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
    path = os.path.join(path, 'imagenet_class_index.json')
    with open(path, 'r') as f:
        labels = json.load(f)
    return labels


def iter_files(path: str) -> str:
    """ Iterate files with a `.npy` extension in the specified path.

    Args:
        path (str): A directory path

    Yields:
        str: A file path
    """

    path = path[:-1] if path.endswith('/') else path
    files = glob.glob(f'{path}/*.npy')
    for file in files:
        yield file


def predict(model: DLRModel, image: np.ndarray) -> np.ndarray:
    """ Predict using a DLR model.

    Args:
        model (DLRModel): A DLR Model
        image (np.ndarray): numpy array to be predicted
                            It is expected to be normalized in advance.<br/>
                            See https://pytorch.org/vision/stable/models.html#classification

    Returns:
        np.ndarray: A prediction result
    """

    return model.run(image)[0]


def parse_args() -> argparse.Namespace:
    """ Get arguments.

    Returns:
        argparse.Namespace: Arguments passed to this script
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--interval', type=int, default=300)
    args, _ = parser.parse_known_args()
    return args


def start(model: DLRModel, path: str, labels: dict) -> None:
    """ Start a prediction process.

    Args:
        model (DLRModel): A Model to be used for prediction
        path (str): A directory path which contains target images
        labels (dict): Dictionary which contains class labels
    """

    for file in iter_files(path):
        image = np.load(file)
        y = predict(model, image)
        index = int(np.argmax(y))
        label = labels.get(str(index), '')
        print(f'Prediction result of {file}: {label}')


if __name__ == '__main__':
    args = parse_args()
    print(f'args: {args}')
    model = load_model()
    labels = load_labels()

    if args.interval == 0:
        start(model, args.test_dir, labels)
    else:
        while True:
            start(model, args.test_dir, labels)
            print(f'Sleep in {args.interval} seconds...')
            time.sleep(args.interval)
