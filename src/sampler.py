from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class CustomDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass
