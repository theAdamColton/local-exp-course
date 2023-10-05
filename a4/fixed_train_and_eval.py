"""
This is the same as the original, but it moves the lm.classes() to the same device to prevent a pytorch error.
"""
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
import torch

from captum.concept._utils.classifier import _train_test_split

def train_and_eval(
    self, dataloader: DataLoader, test_split_ratio: float = 0.33, **kwargs: Any
) -> Union[Dict, None]:
    r"""
     Implements Classifier::train_and_eval abstract method for small concept
     datsets provided by `dataloader`.
     It is assumed that when iterating over `dataloader` we can still
     retain the entire dataset in the memory.
     This method shuffles all examples randomly provided, splits them
     into train and test partitions and trains an SGDClassifier using sklearn
     library. Ultimately, it measures and returns model accuracy using test
     split of the dataset.

    Args:
        dataloader (dataloader): A dataloader that enables batch-wise access to
                the inputs and corresponding labels. Dataloader allows us to
                iterate over the dataset by loading the batches in lazy manner.
        test_split_ratio (float): The ratio of test split in the entire dataset
                served by input data loader `dataloader`.

                Default: 0.33
    Returns:
        stats (dict): a dictionary of statistics about the performance of the model.
                In this case stats represents a dictionary of model accuracy
                measured on the test split of the dataset.

    """
    inputs = []
    labels = []
    for input, label in dataloader:
        inputs.append(input)
        labels.append(label)

    device = "cpu" if input is None else input.device

    x_train, x_test, y_train, y_test = _train_test_split(
        torch.cat(inputs), torch.cat(labels), test_split=test_split_ratio
    )
    self.lm.device = device
    self.lm.fit(DataLoader(TensorDataset(x_train, y_train)))

    predict = self.lm(x_test)

    predict = self.lm.classes()[torch.argmax(predict, dim=1).cpu()]  # type: ignore
    score = predict.long() == y_test.long().cpu()

    accs = score.float().mean()

    return {"accs": accs}

