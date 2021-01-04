import torch
import numpy as np
from typing import List, Tuple, Union
from collections import deque
import logging
import scipy.signal

logger = logging.getLogger("GroundedSCAN_learning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_mask(sequence_lengths: torch.LongTensor, max_len=None) -> torch.tensor:
    """
    Create a sequence mask that masks out all indices larger than some sequence length as defined by
    sequence_lengths entries.

    :param sequence_lengths: [batch_size] sequence lengths per example in batch
    :param max_len: int defining the maximum sequence length in the batch
    :return: [batch_size, max_len] boolean mask
    """
    if max_len is None:
        max_len = sequence_lengths.data.max()
    batch_size = sequence_lengths.size(0)
    sequence_range = torch.arange(0, max_len).long().to(device=device)

    # [batch_size, max_len]
    sequence_range_expand = sequence_range.unsqueeze(0).expand(batch_size, max_len)

    # [batch_size, max_len]
    seq_length_expand = (sequence_lengths.unsqueeze(1).expand_as(sequence_range_expand))

    # [batch_size, max_len](boolean array of which elements to include)
    return sequence_range_expand < seq_length_expand


def log_parameters(model: torch.nn.Module) -> {}:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Total parameters: %d" % n_params)
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("%s : %s" % (name, list(p.size())))


def sequence_accuracy(prediction: List[int], target: List[int]) -> float:
    correct = 0
    total = 0
    prediction = prediction.copy()
    target = target.copy()
    if len(prediction) < len(target):
        difference = len(target) - len(prediction)
        prediction.extend([0] * difference)
    if len(target) < len(prediction):
        difference = len(prediction) - len(target)
        target.extend([-1] * difference)
    for i, target_int in enumerate(target):
        if i >= len(prediction):
            break
        prediction_int = prediction[i]
        if prediction_int == target_int:
            correct += 1
        total += 1
    if not total:
        return 0.
    return (correct / total) * 100


def pack_state_tensors(state: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.tensor, List[int], torch.tensor]:
    """Unpack the state into tensors."""
    input_command = state[0]
    world_state = state[1]
    input_command_t = torch.tensor(input_command, dtype=torch.long, device=device).unsqueeze(0)
    input_command_lengths = [len(input_command)]
    world_state_t = torch.tensor(world_state, dtype=torch.float32, device=device).unsqueeze(0)
    return input_command_t, input_command_lengths, world_state_t


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.
    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
