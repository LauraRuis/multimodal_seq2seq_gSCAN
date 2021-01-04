import logging
import torch

logger = logging.getLogger("GroundedSCAN_learning")
use_cuda = True if torch.cuda.is_available() else False


def rl_train():
    pass