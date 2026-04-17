import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)