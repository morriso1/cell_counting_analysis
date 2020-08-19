# function created by Guillaume Witz

import matplotlib
import numpy as np


def random_cmap():
    np.random.seed(42)
    cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 4))
    # value 0 should just be transparent
    cmap.colors[:, 3] = 0.5
    cmap.colors[0, :] = 1
    cmap.colors[0, 3] = 0

    # if image is a mask, color (last value) should be red
    cmap.colors[-1, 0] = 1
    cmap.colors[-1, 1:3] = 0
    return cmap
