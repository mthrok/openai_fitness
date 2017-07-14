"""Utility functions to save images"""
from __future__ import division

import numpy as np


def plot_images(images, output):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_images = len(images)
    n_row = np.floor(np.sqrt(n_images))
    n_col = np.ceil(n_images // n_row)

    fig = plt.figure(figsize=(n_row, n_col))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(images):
        ax = fig.add_subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(sample, cmap='Greys_r')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
