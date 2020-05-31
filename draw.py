import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def torch_to_np(img_torch):
    return img_torch.detach().cpu().numpy()[0]


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    """
    Draws images in a grid
    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.savefig(output_path + "{}.png".format(name))


def get_image_grid(images_np, nrow=2):
    """
    从一个图像列表中创建一个网格，并连接起来。
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()

