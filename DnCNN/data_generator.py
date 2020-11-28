import os
os.chdir('D:\ML_learning\EC523Project')

import glob
from BM3D import noise_model as nm
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class DenoisingDatatset(Dataset):
    # TODO: add more comments to this.
    """
        Generate a dataset with noisy images.

        Args:
            xs (Tensor): TODO: what's the dtype?
            sigma (float): noise level, e.g., 25.
            eta_min (float): the minimum value for SE yield eta, normally 2.0.
            eta_max (float): the maximum value for SE yield eta, normally 8.0.
            dose (float): ion dose, e.g. 20.0.

    """
    def __init__(self, xs, sigma, eta_min, eta_max, dose):
        super().__init__(self)
        self.xs = xs
        self.sigma = sigma
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.dose = dose

    def __getitem__(self, index):
        # Generate compound Poisson-Poisson noisy image on the fly.
        batch_x = self.xs[index]

        batch_y, _ = nm.poissonpoissonnoise(batch_x,
                                            self.eta_min, self.eta_max,
                                            self.dose, t=100)

        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def show(x, title: str = "None", cbar: bool = False, figsize: tuple = None):
    """
    Show images.

    Args:
        x: image to show.
        title: the image title.
        cbar: whether to show colorbar or nor.
        figsize: the figure size.

    Shapes:
        input:
            TODO: the size of the x needs to be checked.
            x: (h, w), where h is the image height and w is the width.
            title: str.
            figsize: (f1, f2) where f1, f2 are the width and height
                     of figure size in inches.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation="nearest", cmap="gray")

    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode: int = 0):
    """
    Augment input image patch by flipping upside down or rotating.

    Args:
        img: the image to be augmented.
        mode: has 8 modes in total from 0 to 7.
              mode = 0: return the original image.
              mode = 1: flip the image upside down.
              mode = 2: rotate the image 90 degrees anticlockwise.
              mode = 3: rotate the image 90 degrees anticlockwise and flip it upside down.
              mode = 4: rotate the image 180 degrees.
              mode = 5: rotate the image 180 degrees anticlockwise and flip upside down.
              mode = 6: rotate the image 270 degrees anticlockwise.
              mode = 7: rotate the image 270 degrees anticlockwise and flip upside down.
    Returns:
        augmented image.

    Shapes:
        Inputs:
            img: (patch_size, patch_size)
            mode: integer.
        Output:
            img_aug: (patch_size, patch_size)
    """
    # data augmentation
    if mode == 0:
        # Do nothing.
        return img
    elif mode == 1:
        # Flip the image upside down.
        return np.flipud(img)
    elif mode == 2:
        # Rotate the image 90 degrees anticlockwise.
        return np.rot90(img)
    elif mode == 3:
        # Rotate the image 90 degrees anticlockwise and flip it upside down.
        return np.flipud(np.rot90(img))
    elif mode == 4:
        # Rotate the image 180 degrees.
        return np.rot90(img, k=2)
    elif mode == 5:
        # Rotate the image 180 degrees anticlockwise and flip upside down.
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        # Rotate the image 270 degrees anticlockwise.
        return np.rot90(img, k=3)
    elif mode == 7:
        # Rotate the image 270 degrees anticlockwise and flip upside down.
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    """
    For input image, load it as a grayscale image.
    Scale the image and extract patches from the scaled image.
    Then augment those image patches using random mode for aug_times.
    Return the augmented image stack.

    Args:
        file_name (str): full image dir.

    Returns:
        patches (ndarray): a list of augmented images.

    Shapes:
        Input:
            file_name: str.
        Output:
            patches: a list of np.uint8 numpy arrays. Each numpy array has shape (patch_size, patch_size).

    """
    img = cv2.imread(file_name, 0)  # read image as grayscale. TODO: check the img dtype
    h, w = img.shape

    patches = []
    for s in scales:
        # Scale the image for every scale level.
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)

        # Extract patches and augment it using random modes.
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i: i + patch_size, j: j + patch_size]
                for k in range(aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


def datagenerator(data_dir="F:\EC523Project\data\Train400", verbose=False):
    """
    Generate clean patches from a dataset.
    Args:
        data_dir (str): directory of the training data.
        verbose (bool): whether to show the progress of image patches generate/augmentation or not.

    Returns:
        data (np.uint8 ndarray): clean patches.

    Shapes:
        Output:
            data: (n, patch_size, patch_size, 1)

    """

    file_list = glob.glob(data_dir + "/*.png")  # get the name list of all .png files.

    # Generate clean augmented patches.
    data = []
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
        if verbose:
            print("Image [{} / {}] is done.".format(i + 1, len(file_list)))

    # Convert list to ndarray.
    data = np.array(data, dtype=np.uint8)

    # TODO: why to add additional dimension?
    data = np.expand_dims(data, axis=3)

    # Because of batch_normalization TODO: why?
    discard_n = len(data) - len(data) // batch_size * batch_size
    data = np.delete(data, range(discard_n), axis=0)

    print("Training data process finished.")
    return data


if __name__ == "__main__":
    data_dir = "F:\EC523Project\data\Train400"
    data = datagenerator(data_dir=data_dir)

