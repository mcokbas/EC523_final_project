import sys
import os
os.chdir('D:\ML_learning\EC523Project')

import numpy as np
from pathlib import Path
from PIL import Image
import glob
import BM3D.noise_model as nm
from BM3D import tr_method
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


path_current = Path.cwd()

dataset_name = ["12_images", "BSD68"]

test_dset_name = "BSD68"
test_images = []
if test_dset_name in dataset_name:
    data_dir = path_current / "data" / test_dset_name

    # Load test images
    for img_path in glob.glob(str(data_dir) + "/*.png"):
        img = np.array(Image.open(img_path)) / 255.0

        test_images.append(img)

else:
    print("Invalid test dataset name")

# Add noise to the clean test images
eta_min = 2
eta_max = 8
dose_rate = 0.1
t = 100
dose_total = dose_rate * t

print("Test on {}, total dose is {}".format(test_dset_name, dose_total))

eat_hat_trml = []
mse_trml = []
ssim_trml = []
psnr_trml = []


save_file_path = Path("F:/EC523Project/data/reconstruct_images") / test_dset_name
if not save_file_path.exists():
    Path.mkdir(save_file_path)

for i, img in enumerate(test_images):
    img_noisy, img_noisy_tr = nm.poissonpoissonnoise(img, eta_min, eta_max, dose_rate, t)

    img_noisy_tr = img_noisy_tr.reshape(img.size, t)
    img_trml = tr_method.trml_estimate(img_noisy_tr, dose_rate)

    # Rescale the image to [0, 1].
    img_trml /= img_trml.max()
    img_trml = img_trml.reshape(img.shape)

    # img_to_save = Image.fromarray(img_trml)
    #
    # img_to_save.save(save_image_path / f"{i+1}.png")

    eat_hat_trml.append(img_trml)


    # Compute the MSE and SSIM.
    mse_trml.append(mean_squared_error(img_trml, img))

    ssim_trml.append(ssim(img_trml, img, data_range=1))

    psnr_trml.append(20 * np.log10(1 / np.sqrt(mse_trml[-1])))


    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img_noisy, cmap="gray")
    # ax[1].imshow(img_trml, cmap="gray")
    # fig.show()

print("The average MSE of TRML method in {} dataset is: {:.3f}".format(test_dset_name,
                                                                       sum(mse_trml) / len(mse_trml)))
print("The average SSIM of TRML method in {} dataset is: {:.3f}".format(test_dset_name,
                                                                        sum(ssim_trml) / len(ssim_trml)))

f_name = "total_dose_{:d}.npz".format(int(dose_total))
print(f_name)

data_to_save = {"dose_rate": dose_rate, "dose_total": dose_total, 't': t,
                "eta_min": eta_min, "eta_max": eta_max,
                "eta_hat_trml": eat_hat_trml,
                "mse_trml": mse_trml,
                "ssim_trml": ssim_trml,
                "psnr_trml": psnr_trml}
np.savez(save_file_path / f_name, **data_to_save)


# data = np.load(save_file_path / f_name)





pass





pass


