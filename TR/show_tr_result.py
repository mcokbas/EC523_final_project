import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

save_file_path = Path("F:/EC523Project/data/reconstruct_images/BSD68")
f_name = "total_dose_10.npz"


data = np.load(save_file_path / f_name, allow_pickle=True)
mse_trml = data["mse_trml"]
ssim_trml = data["ssim_trml"]
psnr_trml = data["psnr_trml"]

# Print the MSE, SSIM and PSNR metrics of the image.
print(sum(mse_trml) / len(mse_trml))
print(sum(ssim_trml) / len(ssim_trml))
print(sum(psnr_trml) / len(psnr_trml))

# Rescale and save reconstructed image to png.
img_array = data["eta_hat_trml"]
# #Rescale to 0-255 and convert to uint8
rescaled = (255.0 / img_array[0].max() * (img_array[0] - img_array[0].min())).astype(np.uint8)

# im = Image.fromarray(rescaled)
# im.save('test.png')


from skimage.metrics import mean_squared_error

gt = np.array(Image.open(Path('D:/ML_learning/EC523Project/data/12_images/01.png'))) / 255.0
print(mean_squared_error(gt, img_array[0]))


# lam = 20
# eta = np.linspace(0.1, 10, 1000, endpoint=True, dtype=np.float64)
# y = np.arange(500)
#
# eta_eta, yy = np.meshgrid(eta, y)
#
#
#
# log_p = - 1/2 * np.log(eta_eta) - 1/2 * np.log(eta_eta + 1) \
#         - (yy - lam * eta_eta) ** 2 / 2 / lam / eta_eta / (eta_eta + 1)
# d_log_p = - 1/2 / eta_eta - 1/2 / (eta_eta + 1) + (yy - lam * eta_eta) / eta_eta / (eta_eta + 1) \
#           + (yy - lam * eta_eta) ** 2 / 2 / lam / eta_eta / (eta_eta + 1) ** 2 \
#           + (yy - lam * eta_eta) ** 2 / 2 / lam / eta_eta ** 2 / (eta_eta + 1)
# dd_log_p = 1/2 / eta_eta ** 2 + 1/2 / (eta_eta + 1) ** 2 - lam / eta_eta / (eta_eta + 1) \
#            - 2 * (yy - lam * eta_eta) / eta_eta / (eta_eta + 1) ** 2 \
#            - 2 * (yy - lam * eta_eta) / eta_eta ** 2 / (eta_eta + 1) \
#            - (yy - lam * eta_eta) ** 2 / lam / eta_eta / (eta_eta + 1) ** 3 \
#            - (yy - lam * eta_eta) ** 2 - lam / eta_eta ** 2 / (eta_eta + 1) ** 2 \
#            - (yy - lam * eta_eta) ** 2 / lam / eta_eta ** 3 / (eta_eta + 1)
#
# print("The maximum second derivative is: {:.3f}".format(np.max(dd_log_p)))
#
# fig, ax = plt.subplots()
# ax.plot(eta, log_p, label="log likelihood")
# ax.plot(eta, d_log_p, label="derivative")
# ax.plot(eta, dd_log_p, label="Hessian")
# ax.legend()
# ax.set_xlabel(r'$\eta$')
# fig.show()
# print("the end")
#
# fig1, ax1 = plt.subplots()
# ax1.plot(eta, np.log(eta), label="log(x)")
# ax1.plot(eta, eta, label="x")
# ax1.plot(eta, np.log(np.exp(eta) + 1), label="log(exp(x) + 1)")
# ax1.legend()
# fig1.show()

