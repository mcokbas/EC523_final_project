import configparser
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from DnCNN.model import DnCNN
import DnCNN.data_generator as dg
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from timeit import default_timer as timer
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import glob
from BM3D.noise_model import poissonpoissonnoise as nm
import datetime


# manualSeed = 999
# # manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# np.random.seed(manualSeed)
# torch.manual_seed(manualSeed)

def save_model(net: nn.Module, model_save_dir, step, dose_total, train_mode):
    """
    Save the trained model.

    Args:
        net: trained model.
        model_save_dir: saved model directory.
        step: checkpoint.
    """
    if train_mode == "residual":
        model_save_dir = Path(model_save_dir) / "res_learning_smaller_lr_dose{}".format(str(int(dose_total)))
    elif train_mode == "direct":
        model_save_dir = Path(model_save_dir) / "direct_predict_smaller_lr_dose{}".format(str(int(dose_total)))

    if not Path(model_save_dir).exists():
        Path.mkdir(model_save_dir)
    model_path = Path(model_save_dir) / "{}.pth".format(step + 1)

    torch.save(net.state_dict(), model_path)

    print("Saved model checkpoints {} into {}".format(step + 1, model_save_dir))

def restore_model(resume_iters, model_save_dir, net: nn.Module, train=True):
    """
    Restore the trained model.

    Args:
        resume_iters: the iteration to be loaded.
        model_save_dir: the directory for saving the model.
        net: the model instance to be loaded.
        train: if True, then the model is set to training;
               else set it to test.

    Returns:
        net: loaded model instance.

    """
    print("Loading the trained model from step {}".format(resume_iters))
    model_path = Path(model_save_dir) / "{}.pth".format(resume_iters)

    # Restore the model.
    net.load_state_dict(torch.load(model_path))

    if train:
        net.train()
    else:
        net.eval()

    return net


class LossFunc(nn.Module):
    def __init__(self, reduction="sum", weight_mse=0, weight_tv=0, weight_nll=1, total_dose=20.0):
        super(LossFunc, self).__init__()
        self.reduction = reduction
        self.total_dose = total_dose

        # Define the loss function forms.
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.weight_mse = weight_mse
        # TODO: to add TV loss.
        # self.tv_loss =
        self.weight_tv = weight_tv
        # TODO: to add likelihood
        # self.log_loss = self._nll_loss(x, img_noisy)
        self.weight_nll = weight_nll

    def _nll_loss(self, eta, y, eta_min, eta_max, y_max):
        # mask = eta >= 1e-4
        # loss = torch.log(eta[mask]) + \
        #        torch.log(eta[mask] / self.total_dose + 1) + \
        #        (y[mask] - eta[mask]) ** 2 / (eta[mask]) / (eta[mask] / self.total_dose + 1)

        # # If the eta is zero, add a very small value to it so that it wouldn't encounter nan.
        # loss = torch.log(eta + 1e-7) + \
        #        torch.log(eta / self.total_dose + 1) + \
        #        (y - eta) ** 2 / (eta + 1e-7) / (eta / self.total_dose + 1)

        # eta_scaled = eta * (eta_max - eta_min) + eta_min
        loss = 0.0

        return torch.sum(loss).div_(2).div_(y.size(0))

    def forward(self, logits, target):
        # Return the average MSE loss.
        mse_loss = self.weight_mse * self.mse_loss(logits, target).div_(2)
        # nll_loss = self.weight_nll * self._nll_loss(logits, target)
        # loss = mse_loss + nll_loss
        loss = mse_loss
        return loss
    

def train_model(config):
    # Define hyper-parameters.
    depth = int(config["DnCNN"]["depth"])
    n_channels = int(config["DnCNN"]["n_channels"])
    img_channel = int(config["DnCNN"]["img_channel"])
    kernel_size = int(config["DnCNN"]["kernel_size"])
    use_bnorm = config.getboolean("DnCNN", "use_bnorm")
    epochs = int(config["DnCNN"]["epoch"])
    batch_size = int(config["DnCNN"]["batch_size"])
    train_data_dir = config["DnCNN"]["train_data_dir"]
    test_data_dir = config["DnCNN"]["test_data_dir"]
    eta_min = float(config["DnCNN"]["eta_min"])
    eta_max = float(config["DnCNN"]["eta_max"])
    dose = float(config["DnCNN"]["dose"])
    model_save_dir = config["DnCNN"]["model_save_dir"]
    train_mode = config["DnCNN"]["train_mode"]
    log_file_name = config["DnCNN"]["log_file_name"]

    # Save logs to txt file.
    log_dir = config["DnCNN"]["log_dir"]
    if train_mode == "residual":
        log_dir = Path(log_dir) / "res_learning_smaller_lr_dose{}".format(str(int(dose * 100)))
    elif train_mode == "direct":
        log_dir = Path(log_dir) / "direct_predict_smaller_lr_dose{}".format(str(int(dose * 100)))
    log_file = log_dir / log_file_name

    # Define device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initiate a DnCNN instance.
    # Load the model to device and set the model to training.
    model = DnCNN(depth=depth, n_channels=n_channels,
                  img_channel=img_channel,
                  use_bnorm=use_bnorm,
                  kernel_size=kernel_size)

    model = model.to(device)
    model.train()

    # Define loss criterion and optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
    criterion = LossFunc(reduction="mean", weight_mse=1, weight_nll=0, total_dose=dose * 100)

    # Get a validation test set and corrupt with noise for validation performance.
    # For every epoch, use this pre-determined noisy images.
    test_file_list = glob.glob(test_data_dir + "/*.png")
    xs_test = []
    # Can't directly convert the xs_test from list to ndarray because some images are 512*512
    # while the rest are 256*256.
    for i in range(len(test_file_list)):
        img = cv2.imread(test_file_list[i], 0)
        img = np.array(img, dtype="float32") / 255.0
        img = np.expand_dims(img, axis=0)
        img_noisy, _ = nm(img, eta_min, eta_max, dose, t=100)
        xs_test.append((img_noisy, img))
    
    # Get a validation train set and corrupt with noise.
    # For every epoch, use this pre-determined noisy images to see the training performance.
    train_file_list = glob.glob(train_data_dir + "/*png")
    xs_train = []
    for i in range(len(train_file_list)):
        img = cv2.imread(train_file_list[i], 0)
        img = np.array(img, dtype="float32") / 255.0
        img = np.expand_dims(img, axis=0)
        img_noisy, _ = nm(img, eta_min, eta_max, dose, t=100)
        xs_train.append((img_noisy, img))
    
    # Train the model.
    loss_store = []
    epoch_loss_store = []
    psnr_store = []
    ssim_store = []

    psnr_tr_store = []
    ssim_tr_store = []
    for epoch in range(epochs):
        # For each epoch, generate clean augmented patches from the training directory.
        # Convert the data from uint8 to float32 then scale them to make it in [0, 1].
        # Then make the patches to be of shape [N, C, H, W],
        # where N is the batch size, C is the number of color channels.
        # H and W are height and width of image patches.
        xs = dg.datagenerator(data_dir=train_data_dir)
        xs = xs.astype("float32") / 255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))

        train_set = dg.DenoisingDatatset(xs, eta_min, eta_max, dose)
        train_loader = DataLoader(dataset=train_set, num_workers=4,
                                  drop_last=True, batch_size=batch_size,
                                  shuffle=True)  # TODO: if drop_last=True, the dropping in the
                                                 # TODO: data_generator is not necessary?

        # train_loader_test = next(iter(train_loader))

        t_start = timer()
        epoch_loss = 0
        for idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, mode=train_mode)

            loss = criterion(outputs, labels)

            loss_store.append(loss.item())
            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

            if idx % 100 == 0:
                print("Epoch [{} / {}], step [{} / {}], loss = {:.5f}, lr = {:.8f}, elapsed time = {:.2f}s".format(
                    epoch + 1, epochs, idx, len(train_loader), loss.item(), *scheduler.get_last_lr(), timer()-t_start))

        epoch_loss_store.append(epoch_loss / len(train_loader))

        # At each epoch validate the result.
        model = model.eval()

        # Firstly validate on training sets. This takes a long time so I commented.
        tr_psnr = []
        tr_ssim = []
        # t_start = timer()
        with torch.no_grad():
            for idx, test_data in enumerate(xs_train):
                inputs, labels = test_data
                inputs = np.expand_dims(inputs, axis=0)
                inputs = torch.from_numpy(inputs).to(device)
                labels = labels.squeeze()

                outputs = model(inputs, mode=train_mode)
                outputs = outputs.squeeze().cpu().detach().numpy()

                tr_psnr.append(peak_signal_noise_ratio(labels, outputs))
                tr_ssim.append(structural_similarity(outputs, labels))
        psnr_tr_store.append(sum(tr_psnr) / len(tr_psnr))
        ssim_tr_store.append(sum(tr_ssim) / len(tr_ssim))
        # print("Elapsed time = {}".format(timer() - t_start))

        print("Validation on train set: epoch [{} / {}], aver PSNR = {:.2f}, aver SSIM = {:.4f}".format(
            epoch + 1, epochs, psnr_tr_store[-1], ssim_tr_store[-1]))

        # Validate on test set
        val_psnr = []
        val_ssim = []
        with torch.no_grad():
            for idx, test_data in enumerate(xs_test):
                inputs, labels = test_data
                inputs = np.expand_dims(inputs, axis=0)
                inputs = torch.from_numpy(inputs).to(device)
                labels = labels.squeeze()

                outputs = model(inputs, mode=train_mode)
                outputs = outputs.squeeze().cpu().detach().numpy()

                val_psnr.append(peak_signal_noise_ratio(labels, outputs))
                val_ssim.append(structural_similarity(outputs, labels))

        psnr_store.append(sum(val_psnr) / len(val_psnr))
        ssim_store.append(sum(val_ssim) / len(val_ssim))

        print("Validation on test set: epoch [{} / {}], aver PSNR = {:.2f}, aver SSIM = {:.4f}".format(
            epoch + 1, epochs, psnr_store[-1], ssim_store[-1]))

        # Set model to train mode again.
        model = model.train()

        scheduler.step()

        # Save model
        save_model(model, model_save_dir, epoch, dose * 100, train_mode)

        # Save the loss and validation PSNR, SSIM.

        if not log_dir.exists():
            Path.mkdir(log_dir)
        with open(log_file, "a+") as fh:
            fh.write("{} Epoch [{} / {}], loss = {:.6f}, train PSNR = {:.2f}dB, train SSIM = {:.4f}, "
                     "validation PSNR = {:.2f}dB, validation SSIM = {:.4f}\n".format(
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),
                     epoch + 1, epochs, epoch_loss_store[-1],
                     psnr_tr_store[-1], ssim_tr_store[-1],
                     psnr_store[-1], ssim_store[-1]))
            

if __name__=="__main__":
    config = configparser.ConfigParser()

    config.read("D:\ML_learning\EC523Project\DnCNN\cfg.ini")

    train_model(config)
