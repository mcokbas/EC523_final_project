import configparser
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from model import DnCNN
import data_generator as dg
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from timeit import default_timer as timer
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import glob
from noise_model import poissonpoissonnoise as nm
import datetime
from vgg import Vgg16
from torchvision import transforms


STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0

# manualSeed = 999
# # manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# np.random.seed(manualSeed)
# torch.manual_seed(manualSeed)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def _tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def save_model(net: nn.Module, model_save_dir, step, dose_total):
    """
    Save the trained model.

    Args:
        net: trained model.
        model_save_dir: saved model directory.
        step: checkpoint.
    """
    model_save_dir = Path(model_save_dir) / "dose{}".format(str(int(dose_total)))
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
    def __init__(self, reduction="sum"):
        super(LossFunc, self).__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
        # TODO: to add TV loss.
        # self.tv_loss =
        # TODO: to add likelihood
        # self.log_loss =

    def forward(self, logits, target):
        # Return the average MSE loss.
        mse_loss = self.mse_loss(logits, target).div_(2)
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

    # Save logs to txt file.
    log_dir = config["DnCNN"]["log_dir"]
    log_dir = Path(log_dir) / "dose{}".format(str(int(dose * 100)))
    log_file = log_dir / "train_result.txt"

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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    criterion = LossFunc(reduction="mean")

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

    # Train the model.
    loss_store = []
    epoch_loss_store = []
    psnr_store = []
    ssim_store = []

    psnr_tr_store = []
    ssim_tr_store = []
    
    loss_mse = torch.nn.MSELoss()

    dtype = torch.cuda.FloatTensor
    # load vgg network
    vgg = Vgg16().type(dtype)
    
    
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
            img_batch_read = len(inputs)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            # We can use labels for both style and content image
            
                # style image
#             style_transform = transforms.Compose([
#             normalize_tensor_transform()      # normalize with ImageNet values
#             ])
            
#             labels_t = style_transform(labels)
                        
            labels_t = labels.repeat(1, 3, 1, 1)
            outputs_t = outputs.repeat(1, 3, 1, 1)            
            
            y_c_features = vgg(labels_t)
            style_gram = [gram(fmap) for fmap in y_c_features]
            
            y_hat_features = vgg(outputs_t)
            y_hat_gram = [gram(fmap) for fmap in y_hat_features]            
            
            # calculate style loss
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss = style_loss

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss = content_loss
            
            loss = aggregate_content_loss + aggregate_style_loss
#             loss = criterion(outputs, labels)
            
            loss_store.append(loss.item())
            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

            if idx % 100 == 0:
                print("Epoch [{} / {}], step [{} / {}], loss = {:.5f}, lr = {:.6f}, elapsed time = {:.2f}s".format(
                    epoch + 1, epochs, idx, len(train_loader), loss.item(), *scheduler.get_last_lr(), timer()-t_start))

        epoch_loss_store.append(epoch_loss / len(train_loader))

        # At each epoch validate the result.
        model = model.eval()

        # # Firstly validate on training sets. This takes a long time so I commented.
        # tr_psnr = []
        # tr_ssim = []
        # # t_start = timer()
        # with torch.no_grad():
        #     for idx, train_data in enumerate(train_loader):
        #         inputs, labels = train_data
        #         # print(inputs.shape)
        #         # inputs = np.expand_dims(inputs, axis=0)
        #         # inputs = torch.from_numpy(inputs).to(device)
        #         inputs = inputs.to(device)
        #         labels = labels.squeeze().numpy()
        #
        #         outputs = model(inputs)
        #         outputs = outputs.squeeze().cpu().detach().numpy()
        #
        #         tr_psnr.append(peak_signal_noise_ratio(labels, outputs))
        #         tr_ssim.append(structural_similarity(outputs, labels))
        # psnr_tr_store.append(sum(tr_psnr) / len(tr_psnr))
        # ssim_tr_store.append(sum(tr_ssim) / len(tr_ssim))
        # # print("Elapsed time = {}".format(timer() - t_start))
        #
        # print("Validation on train set: epoch [{} / {}], aver PSNR = {:.2f}, aver SSIM = {:.4f}".format(
        #     epoch + 1, epochs, psnr_tr_store[-1], ssim_tr_store[-1]))

        # Validate on test set
        val_psnr = []
        val_ssim = []
        with torch.no_grad():
            for idx, test_data in enumerate(xs_test):
                inputs, labels = test_data
                inputs = np.expand_dims(inputs, axis=0)
                inputs = torch.from_numpy(inputs).to(device)
                labels = labels.squeeze()

                outputs = model(inputs)
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
        save_model(model, model_save_dir, epoch, dose * 100)

        # Save the loss and validation PSNR, SSIM.

        if not log_dir.exists():
            Path.mkdir(log_dir)
        with open(log_file, "a+") as fh:
            # fh.write("{} Epoch [{} / {}], loss = {:.6f}, train PSNR = {:.2f}dB, train SSIM = {:.4f}, "
            #          "validation PSNR = {:.2f}dB, validation SSIM = {:.4f}".format(
            #          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),
            #          epoch + 1, epochs, epoch_loss_store[-1],
            #          psnr_tr_store[-1], ssim_tr_store[-1],
            #          psnr_store[-1], ssim_store[-1]))
            fh.write("{} Epoch [{} / {}], loss = {:.6f}, "
                     "validation PSNR = {:.2f}dB, validation SSIM = {:.4f}\n".format(
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),
                     epoch + 1, epochs, epoch_loss_store[-1],
                     psnr_store[-1], ssim_store[-1]))

        # np.savetxt(log_file, np.hstack((epoch + 1, epoch_loss_store[-1], psnr_store[-1], ssim_store[-1])), fmt="%.6f", delimiter=",  ")

        fig, ax = plt.subplots()
        ax.plot(loss_store[-len(train_loader):])
        ax.set_title("Last 1862 losses")
        ax.set_xlabel("iteration")
        fig.show()

    # print("Continue")


if __name__=="__main__":
    config = configparser.ConfigParser()

    config.read("/home/mertcan/Desktop/CS-EC523-A1-DeepLearning/Final_project/DnCNN_poisson/cfg.ini")

    train_model(config)
