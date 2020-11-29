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


def save_model(net: nn.Module, model_save_dir, step):
    """
    Save the trained model.

    Args:
        net: trained model.
        model_save_dir: saved model directory.
        step: checkpoint.
    """
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


def train_model(config):
    # Define hyper-parameters.
    depth = int(config["DnCNN"]["depth"])
    n_channels = int(config["DnCNN"]["n_channels"])
    img_channel = int(config["DnCNN"]["img_channel"])
    kernel_size = int(config["DnCNN"]["kernel_size"])
    use_bnorm = config.getboolean("DnCNN", "use_bnorm")
    epochs = int(config["DnCNN"]["epoch"])
    batch_size = int(config["DnCNN"]["batch_size"])
    data_dir = config["DnCNN"]["data_dir"]
    eta_min = float(config["DnCNN"]["eta_min"])
    eta_max = float(config["DnCNN"]["eta_max"])
    dose = float(config["DnCNN"]["dose"])

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
    optimizer = optim.Adam(model.parameters(), )
    criterion = nn.MSELoss(reduction="sum")

    # Train the model.
    for epoch in range(epochs):
        # For each epoch, generate clean augmented patches from the training directory.
        # Convert the data from uint8 to float32 then scale them to make it in [0, 1].
        # Then make the patches to be of shape [N, C, H, W],
        # where N is the batch size, C is the number of color channels.
        # H and W are height and width of image patches.
        xs = dg.datagenerator(data_dir=data_dir)
        xs = xs.astype("float32") / 255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))

        train_set = dg.DenoisingDatatset(xs, eta_min, eta_max, dose)
        train_loader = DataLoader(dataset=train_set, num_workers=4,
                                  drop_last=True, batch_size=batch_size,
                                  shuffle=True)  # TODO: if drop_last=True, the dropping in the
                                                 # TODO: data_generator is not necessary?

        train_loader_test = next(iter(train_loader))

        for idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # loss = criterion(outputs, labels) / 2



if __name__=="__main__":
    config = configparser.ConfigParser()

    config.read("D:\ML_learning\EC523Project\DnCNN\cfg.ini")

    train_model(config)